"""The training contract on a small model: reductions, projection and packing.

Normalization layers accumulate in single precision, so equalities hold to single-precision rounding.
"""
from fractions import Fraction

import pytest
import torch
from torch.nn import functional as F

from finllm.execution import (
    all_position_loss, hidden_states, label_restricted_loss, packed_hidden, segment_allowed, selected_logits,
    selected_position_loss,
)
from finllm.model import FinLLM, ModelConfig
from finllm.objective import (
    cell_loss, family_view_weights, global_denominators, local_mean_example_weights, local_mean_loss,
    local_mean_term_weight, presence_probability,
)

WEIGHTS = {"language": 1.0, "stance": 0.5}


@pytest.fixture()
def model():
    torch.manual_seed(7)
    return FinLLM(ModelConfig(vocab_size=97, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                              num_key_value_heads=2, intermediate_size=128, max_position_embeddings=64)).double().eval()


def gradient(model):
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()]).clone()


def sums(model, tokens, kinds):
    logits = model(tokens)
    losses = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), tokens[:, 1:].reshape(-1),
                             reduction="none").view(tokens.shape[0], -1)
    total, counts = {}, {}
    for row, kind in enumerate(kinds):
        chosen = losses[row] if kind == "language" else losses[row, -1:]
        total[kind] = total.get(kind, 0) + chosen.sum()
        counts[kind] = counts.get(kind, 0) + chosen.numel()
    return total, counts


def test_global_denominators_make_the_gradient_independent_of_the_split(model):
    tokens = torch.randint(0, 97, (8, 16), generator=torch.Generator().manual_seed(1))
    kinds = ["language"] * 6 + ["stance"] * 2
    model.zero_grad()
    whole, counts = sums(model, tokens, kinds)
    cell_loss(whole, counts, WEIGHTS, ranks=1).backward()
    reference = gradient(model)

    for layout in ([[[0, 1], [2, 3]], [[4, 5], [6, 7]]],            # two ranks, two microbatches
                   [[[0, 6, 7]], [[1]], [[2, 3, 4]], [[5]]],        # four ranks of unequal size
                   [[[i] for i in range(8)]]):                      # one rank, eight microbatches
        cells = [rows for rank in layout for rows in rank]
        denominators = global_denominators([sums(model, tokens[r], [kinds[i] for i in r])[1] for r in cells])
        assert denominators == counts
        total = None
        for rank in layout:
            model.zero_grad()
            for rows in rank:
                part, _ = sums(model, tokens[rows], [kinds[i] for i in rows])
                cell_loss(part, denominators, WEIGHTS, ranks=len(layout)).backward()
            total = gradient(model) if total is None else total + gradient(model)
        assert torch.allclose(total / len(layout), reference, rtol=1e-4, atol=1e-6)


def test_per_cell_averaging_changes_the_objective(model):
    tokens = torch.randint(0, 97, (8, 16), generator=torch.Generator().manual_seed(1))
    kinds = ["language"] * 6 + ["stance"] * 2
    model.zero_grad()
    whole, counts = sums(model, tokens, kinds)
    cell_loss(whole, counts, WEIGHTS, ranks=1).backward()
    reference = gradient(model)
    cells = [[0, 1], [2, 3], [4, 5], [6, 7]]
    model.zero_grad()
    for rows in cells:
        part, local = sums(model, tokens[rows], [kinds[i] for i in rows])
        (local_mean_loss(part, local, WEIGHTS) / len(cells)).backward()
    assert (gradient(model) - reference).norm() / reference.norm() > 0.05


def test_a_term_absent_from_cells_loses_weight_under_per_cell_averaging():
    cells = [{"language": 30}, {"language": 30}, {"language": 15, "stance": 1}, {"language": 30}]
    assert local_mean_term_weight(cells, "stance") == Fraction(1, 4)
    assert local_mean_term_weight(cells, "language") == 1
    assert local_mean_example_weights(cells, "language") == [Fraction(105, 120)] * 2 + [Fraction(105, 60), Fraction(105, 120)]


def test_cell_loss_omits_globally_absent_terms_and_rejects_an_empty_window():
    value = torch.tensor(3.0)
    assert float(cell_loss({"language": value}, {"language": 6, "stance": 0}, WEIGHTS, ranks=2)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        cell_loss({}, {"language": 0}, WEIGHTS, ranks=1)


def test_family_weights_sum_to_one_and_cap_an_edited_family():
    weights = family_view_weights(["a"] * 20 + ["b"])
    assert sum(weights) == 1
    assert sum(weights[:20]) == weights[20] == Fraction(1, 2)


def test_presence_probability():
    assert presence_probability(Fraction(1, 100), 2) == Fraction(199, 10000)


def test_projecting_supervised_positions_gives_the_same_loss_and_gradient(model):
    tokens = torch.randint(0, 97, (3, 20), generator=torch.Generator().manual_seed(2))
    rows, positions, targets = torch.tensor([0, 1, 2]), torch.tensor([19, 7, 19]), torch.tensor([5, 9, 60])
    model.zero_grad()
    long_way = all_position_loss(model, tokens, rows, positions, targets)
    long_way.backward()
    reference = gradient(model)
    model.zero_grad()
    short_way = selected_position_loss(model, tokens, rows, positions, targets)
    short_way.backward()
    assert float(long_way.detach()) == pytest.approx(float(short_way.detach()), abs=1e-5)
    assert torch.allclose(gradient(model), reference, rtol=1e-4, atol=1e-6)


def test_label_restricted_loss_differs_by_the_log_mass_on_the_labels(model):
    tokens = torch.randint(0, 97, (2, 12), generator=torch.Generator().manual_seed(3))
    rows, positions = torch.tensor([0, 1]), torch.tensor([11, 11])
    label_ids = torch.tensor([5, 9, 60])
    with torch.no_grad():
        logits = selected_logits(model, hidden_states(model, tokens), rows, positions)
        full = F.cross_entropy(logits, label_ids[torch.tensor([0, 2])], reduction="sum")
        restricted = label_restricted_loss(logits, torch.tensor([0, 2]), label_ids)
        mass = logits.softmax(-1)[:, label_ids].sum(-1)
    assert float(full - restricted) == pytest.approx(float(-mass.log().sum()), abs=1e-5)
    assert float(full) > float(restricted)


def test_a_packed_segment_sees_what_it_would_see_alone(model):
    pieces = [torch.randint(0, 97, (n,), generator=torch.Generator().manual_seed(4 + n)) for n in (9, 14, 11)]
    packed = torch.cat(pieces)[None]
    segments = torch.cat([torch.full((p.numel(),), i) for i, p in enumerate(pieces)])[None]
    with torch.no_grad():
        segmented = model.lm_head(packed_hidden(model, packed, segments))[0]
        plain = model(packed)[0]
        start = 0
        for piece in pieces:
            alone = model(piece[None])[0]
            assert torch.allclose(segmented[start:start + piece.numel()], alone, atol=1e-5)
            if start:
                assert not torch.allclose(plain[start:start + piece.numel()], alone, atol=1e-3)
            start += piece.numel()


def test_no_gradient_crosses_a_segment_boundary(model):
    pieces = [torch.randint(0, 97, (n,), generator=torch.Generator().manual_seed(40 + n)) for n in (8, 8)]
    packed = torch.cat(pieces)[None]
    segments = torch.tensor([[0] * 8 + [1] * 8])
    for allowed, crosses in ((segment_allowed(segments), False), (None, True)):
        embeddings = model.embed_tokens(packed).detach().requires_grad_(True)
        hidden = hidden_states(model, allowed=allowed, embeddings=embeddings)
        model.lm_head(hidden)[0, 8:].logsumexp(-1).sum().backward()
        leaked = float(embeddings.grad[0, :8].abs().max())
        assert (leaked > 0) is crosses


def test_segment_mask_is_causal_within_segments_only():
    mask = segment_allowed(torch.tensor([[0, 0, 1, 1]]))[0, 0]
    assert mask.tolist() == [[True, False, False, False], [True, True, False, False],
                             [False, False, True, False], [False, False, True, True]]
