"""The training contract, measured on the Trial model.

Each function builds the 24,255,744-parameter Trial model at a fixed seed on
the CPU, runs one comparison and returns the numbers. Token ids are drawn from
a fixed seed: the comparisons are about how a loss is reduced and executed,
and hold for any content.
"""
from __future__ import annotations

import pathlib
import sys

import torch
from torch.nn import functional as F

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from finllm.execution import (  # noqa: E402
    all_position_loss, hidden_states, label_restricted_loss, packed_hidden, selected_logits, selected_position_loss,
)
from finllm.model import FinLLM, ModelConfig  # noqa: E402
from finllm.objective import cell_loss, global_denominators, local_mean_loss  # noqa: E402
from finllm.permissions import Permission, pair_loss  # noqa: E402

SEED = 130
WEIGHTS = {"language": 1.0, "answer": 0.5, "stance": 0.5}


def trial_model() -> FinLLM:
    torch.manual_seed(SEED)
    torch.set_num_threads(min(4, torch.get_num_threads()))
    return FinLLM(ModelConfig()).eval()


def flat_gradient(model: FinLLM) -> torch.Tensor:
    return torch.cat([p.grad.reshape(-1) for p in model.parameters()])


def relative_difference(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).norm() / b.norm())


# ----------------------------------------------------------------------------- one update, one objective

def window(length: int = 64):
    """Sixteen sequences: twelve of running text, two question-answer pairs, two stance examples."""
    generator = torch.Generator().manual_seed(SEED)
    tokens = torch.randint(0, 8192, (16, length), generator=generator)
    kinds = ["language"] * 12 + ["answer"] * 2 + ["stance"] * 2
    return tokens, kinds


def sequence_sums(model: FinLLM, tokens: torch.Tensor, kinds: list[str]):
    """Per-term loss sums and supervised-position counts for a batch of sequences."""
    logits = model(tokens)
    losses = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                             tokens[:, 1:].reshape(-1), reduction="none").view(tokens.shape[0], -1)
    sums, counts = {}, {}
    for row, kind in enumerate(kinds):
        if kind == "language":          # every next-token position
            chosen = losses[row]
        elif kind == "answer":          # the eight answer positions at the end
            chosen = losses[row, -8:]
        else:                           # one label position
            chosen = losses[row, -1:]
        sums[kind] = sums.get(kind, 0) + chosen.sum()
        counts[kind] = counts.get(kind, 0) + chosen.numel()
    return sums, counts


def partition_invariance() -> dict:
    model = trial_model()
    tokens, kinds = window()

    def gradient_single():
        model.zero_grad()
        sums, counts = sequence_sums(model, tokens, kinds)
        cell_loss(sums, counts, WEIGHTS, ranks=1).backward()
        return flat_gradient(model).clone()

    # four ranks, two microbatches each, two sequences per cell; the stance examples
    # share one cell and the two question-answer pairs sit in two different cells
    order = [0, 1, 2, 12, 3, 4, 5, 13, 6, 7, 8, 9, 10, 11, 14, 15]
    cells = [order[i:i + 2] for i in range(0, 16, 2)]
    layout = [cells[r * 2:(r + 1) * 2] for r in range(4)]

    def gradient_split(local: bool):
        counts_per_cell = []
        for rank in layout:
            for rows in rank:
                counts_per_cell.append({k: v for k, v in sequence_counts([kinds[i] for i in rows], tokens.shape[1]).items()})
        denominators = global_denominators(counts_per_cell)
        total = None
        for rank in layout:
            model.zero_grad()
            for rows in rank:                               # gradients accumulate over microbatches
                sums, counts = sequence_sums(model, tokens[rows], [kinds[i] for i in rows])
                loss = (local_mean_loss(sums, counts, WEIGHTS) / len(rank) if local
                        else cell_loss(sums, denominators, WEIGHTS, ranks=len(layout)))
                loss.backward()
            grad = flat_gradient(model).clone()
            total = grad if total is None else total + grad
        return total / len(layout), counts_per_cell         # the reducer averages over ranks

    reference = gradient_single()
    global_grad, counts_per_cell = gradient_split(local=False)
    local_grad, _ = gradient_split(local=True)
    stance_cells = sum(1 for c in counts_per_cell if c.get("stance", 0) > 0)
    answer_cells = sum(1 for c in counts_per_cell if c.get("answer", 0) > 0)
    return {"sequences": len(kinds), "ranks": len(layout), "microbatches_per_rank": len(layout[0]),
            "cells": len(counts_per_cell),
            "global_relative_difference": relative_difference(global_grad, reference),
            "local_relative_difference": relative_difference(local_grad, reference),
            "cells_with_stance": stance_cells, "cells_with_answer": answer_cells,
            "denominators": global_denominators(counts_per_cell)}


def sequence_counts(kinds: list[str], length: int) -> dict[str, int]:
    per = {"language": length - 1, "answer": 8, "stance": 1}
    counts: dict[str, int] = {}
    for kind in kinds:
        counts[kind] = counts.get(kind, 0) + per[kind]
    return counts


# ----------------------------------------------------------------------------- projecting supervised positions

def projection_equivalence() -> dict:
    model = trial_model()
    generator = torch.Generator().manual_seed(SEED + 1)
    tokens = torch.randint(0, 8192, (2, 128), generator=generator)
    rows, positions = torch.tensor([0, 1]), torch.tensor([127, 127])
    targets = torch.tensor([17, 4711])

    model.zero_grad()
    long_way = all_position_loss(model, tokens, rows, positions, targets)
    long_way.backward()
    reference = flat_gradient(model).clone()
    model.zero_grad()
    short_way = selected_position_loss(model, tokens, rows, positions, targets)
    short_way.backward()
    selected = flat_gradient(model).clone()

    with torch.no_grad():
        logits = selected_logits(model, hidden_states(model, tokens), rows, positions)
        label_ids = torch.tensor([17, 4711, 99])
        restricted = label_restricted_loss(logits, torch.tensor([0, 1]), label_ids)
        mass = logits.float().softmax(-1)[:, label_ids].sum(-1)
    return {"loss_all_positions": float(long_way.detach()), "loss_selected_positions": float(short_way.detach()),
            "loss_difference": abs(float(long_way.detach()) - float(short_way.detach())),
            "gradient_relative_difference": relative_difference(selected, reference),
            "label_restricted_loss": float(restricted),
            "full_minus_restricted": float(short_way.detach()) - float(restricted),
            "minus_log_label_mass": float(-mass.log().sum())}


# ----------------------------------------------------------------------------- packing with segmented attention

def packing_isolation() -> dict:
    model = trial_model()
    generator = torch.Generator().manual_seed(SEED + 2)
    lengths = [20, 24, 20]
    pieces = [torch.randint(0, 8192, (n,), generator=generator) for n in lengths]
    packed = torch.cat(pieces)[None]
    segments = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)])[None]
    with torch.no_grad():
        alone = [model(piece[None])[0] for piece in pieces]
        segmented = model.lm_head(packed_hidden(model, packed, segments))[0]
        plain = model(packed)[0]
    starts = [0, lengths[0], lengths[0] + lengths[1]]
    seg_diff = max(float((segmented[s:s + n] - a).abs().max()) for s, n, a in zip(starts, lengths, alone))
    plain_diff = max(float((plain[s:s + n] - a).abs().max()) for s, n, a in zip(starts[1:], lengths[1:], alone[1:]))

    def leak(use_segments: bool) -> float:
        embeddings = model.embed_tokens(packed).detach().requires_grad_(True)
        from finllm.execution import segment_allowed
        allowed = segment_allowed(segments) if use_segments else None
        hidden = hidden_states(model, allowed=allowed, embeddings=embeddings)
        logits = model.lm_head(hidden)[0, starts[1]:starts[1] + lengths[1]]
        logits.float().logsumexp(-1).sum().backward()
        return float(embeddings.grad[0, :lengths[0]].abs().max())

    return {"segments": len(lengths), "packed_length": int(packed.shape[1]),
            "segmented_max_logit_difference": seg_diff, "plain_causal_max_logit_difference": plain_diff,
            "segmented_gradient_into_earlier_segment": leak(True),
            "plain_causal_gradient_into_earlier_segment": leak(False),
            "typical_logit_magnitude": float(plain.abs().mean())}


# ----------------------------------------------------------------------------- the permission loss, minimized

def consistency_fixed_point(p_original: float, p_edited: float, strength: float, steps: int = 4000) -> dict:
    """Minimize the implemented pair loss over free logits for one pair with known posteriors."""
    logits = torch.zeros(2, 2, dtype=torch.float64, requires_grad=True)   # [endpoint, class]
    optimizer = torch.optim.LBFGS([logits], lr=1.0, max_iter=steps, tolerance_grad=1e-14, tolerance_change=1e-16,
                                  line_search_fn="strong_wolfe")
    targets = torch.tensor([[p_original, 1 - p_original], [p_edited, 1 - p_edited]], dtype=torch.float64)

    def closure():
        optimizer.zero_grad()
        log_q = logits.log_softmax(-1)
        anchors = -(targets * log_q).sum()                      # expected cross-entropy at both ends
        terms = pair_loss(logits[0:1], logits[1:2], torch.tensor([0]), torch.tensor([0]),
                          [Permission.EQUIVALENT], consistency_weight=strength)
        loss = anchors + strength * terms["consistency"].sum()
        loss.backward()
        return loss

    optimizer.step(closure)
    q = logits.detach().softmax(-1)[:, 0]
    return {"original": float(q[0]), "edited": float(q[1])}


def tied_head() -> bool:
    """The input embedding and the output head are one parameter object."""
    model = trial_model()
    return model.lm_head.weight is model.embed_tokens.weight


def run() -> dict:
    return {"partition": partition_invariance(), "projection": projection_equivalence(),
            "packing": packing_isolation(), "tied_head": tied_head()}


if __name__ == "__main__":
    import json
    print(json.dumps(run(), indent=2))
