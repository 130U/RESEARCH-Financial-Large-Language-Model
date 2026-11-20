"""Semantic permissions: the loss, the scorecard and the closed-form costs."""
from fractions import Fraction
from math import log

import pytest
import torch

from finllm import accounting, architecture
from finllm.model import FinLLM, ModelConfig
from finllm.permission_math import (
    calibrated_response, consistency_limit_error, evidence_weight, logit, response_penalty, safe_margin,
    temperature_scaled, unlicensed_consistency_optimum,
)
from finllm.permissions import LICENSES, Permission, consistency, pair_loss, response, scorecard

E, L, R, U = Permission.EQUIVALENT, Permission.LABEL_PRESERVING, Permission.RELABEL, Permission.UNCONSTRAINED


def logits(*rows):
    return torch.tensor(rows, dtype=torch.float64)


def test_only_equivalent_pairs_receive_a_consistency_term():
    original, edited = logits([2.0, 0.0, -1.0]) .repeat(4, 1), logits([0.5, 0.0, -1.0]).repeat(4, 1)
    labels = torch.zeros(4, dtype=torch.long)
    terms = pair_loss(original, edited, labels, torch.tensor([0, 0, 1, 0]), [E, L, R, U])
    assert terms["consistency"].tolist() == pytest.approx([1.5, 0.0, 0.0, 0.0])
    assert float(terms["response"][0]) == float(terms["response"][1]) == float(terms["response"][3]) == 0.0


def test_consistency_is_half_the_squared_change_in_log_odds_for_two_classes():
    assert float(consistency(logits([1.2, 0.0]), logits([-0.4, 0.0]))) == pytest.approx(0.5 * 1.6 ** 2)
    assert float(consistency(logits([3.0, 1.0]), logits([5.0, 3.0]))) == pytest.approx(0.0)


def test_response_measures_the_move_toward_the_new_label():
    original, edited = logits([2.0, 0.0, 0.0]), logits([0.0, 0.0, 1.5])
    d = response(original, edited, torch.tensor([0]), torch.tensor([2]))
    assert float(d) == pytest.approx((1.5 - 0.0) - (0.0 - 2.0))
    terms = pair_loss(original, edited, torch.tensor([0]), torch.tensor([2]), [R], margin=4.0)
    assert float(terms["response"]) == pytest.approx(0.5)
    assert float(pair_loss(original, edited, torch.tensor([0]), torch.tensor([2]), [R], margin=3.0)["response"]) == 0.0


def test_a_model_that_does_not_respond_pays_the_margin():
    same = logits([2.0, 0.0, 0.0])
    terms = pair_loss(same, same.clone(), torch.tensor([0]), torch.tensor([2]), [R], margin=1.25)
    assert float(terms["response"]) == pytest.approx(1.25)


def test_permissions_enforce_their_label_relation():
    a, b = logits([1.0, 0.0]), logits([0.0, 1.0])
    with pytest.raises(ValueError):
        pair_loss(a, b, torch.tensor([0]), torch.tensor([1]), [E])
    with pytest.raises(ValueError):
        pair_loss(a, b, torch.tensor([0]), torch.tensor([0]), [R])
    with pytest.raises(ValueError):
        pair_loss(a, b, torch.tensor([0]), torch.tensor([1]), [E, L])


def test_an_unlabelled_endpoint_contributes_no_anchor():
    a, b = logits([1.0, 0.0]), logits([0.0, 1.0])
    both = pair_loss(a, b, torch.tensor([0]), torch.tensor([1]), [U])
    one = pair_loss(a, b, torch.tensor([0]), torch.tensor([-1]), [U])
    assert float(one["anchors"]) < float(both["anchors"])
    assert float(one["total"]) == float(one["anchors"])


def test_licenses_separate_label_from_distribution():
    assert LICENSES[E] == {"label": True, "distribution": True, "response": False}
    assert LICENSES[L] == {"label": True, "distribution": False, "response": False}
    assert LICENSES[R]["response"] and not LICENSES[R]["label"]
    assert not any(LICENSES[U].values())


def test_scorecard_reads_the_same_metadata_at_test_time():
    original = logits([3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [3.0, 0.0, 0.0])
    edited = logits([3.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 3.0])
    card = scorecard(original, edited, torch.tensor([0, 0, 0]), torch.tensor([0, 0, 2]), [E, L, R], margin=2.0)
    assert card["invariance_gap"] == 0.0
    assert card["label_kept"] == 1.0
    assert card["responded"] == 1.0 and card["flipped_to_new_label"] == 1.0


def test_unlicensed_consistency_pulls_two_calibrated_posteriors_together():
    previous = 0.0
    for strength in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 100.0):
        r = unlicensed_consistency_optimum(0.7, 0.9, strength)
        assert r["original"] + r["edited"] == pytest.approx(1.6)            # the two errors are equal and opposite
        assert 0.7 < r["original"] < r["edited"] < 0.9
        assert r["error"] == pytest.approx(strength * (logit(r["edited"]) - logit(r["original"])))
        assert previous < r["error"] < consistency_limit_error(0.7, 0.9)
        previous = r["error"]
    assert unlicensed_consistency_optimum(0.7, 0.9, 1e6)["error"] == pytest.approx(0.1, abs=1e-6)
    assert unlicensed_consistency_optimum(0.9, 0.7, 1.0)["error"] == pytest.approx(unlicensed_consistency_optimum(0.7, 0.9, 1.0)["error"])


def test_consistency_costs_nothing_on_a_true_equivalence_or_at_zero_strength():
    assert unlicensed_consistency_optimum(0.8, 0.8, 5.0)["error"] == 0.0
    assert unlicensed_consistency_optimum(0.7, 0.9, 0.0)["error"] == 0.0


def test_the_response_constraint_is_free_for_a_calibrated_model_up_to_the_safe_margin():
    assert safe_margin(2.0) == pytest.approx(2 * log(2))
    # both labels assigned at posterior odds of at least 2 against the competing label
    d = calibrated_response(p_old_original=0.6, p_new_original=0.3, p_old_edited=0.3, p_new_edited=0.6)
    assert d == pytest.approx(2 * log(2))
    assert response_penalty(safe_margin(2.0), d) == pytest.approx(0.0, abs=1e-12)
    assert response_penalty(safe_margin(2.0), 0.0) == pytest.approx(safe_margin(2.0))


def test_evidence_weight_and_temperature():
    assert evidence_weight(0.7, 0.9) == pytest.approx(log(27 / 7))
    assert temperature_scaled(0.9, 1.0) == pytest.approx(0.9)
    assert 0.5 < temperature_scaled(0.9, 1.50634) < 0.9


def test_parameter_counts_match_the_model_and_the_published_figures():
    assert architecture.parameters(architecture.TRIAL) == FinLLM(ModelConfig()).num_parameters() == 24_255_744
    assert architecture.layer_parameters(architecture.PRIVATE) == 79_631_616
    assert architecture.parameters(architecture.PRIVATE) == 2_397_459_968
    assert architecture.training_state_bytes(architecture.PRIVATE) == 38_359_359_488
    assert architecture.logits_bytes(2 * 2048, 65536) == 2 ** 30
    assert architecture.logits_bytes(2, 65536) == 2 ** 19
    small = architecture.Shape(layers=2, hidden=64, ffn=192, query_heads=4, kv_heads=2, vocab=512, context=256)
    built = FinLLM(ModelConfig(vocab_size=512, hidden_size=64, num_hidden_layers=2, num_attention_heads=4,
                               num_key_value_heads=2, intermediate_size=192, max_position_embeddings=256))
    assert architecture.parameters(small) == built.num_parameters()


def test_a_constant_predictor_scores_the_majority_identity():
    development = {"bearish": 10, "neutral": 76, "bullish": 15}
    baseline = accounting.majority_baseline(development)
    assert baseline["accuracy"] == Fraction(76, 101)
    assert float(baseline["macro_f1"]) == pytest.approx(0.2863, abs=5e-5)
    collapsed = accounting.summary({"bearish": 0, "neutral": 76, "bullish": 0}, development,
                                   {"bearish": 0, "neutral": 101, "bullish": 0})
    assert collapsed["macro_f1"] == baseline["macro_f1"] and collapsed["accuracy"] == baseline["accuracy"]


def test_the_balanced_model_and_what_it_exchanged():
    development = {"bearish": 10, "neutral": 76, "bullish": 15}
    after = {"bearish": 5, "neutral": 62, "bullish": 12}
    result = accounting.summary(after, development, {"bearish": 11, "neutral": 66, "bullish": 24})
    assert result["accuracy"] == Fraction(79, 101)
    assert float(result["macro_f1"]) == pytest.approx(0.6549, abs=5e-5)
    assert accounting.exchange({"bearish": 0, "neutral": 76, "bullish": 0}, after) == {
        "directional_gained": 17, "neutral_lost": 14, "net": 3}
    assert accounting.sibling_in_training(1, Fraction(7, 10)) == 0
    assert accounting.sibling_in_training(2, Fraction(7, 10)) == Fraction(7, 10)
