"""Closed-form sizes of the two configurations: parameters, training state, logits tensors.

Everything here is integer arithmetic on a configuration. The Trial's count is
checked against the instantiated model in the tests; the 2.4B count follows
from the same formula.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Shape:
    layers: int
    hidden: int
    ffn: int
    query_heads: int
    kv_heads: int
    vocab: int
    context: int

    @property
    def head_dim(self) -> int:
        return self.hidden // self.query_heads


TRIAL = Shape(layers=6, hidden=512, ffn=1664, query_heads=8, kv_heads=4, vocab=8192, context=512)
PRIVATE = Shape(layers=28, hidden=2560, ffn=8320, query_heads=20, kv_heads=4, vocab=65536, context=4096)


def attention_parameters(s: Shape) -> int:
    """Query and output projections are hidden x hidden; key and value serve kv_heads groups."""
    kv_width = s.kv_heads * s.head_dim
    return 2 * s.hidden * s.hidden + 2 * s.hidden * kv_width


def layer_parameters(s: Shape) -> int:
    """Attention, a SwiGLU feed-forward block of three matrices, two RMSNorm scales, and QK-norm scales."""
    feed_forward = 3 * s.hidden * s.ffn
    norms = 2 * s.hidden
    qk_norm = 2 * s.head_dim
    return attention_parameters(s) + feed_forward + norms + qk_norm


def parameters(s: Shape) -> int:
    """All layers, one vocabulary matrix shared by the embedding and the output head, and the final norm."""
    return s.layers * layer_parameters(s) + s.vocab * s.hidden + s.hidden


def training_state_bytes(s: Shape, bytes_per_value: int = 4, tensors: int = 4) -> int:
    """Weights, gradients and the two Adam moments at the given precision; activations excluded."""
    return parameters(s) * bytes_per_value * tensors


def logits_bytes(positions: int, vocab: int, bytes_per_value: int = 4) -> int:
    """Size of a logits tensor with one row per projected position."""
    return positions * vocab * bytes_per_value


def projection_share(s: Shape) -> float:
    """Share of a position's forward matrix work spent in the output projection.

    Matrix work per position is proportional to the number of weights it is
    multiplied by: every layer's matrices, then hidden x vocab for the head.
    """
    layer_matrices = s.layers * (attention_parameters(s) + 3 * s.hidden * s.ffn)
    head = s.hidden * s.vocab
    return head / (layer_matrices + head)


def tokens_per_parameter(tokens: int, s: Shape) -> float:
    return tokens / parameters(s)


# The 2.4B training recipe, in input tokens and in human-labelled records. These are allocations.
RECIPE = {
    "foundation_tokens": 50_000_000_000,
    "adaptation_tokens": {
        "financial_source_text": 4_000_000_000,
        "general_text": 750_000_000,
        "reading_questions_and_answers": 200_000_000,
        "stance_and_event_tasks": 50_000_000,
    },
    "labelled_records": {
        "training": 12_000,
        "development": 4_000,
        "calibration": 4_000,
        "threshold_selection": 4_000,
        "final_test": 8_000,
    },
}
