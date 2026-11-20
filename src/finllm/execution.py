"""Execution paths that leave the objective unchanged.

Two of them are implemented here on the Trial model. Projecting only the
supervised positions through the vocabulary head gives the same full-vocabulary
loss and the same gradients as projecting every position, with a logits tensor
that is smaller by the ratio of positions. Packing several examples into one
sequence with segmented causal attention gives each example the logits it
would have alone, and no gradient crosses a segment boundary.
"""
from __future__ import annotations

import torch
from torch.nn import functional as F

from .model import FinLLM


def hidden_states(model: FinLLM, input_ids: torch.Tensor | None = None, *,
                  allowed: torch.Tensor | None = None,
                  embeddings: torch.Tensor | None = None) -> torch.Tensor:
    """Final-norm hidden states [batch, length, hidden], before the vocabulary head.

    ``allowed`` is a boolean mask [batch, 1, length, length], True where a query
    may attend; None means plain causal attention. Pass ``embeddings`` instead
    of ``input_ids`` to differentiate with respect to the inputs.
    """
    x = model.embed_tokens(input_ids) if embeddings is None else embeddings
    length = x.shape[1]
    if length > model.config.max_position_embeddings:
        raise ValueError("input exceeds max_position_embeddings")
    cos, sin = model.rope_cos[:, :, :length], model.rope_sin[:, :, :length]
    for layer in model.layers:
        x = layer(x, cos, sin, allowed, None)
    return model.norm(x)


def selected_logits(model: FinLLM, hidden: torch.Tensor, rows: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Vocabulary logits at the selected positions only: [selected, vocab]."""
    return model.lm_head(hidden[rows, positions])


def selected_position_loss(model: FinLLM, input_ids: torch.Tensor, rows: torch.Tensor,
                           positions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Full-vocabulary cross-entropy at the supervised positions, summed."""
    logits = selected_logits(model, hidden_states(model, input_ids), rows, positions)
    return F.cross_entropy(logits.float(), targets, reduction="sum")


def all_position_loss(model: FinLLM, input_ids: torch.Tensor, rows: torch.Tensor,
                      positions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """The same loss computed the long way: project every position, then pick the supervised ones."""
    logits = model(input_ids)[rows, positions]
    return F.cross_entropy(logits.float(), targets, reduction="sum")


def label_restricted_loss(logits: torch.Tensor, targets: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """Cross-entropy over the label tokens alone: a different objective, kept here for comparison.

    ``targets`` index into ``label_ids``. The full-vocabulary loss exceeds this
    one by minus the log of the probability mass on the label tokens.
    """
    return F.cross_entropy(logits[:, label_ids].float(), targets, reduction="sum")


def segment_allowed(segment_ids: torch.Tensor) -> torch.Tensor:
    """Causal attention confined to each segment: [batch, 1, length, length], True where attention is allowed."""
    length = segment_ids.shape[1]
    causal = torch.ones(length, length, dtype=torch.bool, device=segment_ids.device).tril()
    same = segment_ids[:, :, None] == segment_ids[:, None, :]
    return (same & causal)[:, None]


def packed_hidden(model: FinLLM, input_ids: torch.Tensor, segment_ids: torch.Tensor) -> torch.Tensor:
    """Hidden states of a packed sequence. Rotary positions are relative, so no position is reset."""
    return hidden_states(model, input_ids, allowed=segment_allowed(segment_ids))
