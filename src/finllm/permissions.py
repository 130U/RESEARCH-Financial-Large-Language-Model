"""Financial semantic permissions as a loss.

An edit turns one text into another. For each task axis the pair carries one
permission, assigned at review time, that says what the model's output is
allowed and required to do across the edit:

    equivalent        the label and the whole distribution are preserved
    label_preserving  the label is preserved; the distribution is free to move
    relabel           the correct label changes from one class to another
    unconstrained     no relation between the two outputs is imposed

Label preservation and distribution preservation are separate licenses. Only
``equivalent`` pairs receive a consistency term; ``relabel`` pairs receive a
relative-response term on the log-odds of the new label against the old one;
every labelled endpoint keeps its cross-entropy anchor. Permissions, labels and
edit names are supervision metadata and never enter the model's input.
"""
from __future__ import annotations

from enum import Enum

import torch
from torch.nn import functional as F


class Permission(str, Enum):
    EQUIVALENT = "equivalent"
    LABEL_PRESERVING = "label_preserving"
    RELABEL = "relabel"
    UNCONSTRAINED = "unconstrained"


LICENSES = {
    Permission.EQUIVALENT: {"label": True, "distribution": True, "response": False},
    Permission.LABEL_PRESERVING: {"label": True, "distribution": False, "response": False},
    Permission.RELABEL: {"label": False, "distribution": False, "response": True},
    Permission.UNCONSTRAINED: {"label": False, "distribution": False, "response": False},
}


def consistency(logits_original: torch.Tensor, logits_edited: torch.Tensor) -> torch.Tensor:
    """Squared distance between centred logits. For two classes this is half the squared change in log-odds."""
    a = logits_original - logits_original.mean(-1, keepdim=True)
    b = logits_edited - logits_edited.mean(-1, keepdim=True)
    return (a - b).square().sum(-1)


def response(logits_original: torch.Tensor, logits_edited: torch.Tensor,
             old: torch.Tensor, new: torch.Tensor) -> torch.Tensor:
    """D: the change across the edit in the log-odds of the new label against the old one."""
    def log_odds(logits):
        return logits.gather(-1, new[:, None]).squeeze(-1) - logits.gather(-1, old[:, None]).squeeze(-1)
    return log_odds(logits_edited) - log_odds(logits_original)


def pair_loss(logits_original: torch.Tensor, logits_edited: torch.Tensor,
              labels_original: torch.Tensor, labels_edited: torch.Tensor,
              permissions: list[Permission], *, consistency_weight: float = 1.0,
              response_weight: float = 1.0, margin: float = 1.0) -> dict[str, torch.Tensor]:
    """Per-pair loss terms for class logits of shape [pairs, classes].

    A label of -1 marks an endpoint without a label; it contributes no anchor.
    """
    if not (len(permissions) == logits_original.shape[0] == logits_edited.shape[0]):
        raise ValueError("one permission per pair is required")
    equivalent = torch.tensor([p is Permission.EQUIVALENT for p in permissions], device=logits_original.device)
    relabel = torch.tensor([p is Permission.RELABEL for p in permissions], device=logits_original.device)
    same = labels_original == labels_edited
    for i, permission in enumerate(permissions):
        if LICENSES[permission]["label"] and not bool(same[i]):
            raise ValueError("a label-preserving permission needs the same label at both ends")
        if permission is Permission.RELABEL and (bool(same[i]) or labels_original[i] < 0 or labels_edited[i] < 0):
            raise ValueError("a relabelling permission needs two different labels")

    def anchor(logits, labels):
        return F.cross_entropy(logits, labels.clamp(min=0), reduction="none") * (labels >= 0)

    anchors = anchor(logits_original, labels_original) + anchor(logits_edited, labels_edited)
    cons = consistency(logits_original, logits_edited) * equivalent
    d = response(logits_original, logits_edited, labels_original.clamp(min=0), labels_edited.clamp(min=0))
    resp = F.relu(margin - d) * relabel
    return {"anchors": anchors, "consistency": cons, "response": resp,
            "total": anchors + consistency_weight * cons + response_weight * resp}


def scorecard(logits_original: torch.Tensor, logits_edited: torch.Tensor,
              labels_original: torch.Tensor, labels_edited: torch.Tensor,
              permissions: list[Permission], *, margin: float = 1.0) -> dict[str, float]:
    """How a model behaves under each permission: the test-time reading of the same metadata."""
    equivalent = torch.tensor([p is Permission.EQUIVALENT for p in permissions])
    relabel = torch.tensor([p is Permission.RELABEL for p in permissions])
    keep = torch.tensor([LICENSES[p]["label"] for p in permissions])
    predicted_original, predicted_edited = logits_original.argmax(-1), logits_edited.argmax(-1)
    d = response(logits_original, logits_edited, labels_original.clamp(min=0), labels_edited.clamp(min=0))

    def mean(values, mask):
        return float(values[mask].float().mean()) if bool(mask.any()) else float("nan")

    return {
        "invariance_gap": mean(consistency(logits_original, logits_edited), equivalent),
        "label_kept": mean(predicted_original == predicted_edited, keep),
        "responded": mean(d >= margin, relabel),
        "flipped_to_new_label": mean(predicted_edited == labels_edited, relabel),
    }
