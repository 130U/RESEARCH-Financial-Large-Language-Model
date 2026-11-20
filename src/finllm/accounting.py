"""Classification accounting: what a confusion table says, and what a constant predictor scores.

Standard library only.
"""
from __future__ import annotations

from fractions import Fraction
from math import log


def majority_baseline(support: dict[str, int]) -> dict:
    """Scores of the predictor that always answers the largest class.

    Accuracy is the majority share pi. Its F1 on the majority class is
    2 pi / (1 + pi) and zero elsewhere, so macro-F1 is that value over the
    number of classes.
    """
    total = sum(support.values())
    share = Fraction(max(support.values()), total)
    return {"accuracy": share, "macro_f1": 2 * share / (1 + share) / len(support)}


def per_class(correct: dict[str, int], support: dict[str, int], predicted: dict[str, int]) -> dict:
    out = {}
    for label in support:
        precision = Fraction(correct[label], predicted[label]) if predicted[label] else Fraction(0)
        recall = Fraction(correct[label], support[label])
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else Fraction(0)
        out[label] = {"precision": precision, "recall": recall, "f1": f1}
    return out


def summary(correct: dict[str, int], support: dict[str, int], predicted: dict[str, int]) -> dict:
    classes = per_class(correct, support, predicted)
    total = sum(support.values())
    return {"accuracy": Fraction(sum(correct.values()), total),
            "macro_f1": sum(c["f1"] for c in classes.values()) / len(classes), "per_class": classes}


def exchange(correct_before: dict[str, int], correct_after: dict[str, int], neutral: str = "neutral") -> dict:
    """What a change of model bought and what it cost, class by class."""
    gained = sum(correct_after[k] - correct_before[k] for k in correct_after if k != neutral)
    lost = correct_before[neutral] - correct_after[neutral]
    return {"directional_gained": gained, "neutral_lost": lost, "net": gained - lost}


def prior_entropy(counts: dict[str, int]) -> float:
    """Cross-entropy, in nats, of the predictor that outputs the class shares."""
    total = sum(counts.values())
    return -sum(c / total * log(c / total) for c in counts.values() if c)


def sibling_in_training(views: int, train_share: Fraction) -> Fraction:
    """Chance that a held-out view has a sibling in training when views of one family are split independently."""
    return 1 - (1 - train_share) ** (views - 1)
