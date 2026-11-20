"""What each semantic permission costs a calibrated model, in closed form.

Standard library only. Probabilities are those of one class against the rest,
log-odds are natural logarithms, and an edit maps an original text to an
edited one.
"""
from __future__ import annotations

from math import exp, log


def sigmoid(s: float) -> float:
    return 1.0 / (1.0 + exp(-s))


def logit(p: float) -> float:
    if not 0.0 < p < 1.0:
        raise ValueError("probability must be strictly between 0 and 1")
    return log(p / (1.0 - p))


def evidence_weight(p_original: float, p_edited: float) -> float:
    """Log-likelihood ratio an edit carries: the change in log-odds between two calibrated posteriors."""
    return logit(p_edited) - logit(p_original)


def unlicensed_consistency_optimum(p_original: float, p_edited: float, strength: float) -> dict:
    """Minimizer of CE(p; q) + CE(p'; q') + (strength / 2) (logit q - logit q')^2.

    The first-order conditions give q - p = p' - q' = strength * (logit q' - logit q),
    so the two predictions are pulled toward each other by the same amount.
    Returns the predictions and that amount, the calibration error at each end.
    """
    if strength < 0:
        raise ValueError("strength must be nonnegative")
    lo, hi = (p_original, p_edited) if p_original <= p_edited else (p_edited, p_original)
    if strength == 0 or lo == hi:
        return {"original": p_original, "edited": p_edited, "error": 0.0}
    # error e in (0, (hi - lo) / 2) solves e = strength * (logit(hi - e) - logit(lo + e));
    # the right side falls from strength * gap to 0 as e rises, so bisection finds the root.
    low, high = 0.0, (hi - lo) / 2.0
    for _ in range(200):
        mid = (low + high) / 2.0
        if mid < strength * (logit(hi - mid) - logit(lo + mid)):
            low = mid
        else:
            high = mid
    error = (low + high) / 2.0
    toward = 1.0 if p_original <= p_edited else -1.0
    return {"original": p_original + toward * error, "edited": p_edited - toward * error, "error": error}


def consistency_limit_error(p_original: float, p_edited: float) -> float:
    """Calibration error at each end when consistency is enforced exactly: half the gap."""
    return abs(p_edited - p_original) / 2.0


def response(log_odds_original: float, log_odds_edited: float) -> float:
    """D: how far the log-odds of the new label against the old one moved across the edit."""
    return log_odds_edited - log_odds_original


def response_penalty(margin: float, d: float) -> float:
    """The relative-response constraint: max(0, margin - D)."""
    return max(0.0, margin - d)


def calibrated_response(p_old_original: float, p_new_original: float,
                        p_old_edited: float, p_new_edited: float) -> float:
    """D of the calibrated model on a relabelling edit, from the posteriors of the old and new labels."""
    return log(p_new_edited / p_old_edited) - log(p_new_original / p_old_original)


def safe_margin(odds_floor: float) -> float:
    """Largest margin that never penalizes a calibrated model.

    If a label is assigned only when its posterior odds against the competing
    label are at least ``odds_floor`` at both ends of a relabelling edit, the
    calibrated response is at least 2 ln(odds_floor).
    """
    if odds_floor < 1:
        raise ValueError("odds floor must be at least 1")
    return 2.0 * log(odds_floor)


def temperature_scaled(probability: float, temperature: float) -> float:
    """A two-class confidence after dividing its log-odds by the temperature."""
    return sigmoid(logit(probability) / temperature)
