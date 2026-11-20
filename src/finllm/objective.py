"""One update, one objective: reductions that do not depend on how the update is executed.

An optimizer update consumes a window of examples. The window may be split
over several ranks and several accumulation microbatches; each such piece is a
cell. The objective is defined on the window:

    L = sum over terms j of  weight_j * (sum of term-j losses in the window) / C_j

where C_j is the term's effective count over the whole window. A cell that
contributes ``ranks * sum_j weight_j * S_j / C_j``, with gradients summed over
microbatches and averaged over ranks, reproduces the gradient of L exactly,
whatever the split. Dividing each cell's sums by the cell's own counts does
not: a term's weight then depends on which cells happen to contain it.
"""
from __future__ import annotations

from collections import Counter
from fractions import Fraction

import torch


def global_denominators(cell_counts: list[dict[str, int]]) -> dict[str, int]:
    """Effective count of every term over all cells of one update window."""
    total: Counter = Counter()
    for counts in cell_counts:
        total.update(counts)
    return dict(total)


def cell_loss(sums: dict[str, torch.Tensor], denominators: dict[str, int],
              weights: dict[str, float], ranks: int) -> torch.Tensor:
    """Loss of one cell under global denominators, for a reducer that averages gradients over ranks.

    Terms whose global count is zero are left out; a term absent from this cell
    contributes nothing here and is still normalized by its global count elsewhere.
    """
    parts = [weights[name] * value / denominators[name]
             for name, value in sums.items() if denominators.get(name, 0) > 0]
    if not parts:
        raise ValueError("the window has no supervised position")
    return ranks * torch.stack(parts).sum()


def local_mean_loss(sums: dict[str, torch.Tensor], counts: dict[str, int],
                    weights: dict[str, float]) -> torch.Tensor:
    """Loss of one cell when each term is averaged over the cell's own count."""
    parts = [weights[name] * value / counts[name] for name, value in sums.items() if counts.get(name, 0) > 0]
    return torch.stack(parts).sum()


def local_mean_term_weight(cell_counts: list[dict[str, int]], term: str) -> Fraction:
    """Share of a term's nominal weight that survives per-cell averaging: the share of cells that contain it."""
    return Fraction(sum(1 for counts in cell_counts if counts.get(term, 0) > 0), len(cell_counts))


def local_mean_example_weights(cell_counts: list[dict[str, int]], term: str) -> list[Fraction]:
    """Relative weight of an example of ``term`` in each cell under per-cell averaging, against the global rule.

    Under global denominators every example of the term weighs 1 / C. Under
    per-cell averaging an example in a cell with n of them weighs 1 / (cells * n).
    """
    total = sum(counts.get(term, 0) for counts in cell_counts)
    cells = len(cell_counts)
    return [Fraction(total, cells * counts[term]) for counts in cell_counts if counts.get(term, 0) > 0]


def presence_probability(share: Fraction, sequences_per_cell: int) -> Fraction:
    """Chance that a cell of independently drawn sequences contains a term whose sequences have the given share."""
    return 1 - (1 - share) ** sequences_per_cell


def family_view_weights(family_of_view: list[str]) -> list[Fraction]:
    """Weight of each view when views are averaged within a family and families are averaged within a term.

    Twenty edits of one source text then weigh, together, as much as one unedited family.
    """
    sizes = Counter(family_of_view)
    families = len(sizes)
    return [Fraction(1, families * sizes[f]) for f in family_of_view]
