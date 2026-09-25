"""Three-objective Pareto utilities used for lightweight verification."""

from __future__ import annotations

from typing import Iterable, Sequence


def dominates(left: Sequence[float], right: Sequence[float]) -> bool:
    """Return whether left dominates right for [quality, params, FLOPs]."""
    if len(left) != 3 or len(right) != 3:
        raise ValueError("points must contain quality, parameters, and FLOPs")
    no_worse = left[0] >= right[0] and left[1] <= right[1] and left[2] <= right[2]
    strictly_better = left[0] > right[0] or left[1] < right[1] or left[2] < right[2]
    return bool(no_worse and strictly_better)


def nondominated_indices(points: Iterable[Sequence[float]]) -> list[int]:
    """Return stable indices of strict nondominated points."""
    values = [tuple(map(float, point)) for point in points]
    return [
        index
        for index, point in enumerate(values)
        if not any(
            other_index != index and dominates(other, point)
            for other_index, other in enumerate(values)
        )
    ]
