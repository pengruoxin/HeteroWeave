"""Validation helpers for variable-cardinality composition encodings."""

from __future__ import annotations

from typing import Hashable, Iterable, Sequence


def validate_composition(
    positions: Sequence[Iterable[Hashable]],
    max_cardinalities: Sequence[int],
) -> tuple[tuple[Hashable, ...], ...]:
    """Validate nonempty, duplicate-free selections at all positions.

    Repeated use of one source is represented by distinct component instance
    identifiers. A mathematical set cannot encode the same identifier twice.
    """
    if len(positions) != len(max_cardinalities):
        raise ValueError("positions and max_cardinalities must have equal length")
    normalized = []
    for index, (selection, maximum) in enumerate(zip(positions, max_cardinalities)):
        items = tuple(selection)
        if int(maximum) < 1:
            raise ValueError(f"position {index}: maximum must be positive")
        if not items:
            raise ValueError(f"position {index}: selection must be nonempty")
        if len(set(items)) != len(items):
            raise ValueError(
                f"position {index}: repeated identifiers require distinct instances"
            )
        if len(items) > int(maximum):
            raise ValueError(f"position {index}: cardinality exceeds its maximum")
        normalized.append(items)
    return tuple(normalized)
