"""Dependency-light reference implementation of Clas.

Clas counts activation patterns at each evaluated functional position and
uses square-root aggregation to reduce score growth caused only by an enlarged
activation space. The full model hooks used by the experiments live in the
task entry points under ``tools/``.
"""

from __future__ import annotations

import math
from typing import Iterable, Sequence

import numpy as np


def clas_from_pattern_counts(pattern_counts: Iterable[int]) -> float:
    """Return the Clas score from nonnegative position-level pattern counts."""
    counts = [int(value) for value in pattern_counts]
    if not counts:
        raise ValueError("at least one pattern count is required")
    if any(value < 0 for value in counts):
        raise ValueError("pattern counts must be nonnegative")
    return float(sum(math.sqrt(value) for value in counts))


def _unique_binary_patterns(activation: np.ndarray) -> int:
    values = np.asarray(activation)
    if values.ndim < 2:
        raise ValueError("each activation must have shape [samples, ...]")
    binary = (values > 0).reshape(values.shape[0], -1).T
    packed = np.packbits(binary, axis=1)
    return int(np.unique(packed, axis=0).shape[0])


def clas_from_activations(activations: Sequence[np.ndarray]) -> float:
    """Compute Clas from activation tensors collected at search positions."""
    if not activations:
        raise ValueError("at least one activation tensor is required")
    return clas_from_pattern_counts(
        _unique_binary_patterns(activation) for activation in activations
    )
