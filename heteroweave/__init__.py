"""Small, dependency-light utilities for inspecting HeteroWeave."""

from .clas import clas_from_activations, clas_from_pattern_counts
from .composition import validate_composition
from .pareto import dominates, nondominated_indices

__all__ = [
    "clas_from_activations",
    "clas_from_pattern_counts",
    "validate_composition",
    "dominates",
    "nondominated_indices",
]
