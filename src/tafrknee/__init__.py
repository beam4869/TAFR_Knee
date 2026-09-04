"""TAFR-Knee: trade-off-active finite-radius knee detection."""

from ._version import __version__
from .api import audit_weight, discover_knees, select_knee
from .config import KneeConfig
from .exceptions import ConfigurationError, SolveError, TAFRKneeError
from .normalization import FrozenNormalizer, fit_normalizer
from .problem import CallableProblem, CallbackProblem, MultiObjectiveProblem, TabularProblem
from .reduction import LinearReduction
from .types import SelectionResult, SolveResult, WeightAudit
from .weights import (
    clipped_perturbation_vertices,
    generate_candidate_weights,
    simplex_lattice,
)

__all__ = [
    "CallbackProblem",
    "CallableProblem",
    "ConfigurationError",
    "FrozenNormalizer",
    "KneeConfig",
    "LinearReduction",
    "MultiObjectiveProblem",
    "SelectionResult",
    "SolveError",
    "SolveResult",
    "TAFRKneeError",
    "TabularProblem",
    "WeightAudit",
    "__version__",
    "audit_weight",
    "clipped_perturbation_vertices",
    "discover_knees",
    "fit_normalizer",
    "generate_candidate_weights",
    "select_knee",
    "simplex_lattice",
]
