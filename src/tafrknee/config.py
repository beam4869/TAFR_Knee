"""Validated configuration for TAFR-Knee searches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .exceptions import ConfigurationError

AuditStrategy = Literal["pairwise", "vertices", "hybrid", "adaptive"]
SelectionMode = Literal["fixed_radius", "max_stability"]


@dataclass(frozen=True)
class KneeConfig:
    """Numerical controls and certification thresholds.

    All objective-space tolerances apply after frozen normalization.
    Weight-space quantities use the infinity norm on the simplex.
    """

    radius: float = 0.05
    objective_tolerance: float = 1e-3
    extreme_threshold: float = 0.05
    kappa_min: float = 1.0
    min_weight: float = 0.0
    distance_norm: float | int = 2
    audit_strategy: AuditStrategy = "hybrid"
    interior_samples: int = 16
    adaptive_rounds: int = 2
    adaptive_samples: int = 12
    candidate_resolution: int = 20
    max_candidates: int = 5_000
    stability_tolerance: float = 1e-3
    stability_iterations: int = 12
    exit_step: float = 0.01
    exit_layers: int = 4
    min_improvement: float = 1e-6
    min_deterioration: float = 1e-6
    tie_tolerance: float = 1e-10
    uniqueness_tolerance: float = 1e-3
    cache_decimals: int = 12
    random_seed: int = 0
    n_jobs: int = 1

    def validate(self, n_weights: int | None = None) -> KneeConfig:
        if not 0 < self.radius <= 1:
            raise ConfigurationError("radius must lie in (0, 1]")
        for name in (
            "objective_tolerance",
            "extreme_threshold",
            "kappa_min",
            "min_weight",
            "stability_tolerance",
            "exit_step",
            "min_improvement",
            "min_deterioration",
            "tie_tolerance",
            "uniqueness_tolerance",
        ):
            if getattr(self, name) < 0:
                raise ConfigurationError(f"{name} must be non-negative")
        if self.distance_norm not in (1, 2, float("inf")):
            raise ConfigurationError("distance_norm must be 1, 2, or infinity")
        if self.audit_strategy not in ("pairwise", "vertices", "hybrid", "adaptive"):
            raise ConfigurationError(
                "audit_strategy must be 'pairwise', 'vertices', 'hybrid', or 'adaptive'"
            )
        for name in (
            "candidate_resolution",
            "max_candidates",
            "stability_iterations",
            "exit_layers",
            "adaptive_samples",
            "n_jobs",
        ):
            if getattr(self, name) < 1:
                raise ConfigurationError(f"{name} must be at least one")
        if self.interior_samples < 0:
            raise ConfigurationError("interior_samples must be non-negative")
        if self.adaptive_rounds < 0:
            raise ConfigurationError("adaptive_rounds must be non-negative")
        if self.cache_decimals < 3:
            raise ConfigurationError("cache_decimals must be at least three")
        if n_weights is not None and n_weights * self.min_weight >= 1:
            raise ConfigurationError("n_weights * min_weight must be strictly less than one")
        return self
