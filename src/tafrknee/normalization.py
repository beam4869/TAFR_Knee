"""Frozen objective normalization based on single-objective anchors."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import ConfigurationError
from .problem import MultiObjectiveProblem
from .types import SolveResult


@dataclass(frozen=True)
class FrozenNormalizer:
    """Affine objective normalization whose constants never change by weight."""

    ideal: NDArray[np.float64]
    reference: NDArray[np.float64]

    def __post_init__(self) -> None:
        ideal = np.asarray(self.ideal, dtype=float).reshape(-1).copy()
        reference = np.asarray(self.reference, dtype=float).reshape(-1).copy()
        if ideal.size < 2 or reference.shape != ideal.shape:
            raise ConfigurationError("ideal and reference must have the same length >= 2")
        if not np.all(np.isfinite(ideal)) or not np.all(np.isfinite(reference)):
            raise ConfigurationError("normalization bounds must be finite")
        if np.any(reference <= ideal):
            bad = np.flatnonzero(reference <= ideal).tolist()
            raise ConfigurationError(
                "reference must be strictly greater than ideal for every objective; "
                f"remove or rescale effectively constant objectives at indices {bad}"
            )
        ideal.setflags(write=False)
        reference.setflags(write=False)
        object.__setattr__(self, "ideal", ideal)
        object.__setattr__(self, "reference", reference)

    @property
    def scale(self) -> NDArray[np.float64]:
        return self.reference - self.ideal

    @property
    def n_objectives(self) -> int:
        return self.ideal.size

    def transform(self, objectives: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(objectives, dtype=float)
        if values.shape[-1] != self.n_objectives:
            raise ConfigurationError("objective array has the wrong trailing dimension")
        return (values - self.ideal) / self.scale

    def inverse_transform(self, normalized: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(normalized, dtype=float)
        if values.shape[-1] != self.n_objectives:
            raise ConfigurationError("normalized array has the wrong trailing dimension")
        return self.ideal + values * self.scale

    def physical_scalar_weights(self, normalized_weights: ArrayLike) -> NDArray[np.float64]:
        weights = np.asarray(normalized_weights, dtype=float).reshape(-1)
        if weights.size != self.n_objectives:
            raise ConfigurationError("normalized weights have the wrong length")
        return weights / self.scale


def fit_normalizer(
    problem: MultiObjectiveProblem,
    *,
    ideal: ArrayLike | None = None,
    reference: ArrayLike | None = None,
) -> tuple[FrozenNormalizer, tuple[SolveResult, ...]]:
    """Solve anchors once and build a frozen pseudo-nadir normalization."""

    n_objectives = int(problem.n_objectives)
    if n_objectives < 2:
        raise ConfigurationError("a multiobjective problem needs at least two objectives")
    anchors: list[SolveResult] = []
    for index in range(n_objectives):
        weights = np.zeros(n_objectives)
        weights[index] = 1.0
        result = problem.solve(weights, None)
        if result.objectives.size != n_objectives:
            raise ConfigurationError("anchor solve returned the wrong number of objectives")
        anchors.append(result)
    payoff = np.vstack([anchor.objectives for anchor in anchors])
    fitted_ideal = np.diag(payoff) if ideal is None else np.asarray(ideal, dtype=float)
    fitted_reference = (
        np.max(payoff, axis=0) if reference is None else np.asarray(reference, dtype=float)
    )
    normalizer = FrozenNormalizer(fitted_ideal, fitted_reference)
    return normalizer, tuple(anchors)
