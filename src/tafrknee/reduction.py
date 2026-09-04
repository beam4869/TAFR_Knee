"""Linear objective-reduction hooks for low-dimensional weight searches."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import ConfigurationError


@dataclass(frozen=True)
class LinearReduction:
    """Map normalized original objectives to reduced objectives ``G = A F``.

    Rows define non-negative within-group aggregations. Certification remains
    in the original normalized objective space.
    """

    matrix: NDArray[np.float64]
    names: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        matrix = np.asarray(self.matrix, dtype=float).copy()
        if matrix.ndim != 2 or min(matrix.shape) < 1:
            raise ConfigurationError("reduction matrix must be two-dimensional")
        if not np.all(np.isfinite(matrix)) or np.any(matrix < 0):
            raise ConfigurationError("reduction matrix must be finite and non-negative")
        row_sums = np.sum(matrix, axis=1)
        if np.any(row_sums <= 0):
            raise ConfigurationError("every reduced objective must contain an original objective")
        matrix /= row_sums[:, None]
        matrix.setflags(write=False)
        object.__setattr__(self, "matrix", matrix)
        if self.names is not None:
            if len(self.names) != matrix.shape[0]:
                raise ConfigurationError("reduction names have the wrong length")
            object.__setattr__(self, "names", tuple(self.names))

    @property
    def n_reduced(self) -> int:
        return self.matrix.shape[0]

    @property
    def n_original(self) -> int:
        return self.matrix.shape[1]

    def original_weights(self, reduced_weights: ArrayLike) -> NDArray[np.float64]:
        weights = np.asarray(reduced_weights, dtype=float).reshape(-1)
        if weights.size != self.n_reduced:
            raise ConfigurationError("reduced weights have the wrong length")
        return self.matrix.T @ weights

    def transform(self, normalized_objectives: ArrayLike) -> NDArray[np.float64]:
        values = np.asarray(normalized_objectives, dtype=float)
        if values.shape[-1] != self.n_original:
            raise ConfigurationError("objective vector has the wrong length")
        return values @ self.matrix.T

    @classmethod
    def from_groups(
        cls,
        groups: Sequence[Sequence[int]],
        *,
        n_objectives: int,
        within_group_weights: Sequence[Sequence[float]] | None = None,
        names: Sequence[str] | None = None,
    ) -> LinearReduction:
        if n_objectives < 2 or len(groups) < 2:
            raise ConfigurationError("reduction needs at least two original and reduced objectives")
        matrix = np.zeros((len(groups), n_objectives), dtype=float)
        seen: set[int] = set()
        for row, group in enumerate(groups):
            indices = tuple(int(index) for index in group)
            if not indices:
                raise ConfigurationError("objective groups cannot be empty")
            if any(index < 0 or index >= n_objectives for index in indices):
                raise ConfigurationError("objective group contains an invalid index")
            overlap = seen.intersection(indices)
            if overlap:
                raise ConfigurationError(f"objective groups overlap at indices {sorted(overlap)}")
            seen.update(indices)
            if within_group_weights is None:
                weights = np.ones(len(indices), dtype=float)
            else:
                weights = np.asarray(within_group_weights[row], dtype=float).reshape(-1)
                if weights.size != len(indices):
                    raise ConfigurationError("within-group weights have the wrong length")
            matrix[row, indices] = weights
        if seen != set(range(n_objectives)):
            missing = sorted(set(range(n_objectives)) - seen)
            raise ConfigurationError(f"objective groups do not cover indices {missing}")
        return cls(matrix=matrix, names=None if names is None else tuple(names))


def identity_reduction(n_objectives: int) -> LinearReduction:
    if n_objectives < 2:
        raise ConfigurationError("a multiobjective problem needs at least two objectives")
    return LinearReduction(np.eye(n_objectives))
