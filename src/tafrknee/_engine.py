"""Cached conversion from preference weights to lower-level solves."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from .exceptions import SolveError
from .normalization import FrozenNormalizer
from .problem import MultiObjectiveProblem
from .reduction import LinearReduction
from .types import SolveResult
from .weights import validate_weight


class EvaluationEngine:
    def __init__(
        self,
        problem: MultiObjectiveProblem,
        normalizer: FrozenNormalizer,
        reduction: LinearReduction,
        *,
        cache_decimals: int = 12,
        n_jobs: int = 1,
    ) -> None:
        self.problem = problem
        self.normalizer = normalizer
        self.reduction = reduction
        self.cache_decimals = cache_decimals
        self.n_jobs = n_jobs
        self._cache: dict[tuple[float, ...], SolveResult] = {}
        self._weight_cache: dict[tuple[float, ...], NDArray[np.float64]] = {}
        self._lock = Lock()
        self.solver_calls = 0
        self.cache_hits = 0

    def _key(self, weight: ArrayLike) -> tuple[float, ...]:
        return tuple(np.round(np.asarray(weight, dtype=float), self.cache_decimals).tolist())

    def solve(self, weight: ArrayLike, *, allow_warm_start: bool = True) -> SolveResult:
        preference = validate_weight(weight, self.reduction.n_reduced)
        key = self._key(preference)
        with self._lock:
            cached = self._cache.get(key)
            if cached is not None:
                self.cache_hits += 1
                return cached
            warm_start = self._nearest_decision(preference) if allow_warm_start else None

        normalized_original_weights = self.reduction.original_weights(preference)
        physical_weights = self.normalizer.physical_scalar_weights(normalized_original_weights)
        result = self.problem.solve(physical_weights, warm_start)
        if result.objectives.size != self.normalizer.n_objectives:
            raise SolveError("lower-level solve returned the wrong number of objectives")
        with self._lock:
            existing = self._cache.get(key)
            if existing is not None:
                self.cache_hits += 1
                return existing
            self._cache[key] = result
            self._weight_cache[key] = preference.copy()
            self.solver_calls += 1
        return result

    def solve_many(self, weights: ArrayLike) -> tuple[SolveResult, ...]:
        values = np.asarray(weights, dtype=float)
        if values.ndim == 1:
            values = values[None, :]
        if self.n_jobs == 1 or values.shape[0] <= 1:
            return tuple(self.solve(weight) for weight in values)
        with ThreadPoolExecutor(max_workers=self.n_jobs) as executor:
            return tuple(
                executor.map(lambda weight: self.solve(weight, allow_warm_start=False), values)
            )

    def _nearest_decision(self, weight: NDArray[np.float64]) -> Any | None:
        if not self._weight_cache:
            return None
        key = min(
            self._weight_cache,
            key=lambda item: float(np.linalg.norm(self._weight_cache[item] - weight)),
        )
        return self._cache[key].decision
