"""Problem protocols and adapters for common input styles."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import Bounds, minimize

from .exceptions import ConfigurationError, SolveError
from .types import SolveResult


@runtime_checkable
class MultiObjectiveProblem(Protocol):
    """Minimal solver-agnostic problem interface.

    ``scalar_weights`` multiply physical, unnormalized objectives. TAFR-Knee
    handles the conversion from normalized preference weights internally.
    """

    n_objectives: int

    def solve(
        self,
        scalar_weights: NDArray[np.float64],
        warm_start: Any | None = None,
    ) -> SolveResult:
        """Solve one scalarized problem and return all physical objectives."""


@dataclass
class CallbackProblem:
    """Adapter for an existing LP/QP/NLP/MILP/MINLP solver callback."""

    n_objectives: int
    solver: Callable[[NDArray[np.float64], Any | None], SolveResult | tuple[Any, ArrayLike]]

    def __post_init__(self) -> None:
        if self.n_objectives < 2:
            raise ConfigurationError("a multiobjective problem needs at least two objectives")

    def solve(
        self,
        scalar_weights: NDArray[np.float64],
        warm_start: Any | None = None,
    ) -> SolveResult:
        weights = _validate_scalar_weights(scalar_weights, self.n_objectives)
        result = self.solver(weights.copy(), warm_start)
        if not isinstance(result, SolveResult):
            decision, objectives = result
            result = SolveResult(decision=decision, objectives=objectives)
        if result.objectives.size != self.n_objectives:
            raise SolveError(
                f"callback returned {result.objectives.size} objectives; "
                f"expected {self.n_objectives}"
            )
        if result.status.lower() not in {"optimal", "success", "ok"}:
            raise SolveError(f"callback solve did not succeed: {result.status}")
        return result


class TabularProblem:
    """Finite set of already evaluated feasible objective vectors.

    This adapter is appropriate for Pareto sets, benchmark outputs, CSV data,
    and candidate schedules. Ties are broken by ``tie_break_values`` and then
    by row index, so repeated calls are deterministic.
    """

    def __init__(
        self,
        objectives: ArrayLike,
        *,
        decisions: Sequence[Any] | None = None,
        tie_break_values: ArrayLike | None = None,
        objective_names: Sequence[str] | None = None,
        primary_tolerance: float = 1e-12,
    ) -> None:
        values = np.asarray(objectives, dtype=float)
        if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
            raise ConfigurationError("objectives must have shape (n_points, n_objectives>=2)")
        if not np.all(np.isfinite(values)):
            raise ConfigurationError("objectives must contain only finite values")
        self.objectives = values.copy()
        self.n_objectives = values.shape[1]
        self.decisions = tuple(range(values.shape[0])) if decisions is None else tuple(decisions)
        if len(self.decisions) != values.shape[0]:
            raise ConfigurationError("decisions must have one entry per objective row")
        if tie_break_values is None:
            self.tie_break_values = np.arange(values.shape[0], dtype=float)
        else:
            tie = np.asarray(tie_break_values, dtype=float).reshape(-1)
            if tie.size != values.shape[0] or not np.all(np.isfinite(tie)):
                raise ConfigurationError("tie_break_values must be finite with one value per row")
            self.tie_break_values = tie.copy()
        if objective_names is None:
            self.objective_names = tuple(f"f{i + 1}" for i in range(self.n_objectives))
        else:
            if len(objective_names) != self.n_objectives:
                raise ConfigurationError("objective_names has the wrong length")
            self.objective_names = tuple(str(name) for name in objective_names)
        if primary_tolerance < 0:
            raise ConfigurationError("primary_tolerance must be non-negative")
        self.primary_tolerance = float(primary_tolerance)

    def solve(
        self,
        scalar_weights: NDArray[np.float64],
        warm_start: Any | None = None,
    ) -> SolveResult:
        del warm_start
        weights = _validate_scalar_weights(scalar_weights, self.n_objectives)
        scalar_values = self.objectives @ weights
        best = float(np.min(scalar_values))
        tolerance = self.primary_tolerance * max(1.0, abs(best))
        tied = np.flatnonzero(scalar_values <= best + tolerance)
        tied = _nondominated_indices(self.objectives, tied)
        order = np.lexsort((tied, self.tie_break_values[tied]))
        index = int(tied[order[0]])
        return SolveResult(
            decision=self.decisions[index],
            objectives=self.objectives[index],
            metadata={"row_index": index, "scalar_value": float(scalar_values[index])},
        )


class CallableProblem:
    """SciPy-backed adapter for bounded continuous decision problems.

    Multiple starts are solved independently. Solutions within
    ``primary_tolerance`` of the best scalar value are selected using a fixed
    secondary function, then lexicographically by the decision vector.
    """

    def __init__(
        self,
        objective: Callable[[NDArray[np.float64]], ArrayLike],
        *,
        n_objectives: int,
        bounds: Sequence[tuple[float, float]] | Bounds,
        x0: ArrayLike | Sequence[ArrayLike] | None = None,
        constraints: Sequence[Any] = (),
        method: str = "SLSQP",
        options: dict[str, Any] | None = None,
        tie_breaker: Callable[[NDArray[np.float64]], float] | None = None,
        primary_tolerance: float = 1e-8,
    ) -> None:
        if n_objectives < 2:
            raise ConfigurationError("a multiobjective problem needs at least two objectives")
        self.objective = objective
        self.n_objectives = int(n_objectives)
        self.bounds = bounds if isinstance(bounds, Bounds) else Bounds(*zip(*bounds, strict=True))
        self.n_variables = np.asarray(self.bounds.lb).size
        self.starts = _prepare_starts(x0, self.bounds)
        self.constraints = tuple(constraints)
        self.method = method
        self.options = {"maxiter": 1000, "ftol": 1e-10, **(options or {})}
        self.tie_breaker = tie_breaker or (lambda x: float(np.sum(self._objectives(x))))
        if primary_tolerance < 0:
            raise ConfigurationError("primary_tolerance must be non-negative")
        self.primary_tolerance = float(primary_tolerance)
        test = self._objectives(self.starts[0])
        if test.size != self.n_objectives:
            raise ConfigurationError(
                f"objective returned {test.size} values; expected {self.n_objectives}"
            )

    def _objectives(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        values = np.asarray(self.objective(np.asarray(x, dtype=float)), dtype=float).reshape(-1)
        if values.size != self.n_objectives or not np.all(np.isfinite(values)):
            raise SolveError("objective callback returned invalid values")
        return values

    def solve(
        self,
        scalar_weights: NDArray[np.float64],
        warm_start: Any | None = None,
    ) -> SolveResult:
        weights = _validate_scalar_weights(scalar_weights, self.n_objectives)
        starts = list(self.starts)
        if warm_start is not None:
            warm = np.asarray(warm_start, dtype=float).reshape(-1)
            if warm.size == self.n_variables:
                starts.insert(0, np.clip(warm, self.bounds.lb, self.bounds.ub))

        successful: list[tuple[float, float, tuple[float, ...], Any]] = []
        failures: list[str] = []
        for start in starts:
            result = minimize(
                lambda x: float(weights @ self._objectives(x)),
                x0=start,
                method=self.method,
                bounds=self.bounds,
                constraints=self.constraints,
                options=self.options,
            )
            if result.success and np.all(np.isfinite(result.x)):
                x = np.asarray(result.x, dtype=float)
                primary = float(weights @ self._objectives(x))
                successful.append((primary, float(self.tie_breaker(x)), tuple(x.tolist()), result))
            else:
                failures.append(str(result.message))
        if not successful:
            raise SolveError("all SciPy starts failed: " + "; ".join(failures))

        best_primary = min(item[0] for item in successful)
        tolerance = self.primary_tolerance * max(1.0, abs(best_primary))
        eligible = [item for item in successful if item[0] <= best_primary + tolerance]
        primary, _, _, optimizer_result = min(eligible, key=lambda item: (item[1], item[2]))
        decision = np.asarray(optimizer_result.x, dtype=float)
        return SolveResult(
            decision=decision,
            objectives=self._objectives(decision),
            metadata={
                "scalar_value": primary,
                "method": self.method,
                "iterations": int(getattr(optimizer_result, "nit", 0)),
                "message": str(optimizer_result.message),
            },
        )


def _validate_scalar_weights(weights: ArrayLike, n_objectives: int) -> NDArray[np.float64]:
    array = np.asarray(weights, dtype=float).reshape(-1)
    if array.size != n_objectives:
        raise ConfigurationError(
            f"scalar weights have length {array.size}; expected {n_objectives}"
        )
    if not np.all(np.isfinite(array)) or np.any(array < 0) or not np.any(array > 0):
        raise ConfigurationError("scalar weights must be finite, non-negative, and nonzero")
    return array


def _prepare_starts(
    x0: ArrayLike | Sequence[ArrayLike] | None,
    bounds: Bounds,
) -> tuple[NDArray[np.float64], ...]:
    lb = np.asarray(bounds.lb, dtype=float)
    ub = np.asarray(bounds.ub, dtype=float)
    if lb.shape != ub.shape or lb.ndim != 1 or np.any(lb > ub):
        raise ConfigurationError("bounds are invalid")
    if not np.all(np.isfinite(lb)) or not np.all(np.isfinite(ub)):
        raise ConfigurationError("CallableProblem currently requires finite bounds")
    if x0 is None:
        return ((lb + ub) / 2.0,)
    array = np.asarray(x0, dtype=float)
    if array.ndim == 1:
        array = array[None, :]
    if array.ndim != 2 or array.shape[1] != lb.size or not np.all(np.isfinite(array)):
        raise ConfigurationError("x0 must contain one or more finite decision vectors")
    if np.any(array < lb) or np.any(array > ub):
        raise ConfigurationError("all x0 values must satisfy the bounds")
    return tuple(row.copy() for row in array)


def _nondominated_indices(
    objectives: NDArray[np.float64], indices: NDArray[np.int64]
) -> NDArray[np.int64]:
    """Remove dominated alternatives before applying a deterministic tie-break."""

    keep: list[int] = []
    for index in indices:
        others = objectives[indices]
        candidate = objectives[index]
        dominated = np.any(np.all(others <= candidate, axis=1) & np.any(others < candidate, axis=1))
        if not dominated:
            keep.append(int(index))
    return np.asarray(keep, dtype=int)
