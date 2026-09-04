"""Optional Pyomo adapter using a user-owned deterministic model factory."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike, NDArray

from ..exceptions import ConfigurationError, SolveError
from ..problem import _validate_scalar_weights
from ..types import SolveResult


class PyomoProblem:
    """Bridge TAFR-Knee to Pyomo without constraining model structure.

    ``model_factory(weights, warm_start)`` must return a model containing the
    scalar objective for the supplied *physical* objective coefficients. It is
    also responsible for a deterministic secondary criterion when alternate
    optima are possible. The extraction callbacks run after the solve.
    """

    def __init__(
        self,
        *,
        n_objectives: int,
        model_factory: Callable[[NDArray[np.float64], Any | None], Any],
        objective_values: Callable[[Any], ArrayLike],
        decision: Callable[[Any], Any],
        solver: str,
        solver_options: dict[str, Any] | None = None,
        tee: bool = False,
    ) -> None:
        if n_objectives < 2:
            raise ConfigurationError("a multiobjective problem needs at least two objectives")
        self.n_objectives = n_objectives
        self.model_factory = model_factory
        self.objective_values = objective_values
        self.decision = decision
        self.solver = solver
        self.solver_options = dict(solver_options or {})
        self.tee = tee

    def solve(
        self,
        scalar_weights: NDArray[np.float64],
        warm_start: Any | None = None,
    ) -> SolveResult:
        try:
            import pyomo.environ as pyo
        except ImportError as error:  # pragma: no cover - dependency-specific
            raise ImportError("install TAFR-Knee with the 'pyomo' extra") from error
        weights = _validate_scalar_weights(scalar_weights, self.n_objectives)
        model = self.model_factory(weights.copy(), warm_start)
        solver = pyo.SolverFactory(self.solver)
        for name, value in self.solver_options.items():
            solver.options[name] = value
        raw_result = solver.solve(model, tee=self.tee)
        termination = raw_result.solver.termination_condition
        if termination not in {
            pyo.TerminationCondition.optimal,
            pyo.TerminationCondition.locallyOptimal,
            pyo.TerminationCondition.globallyOptimal,
        }:
            raise SolveError(f"Pyomo solve terminated with {termination}")
        objectives = np.asarray(self.objective_values(model), dtype=float).reshape(-1)
        if objectives.size != self.n_objectives or not np.all(np.isfinite(objectives)):
            raise SolveError("Pyomo objective extractor returned invalid values")
        return SolveResult(
            decision=self.decision(model),
            objectives=objectives,
            metadata={"solver": self.solver, "termination_condition": str(termination)},
        )
