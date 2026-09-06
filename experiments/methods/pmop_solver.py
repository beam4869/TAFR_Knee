"""Budgeted full-space PMOP solver; never labels heuristic solves as global optima."""
import hashlib
from time import perf_counter

import numpy as np
from scipy.optimize import differential_evolution

from tafrknee import SolveResult
from experiments.benchmarks.pmop_decision import objectives


class PMOPProblem:
    def __init__(self, number, m, *, seed=0, maxiter=40, popsize=5, linkage=False):
        self.number, self.n_objectives, self.n_variables = number, m, m + 9
        self.seed, self.maxiter, self.popsize = seed, maxiter, popsize
        self.linkage = linkage
        self.calls = 0
        self.objective_calls = 0
        self.logs = []

    def solve(self, coefficients, warm_start=None):
        w = np.asarray(coefficients, float)
        w = w / np.max(np.abs(w))
        start = perf_counter()
        evaluations = 0

        def cost(x):
            nonlocal evaluations
            x = np.asarray(x)
            batch = x[None, :] if x.ndim == 1 else x.T
            evaluations += len(batch)
            values = objectives(self.number, batch, self.n_objectives, self.linkage) @ w
            return float(values[0]) if x.ndim == 1 else values

        # A weight-specific deterministic seed makes reordering solver calls harmless.
        digest = hashlib.sha256(np.asarray(w, dtype='<f8').tobytes()).digest()
        seed = (self.seed + int.from_bytes(digest[:4], 'little')) % 2**32
        result = differential_evolution(
            cost, [(0., 1.)] * (self.n_objectives - 1) + [(0., 10.)] * 10,
            seed=seed, maxiter=self.maxiter, popsize=self.popsize, polish=True,
            vectorized=True, updating='deferred', tol=1e-7, atol=1e-9)
        values = objectives(self.number, result.x, self.n_objectives, self.linkage)[0]
        self.calls += 1
        self.objective_calls += evaluations
        metadata = dict(solver='scipy-DE-plus-LBFGSB', converged=bool(result.success),
                        message=str(result.message), objective_vector_evaluations=evaluations,
                        seed=seed, iterations=int(result.nit), wall_time_seconds=perf_counter()-start,
                        global_optimality_proven=False, distance_variables_optimized=10,
                        full_decision_dimension=self.n_variables)
        self.logs.append(metadata)
        return SolveResult(result.x, values, status='heuristic_feasible', metadata=metadata)
