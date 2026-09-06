"""Strictly convex simplex-QP companion to the interior-cell example."""
import numpy as np
from tafrknee import CallbackProblem, SolveResult


def simplex_projection(v):
    v = np.asarray(v, float)
    sorted_v = np.sort(v)[::-1]
    cssv = np.cumsum(sorted_v) - 1
    indices = np.arange(1, len(v) + 1)
    rho = np.flatnonzero(sorted_v > cssv / indices)[-1]
    return np.maximum(v - cssv[rho] / (rho + 1), 0)


def smooth_companion(y, tau):
    y = np.asarray(y, float)

    def solve(coefficients, warm_start):
        # Sum_i c_i f_i(p) = (Yc)'p + tau*sum(c)/2 * ||p||^2.
        p = simplex_projection(-(y @ coefficients) / (tau * coefficients.sum()))
        objectives = p @ y + tau * np.dot(p, p) / 2
        return SolveResult(p, objectives, metadata={"solver": "analytic simplex projection"})

    return CallbackProblem(n_objectives=y.shape[1], solver=solve)
