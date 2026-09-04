"""Small deterministic examples for smoke tests and tutorials."""

from __future__ import annotations

import numpy as np

from .problem import CallableProblem, TabularProblem


def plateau_knee() -> TabularProblem:
    """Three-point problem with a supported interior weight plateau.

    The point (4, 4) is optimal for weights between 0.4 and 0.6. Exiting
    to either anchor improves one normalized objective by 0.4 and worsens
    the other by 0.6, yielding an exit trade-off ratio of 1.5.
    """

    return TabularProblem(
        [[0.0, 10.0], [4.0, 4.0], [10.0, 0.0]],
        decisions=["f1_anchor", "knee", "f2_anchor"],
        objective_names=["f1", "f2"],
    )


def smooth_biobjective() -> CallableProblem:
    """One-dimensional smooth trade-off: ``(x^2, (1-x)^2)``."""

    return CallableProblem(
        lambda x: np.array([x[0] ** 2, (1.0 - x[0]) ** 2]),
        n_objectives=2,
        bounds=[(0.0, 1.0)],
        x0=[[0.1], [0.5], [0.9]],
    )
