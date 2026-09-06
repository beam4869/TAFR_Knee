"""Exact matrices specified in the common-core experiment plan."""
import numpy as np
from tafrknee import TabularProblem
from experiments.metrics.exact_tabular_oracle import canonical_priorities


def extreme_plateau(scale=1.0):
    return np.array([[0, 10 * scale], [4, 4 * scale], [10, 0]], float)


def incomplete_vertices():
    anchors = 10 * (np.ones((4, 4)) - np.eye(4))
    return np.vstack([anchors, [5, 5, 5, 5], [4.3, 4.3, 6.3, 6.3]])


def interior_cell():
    return np.array([[0, 10, 10], [10, 0, 10], [10, 10, 0], [3, 3, 3],
                     [1, 4, 5], [2.3426, 3.6574, 3.0600], [2.1564, 2.4325, 4.4712]])


def as_problem(y):
    return TabularProblem(y, tie_break_values=canonical_priorities(y), primary_tolerance=1e-12)
