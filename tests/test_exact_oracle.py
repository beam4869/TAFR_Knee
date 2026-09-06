import numpy as np

from experiments.benchmarks.adversarial_tabular import (
    extreme_plateau,
    incomplete_vertices,
    interior_cell,
)
from experiments.metrics.exact_tabular_oracle import ExactTabularOracle
from tafrknee.weights import perturbation_weights


def test_interval_and_tie_reachability():
    o = ExactTabularOracle(extreme_plateau() / 10)
    assert np.isclose(o.stability_radius([.5, .5]), .1)
    assert o.audit([.5, .5], .05)["robustness"] == 0
    # Duplicate rows must not inflate reachable-solution cardinality.
    duplicate = ExactTabularOracle([[0, 1], [0, 1], [1, 0]])
    assert len(duplicate.reachable([.5, .5], .5)[0]) == 2


def test_three_dimensional_grid_parity():
    y = interior_cell() / 10
    o = ExactTabularOracle(y)
    w = np.full(3, 1 / 3)
    exact = o.audit(w, .18)
    grid = np.array([[a, b, 1-a-b] for a in np.linspace(w[0]-.18, w[0]+.18, 241)
                     for b in np.linspace(w[1]-.18, w[1]+.18, 241)
                     if abs(1-a-b-w[2]) <= .18 + 1e-12])
    idx = np.argmin(grid @ y.T, axis=1)
    assert set(idx) == set(exact["reachable"])
    assert np.isclose(exact["robustness"], .3)


def test_four_objective_trap():
    y = incomplete_vertices() / 10
    w = np.full(4, .25)
    assert ExactTabularOracle(y).audit(w, .1)["worst_index"] == 5
    pair = perturbation_weights(w, .1, strategy="pairwise")
    assert set(np.argmin(pair @ y.T, axis=1)) == {4}


def test_two_dimensional_exhaustive_grid():
    y = extreme_plateau() / 10
    for w1 in [.1, .3, .5, .7, .9]:
        o = ExactTabularOracle(y)
        center = np.array([w1, 1-w1])
        x = np.linspace(max(0, w1-.151), min(1, w1+.151), 10001)
        indices = np.argmin(np.column_stack([x, 1-x]) @ y.T, axis=1)
        assert set(indices) == set(o.reachable(center, .151)[0])
