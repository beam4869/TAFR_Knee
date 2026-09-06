"""Independent LP parity for the bounded-simplex stability calculation."""
import numpy as np
import pytest

from experiments.metrics.exact_tabular_oracle import ExactTabularOracle


@pytest.mark.parametrize("m", [2, 3, 5])
def test_fast_stability_matches_cell_union_lp(m):
    rng = np.random.default_rng(400 + m)
    y = rng.uniform(size=(12, m))
    oracle = ExactTabularOracle(y)
    centers = [np.full(m, 1 / m), *np.eye(m), *rng.dirichlet(np.ones(m), size=5)]
    for w in centers:
        expected = oracle.stability_radius(w, 1e-3)
        assert oracle.stability_radius_fast(w, 1e-3) == pytest.approx(expected, abs=2e-8)


def test_close_outputs_fall_back_to_union_lp():
    y = np.array([[0., 1.], [.4, .4], [.4001, .3999], [1., 0.]])
    oracle = ExactTabularOracle(y)
    w = np.array([.5, .5])
    assert oracle.stability_radius_fast(w, .001) == oracle.stability_radius(w, .001)


def test_dominated_and_duplicate_competitors_do_not_create_false_exits():
    oracle = ExactTabularOracle([[0., 0.], [0., 0.], [0., 1.], [1., 0.]])
    assert oracle.stability_radius_fast([.5, .5]) == .5
