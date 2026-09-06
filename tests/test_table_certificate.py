"""Analytic counterexamples and independent evidence checks for the research gate."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from experiments.benchmarks.adversarial_tabular import (
    extreme_plateau,
    incomplete_vertices,
    interior_cell,
)
from experiments.methods.table_certificate import (
    CheckedTableOracle,
    certify_table_weight,
    radius_evidence,
)
from tafrknee import KneeConfig
from tafrknee.weights import perturbation_weights


def test_samples_cannot_prove_a_bound_and_violations_refute_it():
    assert radius_evidence(.1, lower_bound=.02) == "unverified"
    assert radius_evidence(.1, lower_bound=.11) == "refuted"
    assert radius_evidence(.1, lower_bound=.02, upper_bound=.09) == "verified"
    with pytest.raises(ValueError, match="Inconsistent"):
        radius_evidence(.1, lower_bound=.2, upper_bound=.1)


def test_analytic_plateau_positive_control_and_closed_ball_endpoint():
    oracle = CheckedTableOracle(extreme_plateau() / 10)
    config = KneeConfig(min_improvement=.001, min_deterioration=.001)
    result = certify_table_weight(oracle, [.5, .5], config)
    assert result["accepted"]
    assert result["R_exact"] == 0
    assert result["exact_stability_radius"] == pytest.approx(.1, abs=1e-9)
    assert result["exit_tradeoff"] == pytest.approx(1.5)
    # At rho itself the lexicographic tie includes an extreme output. A radius
    # supremum is not a guarantee for the closed weight ball at that endpoint.
    boundary = certify_table_weight(oracle, [.5, .5], replace(config, radius=.1))
    assert boundary["radius_status"] == "refuted"
    assert not boundary["accepted"]


@pytest.mark.parametrize("family,radius,epsilon", [("C", .1, .2), ("D", .18, .28)])
def test_missing_outputs_reject_a_sampled_passing_bound(family, radius, epsilon):
    y = (incomplete_vertices() if family == "C" else interior_cell()) / 10
    w = np.full(y.shape[1], 1 / y.shape[1])
    oracle = CheckedTableOracle(y)
    strategy = "pairwise" if family == "C" else "vertices"
    weights = perturbation_weights(w, radius, strategy=strategy)
    selected = [oracle.select(u) for u in weights]
    sampled = np.linalg.norm(y[selected] - y[oracle.select(w)], axis=1).max()
    assert sampled < epsilon
    checked = certify_table_weight(oracle, w, KneeConfig(radius=radius,
                                                        objective_tolerance=epsilon))
    assert checked["R_exact"] > epsilon
    assert checked["radius_status"] == "refuted"


def test_inactive_first_shell_exit_is_retained():
    y = np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0],
                  [.5, .5, .5], [.5 - 1e-5, .9, .5]])
    result = certify_table_weight(CheckedTableOracle(y), [.499975, .00005, .499975],
                                 KneeConfig(radius=1e-6, min_improvement=.001,
                                            min_deterioration=.001))
    assert result["radius_status"] == "verified"
    assert any(not e["active"] for e in result["exits"])
    assert "inactive_first_shell_exit" in result["reasons"]
    assert not result["accepted"]


def test_lp_numerical_failure_never_becomes_a_certificate():
    with (
        patch("experiments.metrics.exact_tabular_oracle.linprog",
              return_value=SimpleNamespace(status=4, message="numerical failure")),
        pytest.raises(RuntimeError, match="Unresolved cell LP"),
    ):
        certify_table_weight(CheckedTableOracle(extreme_plateau() / 10), [.5, .5], KneeConfig())
