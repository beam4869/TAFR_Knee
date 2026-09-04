import numpy as np
import pytest

from tafrknee import KneeConfig, TabularProblem, audit_weight, discover_knees, select_knee
from tafrknee.benchmarks import plateau_knee


def plateau_config(**overrides: object) -> KneeConfig:
    values: dict[str, object] = {
        "radius": 0.05,
        "objective_tolerance": 1e-8,
        "extreme_threshold": 0.05,
        "kappa_min": 1.0,
        "candidate_resolution": 20,
        "audit_strategy": "vertices",
        "interior_samples": 0,
        "stability_tolerance": 1e-4,
        "exit_step": 0.01,
    }
    values.update(overrides)
    return KneeConfig(**values)


def test_audit_weight_recovers_plateau_and_exit_tradeoff() -> None:
    audit = audit_weight(plateau_knee(), [0.5, 0.5], config=plateau_config())
    assert audit.certified
    assert audit.decision == "knee"
    assert audit.robustness == pytest.approx(0.0)
    assert audit.stability_radius == pytest.approx(0.1, abs=2e-4)
    assert audit.exit_tradeoff == pytest.approx(1.5)


@pytest.mark.parametrize("mode", ["fixed_radius", "max_stability"])
def test_select_knee_uses_lexicographic_ranking(mode: str) -> None:
    result = select_knee(plateau_knee(), config=plateau_config(), mode=mode)
    assert result.found
    assert result.selected is not None
    np.testing.assert_allclose(result.selected.weight, [0.5, 0.5])
    assert result.selected.decision == "knee"
    assert result.knees == (result.selected,)
    assert result.solver_calls > 2
    assert result.cache_hits > 0


def test_extreme_only_problem_reports_no_certified_knee() -> None:
    problem = TabularProblem([[0.0, 1.0], [1.0, 0.0]])
    result = select_knee(
        problem,
        config=plateau_config(candidate_resolution=4),
        reference=[1.0, 1.0],
    )
    assert not result.found
    assert result.selected is None
    assert all(not audit.non_extreme for audit in result.audits)
    assert "No certified" in result.message


def test_discover_knees_returns_serializable_diagnostics() -> None:
    result = discover_knees(
        plateau_knee(),
        config=plateau_config(candidate_resolution=10),
        max_knees=3,
    )
    payload = result.to_dict()
    assert payload["found"] is True
    assert payload["selected"]["decision"] == "knee"
    assert payload["audits"]


def test_finite_radius_and_tradeoff_active_filters_both_apply() -> None:
    strict = select_knee(
        plateau_knee(),
        config=plateau_config(kappa_min=2.0, candidate_resolution=10),
    )
    assert strict.selected is None
    central = next(audit for audit in strict.audits if np.allclose(audit.weight, [0.5, 0.5]))
    assert central.non_extreme
    assert central.tradeoff_active
    assert "exit_tradeoff_below_threshold" in central.reasons


def test_adaptive_audit_remains_deterministic() -> None:
    settings = plateau_config(
        audit_strategy="adaptive",
        interior_samples=4,
        adaptive_rounds=1,
        adaptive_samples=4,
    )
    first = audit_weight(plateau_knee(), [0.5, 0.5], config=settings)
    second = audit_weight(plateau_knee(), [0.5, 0.5], config=settings)
    assert first.to_dict() == second.to_dict()
    assert first.perturbation_count > 3
