import numpy as np
import pytest

from tafrknee import CallableProblem, CallbackProblem, SolveError, SolveResult, TabularProblem


def test_tabular_problem_uses_deterministic_secondary_values() -> None:
    problem = TabularProblem(
        [[1.0, 1.0], [1.0, 1.0]],
        decisions=["second", "first"],
        tie_break_values=[2.0, 1.0],
    )
    assert problem.solve(np.array([0.5, 0.5])).decision == "first"


def test_tabular_tie_break_never_selects_a_dominated_row() -> None:
    problem = TabularProblem(
        [[0.0, 2.0], [0.0, 1.0]],
        decisions=["dominated", "efficient"],
        tie_break_values=[0.0, 1.0],
    )
    assert problem.solve(np.array([1.0, 0.0])).decision == "efficient"


def test_callback_problem_passes_weights_and_warm_start() -> None:
    calls: list[tuple[np.ndarray, object]] = []

    def solver(weights: np.ndarray, warm_start: object) -> SolveResult:
        calls.append((weights, warm_start))
        return SolveResult("x", [weights[0], weights[1]])

    problem = CallbackProblem(2, solver)
    result = problem.solve(np.array([2.0, 3.0]), warm_start="previous")
    assert result.decision == "x"
    np.testing.assert_allclose(calls[0][0], [2.0, 3.0])
    assert calls[0][1] == "previous"


def test_callback_problem_rejects_failed_status() -> None:
    problem = CallbackProblem(2, lambda _w, _x: SolveResult(None, [0, 0], status="failed"))
    with pytest.raises(SolveError):
        problem.solve(np.ones(2))


def test_callable_problem_finds_expected_weighted_sum_solution() -> None:
    problem = CallableProblem(
        lambda x: [x[0] ** 2, (1 - x[0]) ** 2],
        n_objectives=2,
        bounds=[(0.0, 1.0)],
        x0=[[0.1], [0.5], [0.9]],
    )
    result = problem.solve(np.array([1.0, 1.0]))
    assert result.decision[0] == pytest.approx(0.5, abs=1e-5)
    np.testing.assert_allclose(result.objectives, [0.25, 0.25], atol=1e-5)
