"""Scientific contract for the budgeted, full-space lower-level adapter."""
import numpy as np

from experiments.methods.pmop_solver import PMOPProblem


def test_direct_solver_rescaling_and_honest_status():
    problem = PMOPProblem(2, 3, seed=12, maxiter=2, popsize=3)
    first = problem.solve([1., 2., 1.])
    problem.solve([.1, .8, .1])
    again = problem.solve([2., 4., 2.])
    np.testing.assert_allclose(first.objectives, again.objectives, atol=1e-12)
    assert first.status == "heuristic_feasible"
    assert not first.metadata["global_optimality_proven"]
    assert first.metadata["objective_vector_evaluations"] > 12
    assert first.decision.shape == (12,)
    assert np.all(first.decision >= 0)
    assert np.all(first.decision[:2] <= 1) and np.all(first.decision[2:] <= 10)
