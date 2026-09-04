"""Connect an existing optimizer through CallbackProblem."""

import numpy as np

from tafrknee import CallbackProblem, KneeConfig, SolveResult, select_knee

OBJECTIVES = np.array([[0.0, 10.0], [4.0, 4.0], [10.0, 0.0]])


def existing_solver(coefficients: np.ndarray, warm_start: int | None) -> SolveResult:
    """Replace this finite example with a model update and optimizer call."""

    del warm_start
    index = int(np.argmin(OBJECTIVES @ coefficients))
    return SolveResult(
        decision=index,
        objectives=OBJECTIVES[index],
        metadata={"row_index": index},
    )


problem = CallbackProblem(n_objectives=2, solver=existing_solver)
result = select_knee(
    problem,
    config=KneeConfig(
        objective_tolerance=1e-8,
        candidate_resolution=20,
        audit_strategy="vertices",
    ),
)
print(result.to_dict(include_audits=False))
