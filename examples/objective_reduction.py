"""Search reduced weights while retaining original-space certification."""

from tafrknee import KneeConfig, LinearReduction, TabularProblem, select_knee

problem = TabularProblem(
    [
        [0.0, 6.0, 8.0, 8.0],
        [6.0, 0.0, 8.0, 8.0],
        [8.0, 8.0, 0.0, 6.0],
        [8.0, 8.0, 6.0, 0.0],
        [4.0, 4.0, 4.0, 4.0],
    ],
    decisions=["f1_anchor", "f2_anchor", "f3_anchor", "f4_anchor", "knee"],
)
reduction = LinearReduction.from_groups([[0, 1], [2, 3]], n_objectives=4)
result = select_knee(
    problem,
    reduction=reduction,
    config=KneeConfig(candidate_resolution=10, kappa_min=0.0),
)
print(result.message)
if result.selected is not None:
    print(result.selected.to_dict())
