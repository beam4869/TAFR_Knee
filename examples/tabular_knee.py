"""Select a knee from an evaluated objective table."""

from tafrknee import KneeConfig, TabularProblem, select_knee

problem = TabularProblem(
    [[0.0, 10.0], [4.0, 4.0], [10.0, 0.0]],
    decisions=["f1_anchor", "balanced_schedule", "f2_anchor"],
)
config = KneeConfig(
    radius=0.05,
    objective_tolerance=1e-8,
    extreme_threshold=0.05,
    kappa_min=1.0,
    candidate_resolution=20,
)
result = select_knee(problem, config=config)

print(result.message)
if result.selected is not None:
    print(result.selected.to_dict())
