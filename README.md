# TAFR-Knee

**Trade-off-Active Finite-Radius Knee Detection via Preference Stability**

TAFR-Knee selects implementable compromise solutions without first constructing a
dense approximation of the entire Pareto frontier. It repeatedly solves a scalarized
problem near a nominal preference weight, measures the resulting movement in a
*frozen normalized objective space*, rejects trivial single-objective extremes, and
certifies the trade-off encountered when the optimizer exits a stable weight region.

The method is designed for continuous, nonsmooth, and mixed-integer models. The core
package only requires NumPy and SciPy; existing optimization models can be connected
through a solver callback.

## What is implemented

- Frozen ideal/pseudo-nadir objective normalization computed once from anchor solves.
- Deterministic solution selection delegated to explicit problem adapters.
- Complete vertices of the clipped zero-sum perturbation polytope, including the
  non-pairwise vertices required for four or more objectives.
- Pairwise, complete-vertex, hybrid vertex/interior, and adaptive adversarial
  finite-radius audits.
- Non-extreme filtering by normalized distance to the nearest anchor.
- Plateau-aware finite-radius displacement `R_rho` and estimated stability radius
  `r_epsilon`.
- Trade-off-active exit screening and weakest exit ratio `K_exit`.
- Lexicographic selection for a known radius or an unknown application radius.
- Optional objective reduction in weight space with certification in the original
  normalized objective space.
- Solve caching, neighboring warm starts, deterministic candidate designs, and
  optional threaded evaluation.

## Installation

From a clone of this repository:

```bash
python -m pip install -e .
```

Optional features:

```bash
python -m pip install -e '.[plot]'
python -m pip install -e '.[pyomo]'
python -m pip install -e '.[dev]'
```

TAFR-Knee requires Python 3.10 or newer. The distribution name is `tafr-knee`; the
Python import name is `tafrknee`.

## Quick start: evaluated points or a Pareto set

```python
from tafrknee import KneeConfig, TabularProblem, select_knee

problem = TabularProblem(
    objectives=[
        [0.0, 10.0],
        [4.0, 4.0],
        [10.0, 0.0],
    ],
    decisions=["f1 anchor", "candidate schedule", "f2 anchor"],
)

config = KneeConfig(
    radius=0.05,
    objective_tolerance=1e-8,
    extreme_threshold=0.05,
    kappa_min=1.0,
    candidate_resolution=20,
)
result = select_knee(problem, config=config)

if result.found:
    knee = result.selected
    print(knee.weight)
    print(knee.decision)
    print(knee.objectives)
    print(knee.robustness, knee.stability_radius, knee.exit_tradeoff)
else:
    print(result.message)
```

For this example, `(4, 4)` is optimal over a finite interval of weights. TAFR-Knee
treats this plateau as preference stability, then certifies the trade-off when the
solution changes to an anchor.

## Connect an existing optimizer

`CallbackProblem` is the general interface for LP, QP, NLP, MILP, MINLP, or custom
solvers. The callback receives coefficients for the *physical, unnormalized*
objectives. This conversion is what makes a normalized preference vector correspond
to the correct lower-level scalarized solve.

```python
import numpy as np

from tafrknee import CallbackProblem, SolveResult, select_knee


def solve_model(coefficients: np.ndarray, warm_start):
    # Build/update and solve your model using
    #   coefficients @ [f1(x), ..., fm(x)]
    # Apply the same deterministic secondary criterion at every weight.
    x = my_solver(coefficients, warm_start=warm_start)
    return SolveResult(
        decision=x,
        objectives=[f1(x), f2(x), f3(x)],
        metadata={"termination": "optimal"},
    )


problem = CallbackProblem(n_objectives=3, solver=solve_model)
result = select_knee(problem)
```

`CallableProblem` provides a SciPy adapter for bounded continuous problems.
`tafrknee.adapters.PyomoProblem` is available with the `pyomo` extra and uses a
user-owned model factory, so it does not impose a particular model structure.

## Public API

```python
audit = audit_weight(problem, weight=[0.5, 0.5], config=config)
selection = select_knee(problem, config=config)
discovery = discover_knees(problem, config=config, max_knees=5)
```

All three APIs accept:

- `ideal` and `reference` to replace payoff-matrix normalization with engineering
  bounds;
- `reduction` to search a lower-dimensional grouped-objective weight space; and
- `config` for uncertainty radii, tolerances, certification thresholds, candidate
  density, perturbation strategy, and parallelism.

`select_knee` and `discover_knees` return a `SelectionResult`. It includes every
`WeightAudit`, the frozen bounds and anchors, solve/cache counts, and serializable
`to_dict()` output. If no point satisfies all filters, `selected` is `None`; the
package never forces an uncertified answer.

## Selection modes

`mode="fixed_radius"` implements the known-preference-radius formulation. Among
certified candidates it lexicographically minimizes `R_rho`, then maximizes
`r_epsilon` and `K_exit`.

`mode="max_stability"` is intended when no application-specific radius is known. It
lexicographically maximizes `r_epsilon`, then minimizes `R_rho` and maximizes
`K_exit`.

The thresholds are not arbitrary tuning knobs: `radius` should describe plausible
preference uncertainty, `objective_tolerance` should reflect solver or engineering
indifference, `extreme_threshold` should exclude practically single-objective
solutions, and `kappa_min` should encode the minimum acceptable exit trade-off.

## Objective reduction

```python
from tafrknee import LinearReduction, select_knee

reduction = LinearReduction.from_groups(
    [[0, 1], [2, 3]],
    n_objectives=4,
    names=["cost-emissions", "water-safety"],
)
result = select_knee(problem, reduction=reduction)
```

The scalarized search occurs in the two-dimensional reduced space, but `R_rho`,
`r_epsilon`, anchor distance, and `K_exit` are evaluated using all four original
normalized objectives. Aggregation therefore cannot hide a severe within-group
deterioration during certification.

## Command line for CSV data

```bash
tafr-knee select objectives.csv --radius 0.05 --epsilon 0.001 --output result.json
tafr-knee discover objectives.csv --skiprows 1 --compact
```

Rows are feasible decisions and columns are minimization objectives. The exit code is
zero when a certified knee is found and two otherwise.

## Numerical interpretation and limitations

The default weighted-sum scalarization recovers supported Pareto points. Unsupported
knees on nonconvex objective images require a callback that exposes another
parameterization, such as an augmented Tchebycheff or epsilon-constraint solve; this
is planned for a later adapter API.

Complete perturbation vertices are a deterministic stress test, but they are not a
mathematical maximizer of `R_rho` when the optimizer map is nonlinear or
discontinuous. The default `hybrid` audit therefore adds deterministic-seed interior
hit-and-run samples. `audit_strategy="adaptive"` starts from the full hybrid audit and
iteratively samples from the most adverse weight found. Increase `interior_samples`
or `adaptive_samples`, compare strategies, and perform a sensitivity analysis before
treating a result as application-level certification.

The current `r_epsilon` and exit boundary are numerical estimates obtained by nested
finite-radius audits. Report the configuration and solver tolerances with scientific
results.

## Development

```bash
python -m pytest
python -m ruff check src tests examples
python -m build
```

See [VERSIONING.md](VERSIONING.md) for the release policy and [CHANGELOG.md](CHANGELOG.md)
for user-visible changes.

## Citation

If this implementation supports published work, cite the software metadata in
[`CITATION.cff`](CITATION.cff). The method is based on the finite-radius
preference-stable knee formulation by Hongxuan Wang and is informed by classical
knee-point definitions and the benchmark framework of Yu, Jin, and Olhofer (2020).

## License

MIT License. See [LICENSE](LICENSE).
