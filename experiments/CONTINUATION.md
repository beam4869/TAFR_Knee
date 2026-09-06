# Experiment continuation, 6 September 2026

The subsequent [certification gate study](CERTIFICATION_GATE.md) completes a
paired comparison of radius and exit conditions. It also fixes a strict-margin
omission in a separate conservative finite-table oracle. The historical PMOP
LP audits below require conservative revalidation before supporting certificates.
Its new raw archive awaits explicit public-upload authorization; code and
aggregate outcomes accompany the study.

The production package remains at v0.1.0 and its source is byte-for-byte unchanged
from `9f7a65d7ffeaddedf2969b7db3d764623e0cca27`. This continuation implements
additional experiments in separate, named variants. It is not a final-paper
performance release.

## Executed extension

| Component | Configuration and evidence |
|---|---|
| MATLAB/Python parity | 126 checks: PMOP1-14, m=3/5/8, PF and CalObj with linkage off/on, executed with Octave 10.3.0 |
| SNEE Hessian control | 124 GRV2 jobs: 30 paired NM starts in two normalizations and two Hessian variants, plus four DIRECT jobs |
| PMOP Track A | 42 fixed finite tables, 30 paired seeds, 17 candidate weights; every returned weighted-sum selection receives an exact finite-table LP displacement audit |
| Component ablations | 10 mechanism/discrete cases, 30 seeds, 11 variants: 3,300 method records, including direct AWT selection |
| Ammonia | All 12 stratified 48-hour windows; fixed forecasts and the original MILP objectives |
| PMOP Track B pilot | Eight representative three-objective problems; all m+9 decision variables optimized, with explicit heuristic lower-level status |
| Software validation | 43 local tests, Ruff, wheel/sdist build; GitHub CI succeeds on Python 3.10, 3.11 and 3.12 |

Machine-readable completion counts, all method outcomes and figures are generated
by `report_continuation.py`. Its normal mode refuses an incomplete PMOP campaign.
The `--allow-partial` option is only for development progress inspection.

## Reproduce

```bash
git submodule update --init --recursive
python -m pip install -e . -r experiments/requirements.txt
python -m experiments.runners.run_adversarial
python -m experiments.runners.validate_pmop_port --octave /path/to/octave
OPENBLAS_NUM_THREADS=1 python -m experiments.runners.run_snee_hessian
OPENBLAS_NUM_THREADS=1 python -m experiments.runners.run_paired_pmop
OPENBLAS_NUM_THREADS=1 python -m experiments.runners.run_ablation
OPENBLAS_NUM_THREADS=1 python -m experiments.runners.run_ammonia --all-windows
OPENBLAS_NUM_THREADS=1 python -m experiments.runners.run_pmop_direct
python -m experiments.runners.report_continuation
```

For a relocated Octave installation, set `OCTAVE_HOME` to its installation prefix.
Per-seed files are atomic checkpoints. Use a single campaign controller for each
output directory. Results include launch commits, source hashes, dependency
revisions and configs; `provenance/continuation_commit_map.json` maps local launch
commits to equivalent GitHub API publication commits.

## Interpretation

The GRV2 corrected Hessian passes independent finite-difference tests, while the
paired knee outputs remain effectively unchanged. The original implementation
is retained as a control. The paper reference remains
[arXiv:2501.16993v3](https://arxiv.org/html/2501.16993v3); no missing numerical
result table was invented from its figures.

The PMOP Python mappings now match the pinned MATLAB formula bodies numerically.
Source anomalies are retained and documented in `THIRD_PARTY.md`. In particular,
the PMOP7 zero-distance witness does not reproduce the supplied PF mapping.
The oracle-table track is a comparison on a common supplied finite table,
including official knee points; it is not an end-to-end Pareto-front discovery
test. Supported-knee labels are relative to that finite table.

`certified` in the existing core denotes its non-extreme and exit-trade-off
screens. It does not prove a global displacement bound. Weighted-sum tabular
validation uses the exact cell oracle, up to the declared LP tolerance. AWT
validation and continuous/MILP sampling are empirical lower bounds. The Track B
runner therefore records the original core screen separately and does not mark
heuristic lower-level outputs as globally certified.

The direct pilot retains DE iteration-limit outcomes and every returned feasible
point. Its budget is a solver diagnostic, not a global-optimality guarantee or a
fair-budget comparison with evolutionary methods.

Ammonia uses the original water and throughput-based safety objectives. With
fixed total ammonia production, they are affine functions of the same quantity.
The measured schedules verify
`water = 4.5 * 33.3 * safety + 4.5 * 3/17 * total_NH3`.
Strata describe temporal price-carbon correlation, not measured objective-space
correlation. The original five-candidate configuration returns a screened TAFR
selection for one of the twelve windows; all abstentions remain in the results.

The PMOP campaign contains both original LP and independently LP-validated
accelerated stability calculations. Existing checkpoints were preserved during
the restart. Accuracy comparisons can use the complete campaign; pooled timings
must not be presented as measurements of one uniform implementation. Statistical
blocks are the 42 problem/objective-count instances, not 1,260 independent copies
of deterministic equal-weight baselines.

## Remaining final-paper gates

1. Freeze the meaning of the final selection and certification criteria after
   evaluating the observed abstentions and missed perturbation outputs.
2. Establish adequate full-decision solver budgets and run the remaining PMOP
   objective counts/seeds with the specified evolutionary baselines.
3. Complete method-level budget scaling, reduction-search ablations and direct
   AWT comparison on larger MILPs under comparable lower-level budgets.
4. Establish a justified fourth independent process objective or explicitly
   evaluate the current ammonia case as an effectively reduced objective system;
   add the reference-front and reduction comparisons.

No absent method, timeout, or unfinished scientific gate is assigned a fabricated
result. The original uploaded papers and proposal/CV are never redistributed.

## Publication status

Code, tests, configurations, reporting scripts and measured artifacts are published
on `experiments/common-core-v1`. The owner explicitly authorized public release
after the initial automatic-review block. Raw results are preserved in grouped ZIP
archives; figures, smaller summaries and the historical Note PDF are directly
available. Run `python experiments/restore_artifacts.py` before rebuilding reports
from archived measurements. See [PUBLICATION_STATUS.md](PUBLICATION_STATUS.md)
for download paths, SHA-256 verification and the historical Note's scope.
