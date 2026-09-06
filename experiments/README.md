> Results, figures and the historical Note PDF are published. See [PUBLICATION_STATUS.md](PUBLICATION_STATUS.md) for downloads and the verified raw-data restore command.

# Common-core experiments

The 6 September continuation is documented in [CONTINUATION.md](CONTINUATION.md),
including the paired campaigns, independent Octave checks, result interpretation,
and remaining final-paper gates. Public artifact availability is described in
[PUBLICATION_STATUS.md](PUBLICATION_STATUS.md).

This branch implements the user-supplied six-phase plan, starting with exact
mechanism checks. The base package is commit
`9f7a65d7ffeaddedf2969b7db3d764623e0cca27` (v0.1.0). Experimental variants are
named explicitly; upstream code and package defaults are not silently changed.

```bash
git submodule update --init --recursive
python -m pip install -e '.[dev]' pandas pyyaml scikit-learn
python -m experiments.runners.run_adversarial
```

Every bundle records code/dependency commits, configuration, environment, and seeds.
Raw values are the source of processed tables and figures. Null fields denote
unmeasured/not applicable quantities. An LP oracle is exact for the finite supplied
table up to solver tolerances; a sampled audit is an empirical lower bound on
worst-case displacement. It does not provide a global certificate.

External code lives in pinned submodules under `external/`. The SNEE README states
GNU LGPL availability; its repository does not contain a separate license file or
specify an LGPL version. PMOPs includes its original EPL-2.0 license and source
notices. Those resources retain their own terms, separate from this package's MIT
license. Uploaded papers and personal CV materials are not redistributed here.

### PMOP normalization diagnosis
The first 42-instance run records 33 frozen payoff failures: deterministic
lexicographic single-objective ties omit some objective ranges. This is retained
as `pmop_pilot.json`. The separate `--table-range` oracle-front variant supplies
finite-table minima/maxima through the existing API. It uses PF information and
must not be described as the production anchor-only normalizer or direct search.

### Ammonia interpretation
The primary pilot uses the original payoff normalization and five candidates.
`--global-range` is a separately named diagnostic with four anti-anchor solves,
17 candidates and six radius bisections. It changes bounds AND search budget;
its outcome does not isolate the causal effect of normalization alone. The
source safety objective is H2 electrolyzer throughput divided by 33.3. It does
not include source-declared but unused startup, inventory or ramp-risk terms.
When total NH3 is fixed, water and safety are affinely dependent. Window strata
use temporal price-carbon Pearson correlation as an explicit proxy, not the
cost-emission correlation of feasible schedules. All forecasts stay fixed.

### SNEE compatibility adapter v2
The first compatibility adapter flattened the gradient globally. Independent
inspection found that this leaked into the upstream KKT multiplier routine,
which needs a column gradient. All first-batch compatibility outcomes are
retained as superseded adapter-v1 diagnostics. Adapter v2 flattens only when
SLSQP supplies a one-dimensional decision vector; column inputs keep the original
shape. Main comparisons use `snee_reproduction_v2` and `snee_multistart_v2`.
The GRV2 Hessian column-vector broadcasting discrepancy is upstream and remains
unchanged in both versions; `diagnostics.json` contains a finite-difference check.

### Reproduce the release

```bash
git submodule update --init --recursive
python -m pip install -e . -r experiments/requirements.txt
PYTHONPATH=src python -m experiments.runners.run_experiment smoke
PYTHONPATH=src python -m experiments.runners.run_experiment adversarial
PYTHONPATH=src python -m experiments.runners.run_experiment snee
PYTHONPATH=src python -m experiments.runners.run_experiment pmop
PYTHONPATH=src python -m experiments.runners.run_experiment scalability
PYTHONPATH=src python -m experiments.runners.run_experiment ammonia
PYTHONPATH=src python -m experiments.runners.extend_validation
PYTHONPATH=src python -m experiments.runners.run_experiment report
(cd experiments/note && latexmk -pdf TAFR_Experiment_Note.tex)
```

Start with `STATUS.md` and the note PDF. `processed/standardized_runs.jsonl`
provides a common schema for the main experiment records; the nested raw bundles
retain full histories, configs, failures, and diagnostics. Early serialization
checkpoints have explicit launch-revision corrections in
`PROVENANCE_CORRECTIONS.json`; later runs freeze source hashes at import.

The final validation overlay contains 84 exact PMOP baseline audits, six exact
portfolio baseline audits, and nine continuous sample-budget extensions.
Continuous validation uses at least ten times the total training audit-point
count. Expected displacement and persistence still describe samples even when
worst-case displacement comes from an exact finite-table oracle.

SVG originals accompany PDF/PNG figures. Cairo converts the SVG to PDF to avoid
truncated Matplotlib PDF streams observed in this runtime. All figures consume
processed data; generated tables and note macros have a claim ledger.
