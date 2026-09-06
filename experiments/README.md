# Common-core experiments

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
