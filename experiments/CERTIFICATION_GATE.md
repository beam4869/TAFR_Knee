# A displacement limit changes the selection task

This experiment separates returning a knee candidate from verifying a displacement
limit. The production `certified` flag checks distance from objective anchors and
sampled exit trade-offs. Its selection rule minimizes sampled displacement among
passing candidates. It permits displacement above `objective_tolerance`, which
defines the stable-output region during the stability-radius calculation.

We compare four experimental rules on the earlier finite tables. The measurements
support explicit evidence labels and a separate user-specified displacement limit.
The production package remains byte-for-byte unchanged from `9f7a65d7`.

## Four rules isolate different checks

Let y(w) denote the nominal normalized objective output at preference weight w.
Let R denote the largest Euclidean distance from y(w) among outputs at simplex
weights within infinity-norm distance delta of w. We fix delta=0.05 and epsilon=0.001.
Here epsilon serves both as the stable-output tolerance and, for diagnosis, the
new acceptance limit. A sample maximum supplies a lower bound on R. The finite-table
oracle uses linear programs (LPs) to bound potentially reachable weighted-sum outputs.
The conservative output set contains every actual output, subject to the LP
feasibility tolerance. Let U be the largest displacement from y(w) among outputs
in this set. Ambiguous tie outputs can enlarge the set and increase U.

| Rule | Eligibility | Ranking |
| :-- | :-- | :-- |
| Original screens | Existing anchor and exit checks | Sampled R, then stability radius, exit ratio and weight |
| Sampled radius gate | Original screens and sampled R <= epsilon | Original ranking |
| Conservative radius gate | Original screens and U <= epsilon | Original ranking |
| Conservative radius + exits | Anchor check, U <= epsilon, and every first-shell exit passes | U, then conservative stability radius, minimum exit ratio and weight |

Rankings prefer smaller R or U, larger stability radius and larger exit ratio,
then lexicographically smaller weight. The last rule recomputes the stability
radius and exits independently of the sampled screens. Define rho as the supremum
radius at which the conservative output envelope stays within epsilon of y(w).
Its first shell contains every potentially reachable output beyond epsilon at
radius rho+0.01, capped by the simplex. If that shell has no exits, the rule tries up to four shells. An exit
passes when it has an improvement and deterioration above 0.001 in at least one
objective each, and its summed deterioration/improvement ratio reaches 1.
The implementation retains the core ratio safeguard of 1e-12.

The [configuration](configs/certification_gate.yaml) fixes 10 cases, 30 paired
seeds and 17 candidate weights. Each table uses its full objective ranges for
frozen normalization. The original-screen control reproduces all 300 earlier
`full` ablation selections, which use a minimum gain of 0.001. The package's
default minimum gain is 0.000001.

## A hard limit removes twelve selections

The campaign completes 5,100 candidate audits and 1,200 method records.
[Per-case measurements](results/certification_gate/mechanism_summary.csv) and
[method totals](results/certification_gate/method_totals.csv) retain all abstentions.

| Rule | Trials | Returns | Returned U > 0.001 | Returns passing the full table rule |
| :-- | --: | --: | --: | --: |
| Original screens | 300 | 234 | 12 | 217 |
| Sampled radius gate | 300 | 222 | 0 | 217 |
| Conservative radius gate | 300 | 222 | 0 | 217 |
| Conservative radius + exits | 300 | 221 | 0 | 221 |

The full table rule includes the radius condition and all-exit condition.
All twelve above-limit upper envelopes occur in the three enumerated portfolio
cases; the original sample maxima also exceed epsilon for these twelve returns.
The complete exit rule changes selections within the surviving trials and adds
one abstention, in portfolio1 seed 19. The linear and unsupported-front controls
remain abstentions. The central candidate in `flat_gain` still passes: its first
exit shell contains ordinary trade-offs. A tiny-gain alternative elsewhere in
the table does not by itself invalidate that candidate's local exit condition.

The [comparison figure](results/certification_gate/gate_comparison.svg) shows the
return-count cost of the added checks. Zero above-limit returns for the conservative
gate follow from its eligibility definition; they do not establish better knee
accuracy. These paired diagnostic runs reuse existing cases and seeds, so their
performance interpretation is exploratory.

## Targeted witnesses expose missed outputs

The [witness report generator](runners/report_certification_gate.py) evaluates
fixed preferences in the existing C, D and tiny-gain constructions. Their
purpose is to test the checks at known failure locations, separately from the
300-trial campaign.

| Witness | Weight radius | Limit | Sampled R | Table U | New rejection |
| :-- | --: | --: | --: | --: | :-- |
| C, pairwise perturbations | 0.1 | 0.2 | 0 | 0.208806 | Displacement limit |
| D, vertex perturbations | 0.18 | 0.28 | 0.178834 | 0.300000 | Displacement limit |
| Tiny-gain exit near a simplex edge | 0.000001 | 0.001 | 0 | 0 | Inactive first-shell exit |

The existing core screens accept all three preferences. The first two witnesses
show that a passing sample maximum can miss a violating output. The third shows
that discarding an inactive exit can change acceptance even when the fixed-radius
displacement vanishes. These examples support separate radius and exit checks;
they provide no frequency estimate for unseen problems.

## PMOP filtering loses most returns

We also filter the existing 1,260 trial records for PMOP1 through PMOP14,
with 3, 5 and 8 objectives and 30 seeds. PMOP denotes the pinned knee benchmark
suite in [THIRD_PARTY.md](THIRD_PARTY.md). This analysis only retains or discards
the previous selection; it performs no candidate reselection or solver rerun.
The common finite tables contain supplied reference knees, and their conclusions
apply to those tables. The original campaign returns 1,094 selections, including
564 successes within normalized Euclidean distance 0.025 of a reference knee.

| Displacement limit | Returns after historical LP filtering | Successes at knee distance 0.025 | Sample-filtered returns whose historical LP estimate exceeds the limit |
| :-- | --: | --: | --: |
| 0.001 | 162 | 109 | 0 |
| 0.010 | 202 | 149 | 0 |
| 0.025 | 264 | 196 | 0 |
| 0.050 | 290 | 207 | 3 |

The [PMOP breakdown](results/certification_gate/pmop_postfilter.csv) reports counts
by objective count and retains 1,260 as the denominator for each limit. At 0.05,
sample filtering returns 293 selections and historical LP filtering removes three more.
The loss of returns at 0.001 shows that copying the stability tolerance into a
hard acceptance limit changes the practical selection task. Repeating seeds
within 42 fixed instances does not create 1,260 independent benchmark problems;
we make no statistical-significance claim from these aggregate counts.

These PMOP counts reuse the earlier deterministic-cell LP estimates. The old
oracle drops cells whose strict margin reaches at most 1e-9, although selection
uses a 1e-12 tie tolerance. A targeted regression exposes a missed displacement
of 0.297321 despite an old reported value of zero. The new conservative oracle
retains that output and rejects verification. We reran the 300-trial gate campaign
with this correction. The PMOP postfilters remain historical diagnostics;
their estimates require conservative revalidation before supporting certificates.

## Evidence labels support two usage modes

The evidence interface returns `verified`, `refuted`, or `unverified`. An upper
bound at or below the requested limit verifies the radius condition. A lower bound
above the limit refutes it. Passing samples alone leave it unverified. The new
table wrapper aborts on unresolved LP statuses. It retains every feasible cell
whose row cost is within 1e-12 of all competitors, including small-margin and
tie-ambiguous cells. Directly checked feasible witnesses supply separate lower
bounds, so an excessive upper envelope alone does not establish a violation.

The table certificate uses LP feasibility tolerance 1e-9 and an enlarged cell
union that covers the selection tie tolerance of 1e-12. Its numerical scope
is the supplied normalized finite weighted-sum table. Continuous solvers, mixed
integer solvers and augmented Tchebycheff preference maps require their own upper
bounds. At a stability-radius boundary, a tie can already change the output;
the direct closed-ball audit therefore controls acceptance.

The next implementation should expose two modes. A ranking mode returns screened
candidates with their measured displacement and evidence status. An optional
constrained mode additionally requires a declared displacement limit tau.
The stable-output tolerance epsilon should remain a separate parameter.
The full-table exit rule remains an experimental option pending its scientific
definition and solver-budget evaluation. The present experiment deliberately
sets tau=epsilon to measure that special case; it does not select a final tau.

## Reproduction and remaining work

```bash
python experiments/restore_artifacts.py
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m experiments.runners.run_certification_gate
OPENBLAS_NUM_THREADS=1 python -m experiments.runners.report_certification_gate
python -m pytest
```

The separate raw archive also supports restoration with
`python experiments/restore_certification_artifacts.py` after placing
`raw_records.zip` beside `raw_manifest.json`. Automatic approval review blocked
public upload of the new raw archive and its checksum inventories pending
authorization for those specific records and metadata.
The repository's runner can regenerate them from the published configuration.

The runner checks source fingerprints, table hashes and configurations before
reusing atomic per-seed checkpoints. The report refuses incomplete campaigns
and checks the original-screen control against the earlier ablations. Raw records
include every candidate, table, exit witness, launch commit and measured shared
cost. Shared caches and the oracle's additional work prevent an equal-budget
runtime claim. The report generator writes `input_checksums.json` to identify
every source record used in the report; this inventory shares the pending
public-upload status of the raw archive.

Local verification passes 49 tests; three unchanged SNEE tests skip because this
checkout has no initialized SNEE submodule. Ruff and wheel/source builds pass.
Gate 1 now has a completed diagnostic comparison and an evidence interface.
Freezing the final selection definition, full-decision solver budgets, population
baselines and the remaining process-case comparisons remain open.
