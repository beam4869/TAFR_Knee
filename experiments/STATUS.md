# Historical pilot status

For the completed later campaigns, read [CONTINUATION.md](CONTINUATION.md).
The latest criterion study and the numerical oracle correction are in
[CERTIFICATION_GATE.md](CERTIFICATION_GATE.md). The table below preserves the
earlier pilot's scope and outstanding work at that time.

This is a mechanism-validation and pilot release, not the final paper experiment matrix.
Core source under `src/tafrknee` remains unchanged from 9f7a65d7.

| Plan component | Executed evidence | Remaining gate |
|---|---|---|
| Exact oracle | LP cell reachability and stability, 2D/3D grid parity | Numerical LP tolerances remain explicit |
| E0 A-F | All six families, 100-seed interior audit, smooth witnesses | Full default still accepts 1e-5 improvement |
| SNEE | 54 versioned original-start jobs; 540 paired NM jobs with adapter v2 | Code/paper-v3 numerical table matching and Hessian correction variant |
| PMOP Track A | 14 problems x 3/5/8 objectives, same finite table, exact selected TAFR audit, every reference-knee support LP | Python PF port numerical parity; final 30 paired TAFR runs |
| PMOP Track B | Official MATLAB sources pinned | Full-decision solver and population baseline execution |
| Discrete | Three enumerated 15-binary portfolios, exact TAFR audit, AWT MC-RI diagnostic | Larger MILP and direct TAFR-AWT comparison |
| Scalability | Exact counts, enumeration through m=10, 30-seed Sobol coverage, grouped-objective counterexample | Full method accuracy/effort scaling and nine-ablation matrix |
| Ammonia | 12-window input manifest; 2 actual 48h direct-MILP windows, primary and global-bound variants | Objective independence, reference-front knees, remaining 10 windows |
| Figures/tables/note | Generated from committed measurements | Final journal claims require the gates above |

## Scientific issues that prevent freezing the final experiment matrix

1. A finite sampled optimizer-map audit is a lower bound on the true maximum.
   Complete weight-polytope vertices do not establish an objective-displacement bound.
2. The production `certified` field checks anchor distance and the first active exit
   ratio; it neither proves the global audit maximum nor enforces R <= epsilon.
   Processed `false_reported_bound` means validation exceeded the reported R by 1e-7.
3. Payoff normalization can collapse through alternate optima, or produce ranges
   at machine precision. Full-range oracle variants use additional information.
4. All sampled PMOP payoff anchor sets are affinely dependent; their minimum-norm
   CHIM scores remain raw diagnostics and are unavailable as a valid main baseline.
5. Equal normalized weights outperform TAFR on PMOP Success@0.025 in this pilot.
   The tables include supplied true knees, and PMOP defaults contain symmetries.
6. The source ammonia safety metric measures electrolyzer throughput. At fixed
   NH3 output, water and safety are affine functions of total electrolytic H2.
7. TPE-2024 is a grid/pairwise reconstruction of the proposal, not a KKT reproduction.
   MC-RI is post-analysis here; candidate ranking by RI and direct NBI remain pending.
8. Runtime/call budgets differ across method families. These pilots support diagnosis,
   not claims of equal-budget superiority. Null counters mean unmeasured.

No missing experiment has a fabricated result. Adapter-v1 SNEE outputs are retained
but superseded; use v2 for all main descriptions.
