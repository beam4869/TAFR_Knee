# Pinned sources

| Resource | Repository and revision | Use in this release |
|---|---|---|
| SNEE | tommaso-giovannelli/snee, c3464e6690a89a1198a7cc7e6dd13969ce102029 | Official objectives and search, explicit runtime adapters |
| PMOP | GYResearch/PMOPs-Benchmark, 4fb56cea31bae48f7ef08bc3180c3be150eb8f70 | Official MATLAB sources, true knees, knee-region references; attributed Python PF mapping |
| Ammonia | beam4869/ML_scheduling, 8a5e98691299c38d433a864eef9256ba0393429b | Existing model equations and price/carbon datasets |

All three are git submodules under `external/`, retaining the upstream files,
notices and licenses at the exact gitlinks. Do not infer a new license where an
upstream project has not supplied one. The PMOP distribution embeds PlatEMO;
its headers request research-use acknowledgement and the Tian et al. 2017
platform citation, included in the note. The uploaded PMOP PDF and proposal/CV
are reading sources and are not redistributed in this repository.

The closest preference-robustness reference is Mavrotas et al., EJOR 240(1),
193–201 (2015), DOI 10.1016/j.ejor.2014.06.039. The `mavrotas_ri.py` module is an
attributed experiment implementation, not a claimed official author release.

The SNEE paper version is arXiv:2501.16993v3 (19 March 2026). Code and paper
versions are recorded independently. The upstream GRV2 Hessian discrepancy and
the adapter-v1/v2 distinction are documented in README.md and diagnostics.json.

`experiments/benchmarks/pmop_decision.py` is a derived Python port of the pinned
PMOP objective functions and is explicitly marked EPL-2.0. The surrounding
experiment harness retains the repository's license. `validate_pmop_port.py`
executes temporary wrappers around the original MATLAB function bodies in
Octave, with shared inputs; it does not edit the submodule or redistribute
modified upstream source files.

The source-preserving port retains these upstream behaviors: PMOP1 sets `g=0`;
PMOP6 with linkage writes `temp2` while reading zero-initialized `temp`; PMOP7
uses a broadcast sum and a positive cosine product in its distance function;
PMOP10 computes a distance expression into `g` while leaving `g7` at zero.
Any mathematical correction requires a separately named benchmark variant.
