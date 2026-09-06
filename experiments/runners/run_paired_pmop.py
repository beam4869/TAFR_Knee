"""Resume 42 PMOP oracle-table instances across 30 paired search seeds."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import hashlib
import json
from time import perf_counter

import numpy as np

from experiments.benchmarks.pmop_suite import SOURCE, load_instance
from experiments.common import RESULTS, config_file, provenance, save_json, write_bundle
from experiments.metrics.exact_tabular_oracle import ExactTabularOracle
from experiments.metrics.knee_matching import metrics
from experiments.runners.run_pilots import FastTable, evaluate


def require_parity():
    path = RESULTS / "raw/pmop_numeric_parity.json"
    if not path.exists():
        raise RuntimeError("Run validate_pmop_port with Octave before the paired campaign")
    gate = json.loads(path.read_text())
    checks = [r for r in gate["results"] if r["function"] != "PF_distance_witness"]
    if len(checks) != 126 or not all(r["passed"] for r in checks):
        raise RuntimeError("The complete PMOP numeric parity gate has not passed")
    for name, expected in gate["config"]["source_sha256"].items():
        if hashlib.sha256((SOURCE / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError("PMOP source changed since parity validation")


def instance_job(number, m, cfg):
    provenance_at_start = provenance()
    y, knees, region, metadata = load_instance(number, m, cfg["front_samples"], cfg["front_seed"])
    problem = FastTable(y)
    ideal, scale = y.min(axis=0), np.ptp(y, axis=0)
    normalized = (problem.y - ideal) / scale
    anchors = np.array([(problem.solve(w).objectives - ideal) / scale for w in np.eye(m)])
    anchor_affine_rank = int(np.linalg.matrix_rank(anchors[1:] - anchors[0]))
    oracle = ExactTabularOracle(normalized)
    targets = (knees - ideal) / scale
    support = oracle.supported(targets)
    metadata.update(python_port_validation="Octave parity verified for all 14 PF and CalObj functions",
                    finite_table_sha256=hashlib.sha256(y.tobytes()).hexdigest(),
                    supported_mask=support, normalization_ideal=ideal, normalization_scale=scale)
    rows = []
    validation_cache = {}
    for seed in cfg["seeds"]:
        path = RESULTS / "raw/pmop_paired" / f"PMOP{number}-{m}-{seed}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if saved["config"] != cfg:
                raise RuntimeError(f"Cached configuration differs: {path}")
            rows.extend(saved["rows"])
            continue
        start = perf_counter()
        try:
            rr, details = evaluate(problem, f"PMOP{number}", "PMOP", cfg,
                                   knees, region, seed=seed, table_exact=True)
            for row in rr:
                row.update(git_sha=provenance_at_start["git_sha"],
                           external_commit_shas=provenance_at_start["external_commit_shas"],
                           variant="30-seed oracle-table", instance=f"PMOP{number}-{m}",
                           all_knee_count=len(knees), supported_knee_count=int(support.sum()))
                for field, source in [("success_001", "success_0.01"),
                                      ("success_0025", "success_0.025"),
                                      ("success_005", "success_0.05")]:
                    row[field] = row.get(source, False)
                yn = row["normalized_objectives"]
                row["supported_metrics"] = (metrics([] if yn is None else [yn], targets[support])
                                             if support.any() else None)
                if row["selected_weight"] is not None:
                    key = tuple(row["selected_weight"])
                    if key not in validation_cache:
                        validation_cache[key] = (
                            oracle.audit(row["selected_weight"], cfg["radius"]),
                            oracle.stability_radius_fast(row["selected_weight"], cfg["objective_tolerance"]))
                    exact, exact_radius = validation_cache[key]
                    row["R_validation"] = exact["robustness"]
                    row["exact_worst_weight"] = exact["worst_weight"]
                    row["validation_type"] = "exact finite-table LP"
                    row["exact_stability_radius"] = exact_radius
                    row["exact_stability_algorithm"] = "bounded-simplex halfspaces, with cell-union LP fallback"
                    if row["R_reported"] is not None:
                        row["audit_gap"] = row["R_validation"] - row["R_reported"]
                        row["false_reported_bound"] = row["audit_gap"] > 1e-7
                if row["method"] == "CHIM-posthoc":
                    row["baseline_valid"] = (anchor_affine_rank == m - 1 and
                                             details["CHIM"]["chim_residual"] < 1e-8)
                    row["anchor_affine_rank"] = anchor_affine_rank
                    row["CHIM_metadata"] = details["CHIM"]
            out = dict(status="completed", rows=rr, details=details)
        except Exception as error:
            out = dict(status="failed", error=repr(error), rows=[])
        out.update(problem=f"PMOP{number}", m=m, seed=seed, config=cfg,
                   provenance=provenance_at_start, metadata=metadata,
                   total_wall_time_seconds=perf_counter() - start)
        save_json(path, out)
        rows.extend(out["rows"])
    return dict(problem=f"PMOP{number}", m=m, rows=rows)


def main():
    require_parity()
    cfg = config_file("pmop_paired")
    output = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        jobs = [pool.submit(instance_job, number, m, cfg)
                for m in cfg["objectives"] for number in cfg["problems"]]
        for future in as_completed(jobs):
            result = future.result()
            output.append(result)
            print(f"Paired PMOP {len(output)}/{len(jobs)}: {result['problem']}-{result['m']}", flush=True)
            write_bundle("pmop_paired", cfg, output)


if __name__ == "__main__":
    main()
