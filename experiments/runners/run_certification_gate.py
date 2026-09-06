"""Paired finite-table gate experiment with atomic, source-checked checkpoints."""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from experiments.common import ROOT, config_file, provenance, save_json
from experiments.methods.table_certificate import CheckedTableOracle, certify_table_weight
from experiments.runners.run_ablation import case
from experiments.runners.run_adversarial import ColdEngine
from experiments.runners.run_pilots import FastTable, candidates
from tafrknee import KneeConfig
from tafrknee.audit import audit_prepared_weight
from tafrknee.normalization import FrozenNormalizer
from tafrknee.reduction import identity_reduction

VARIANTS = ("core_screen", "sampled_radius_gate", "exact_radius_gate", "exact_all_exits")


def source_fingerprint():
    """Ignore report-only edits but bind every executable experiment dependency."""
    digest = hashlib.sha256()
    for folder in ("src", "experiments/methods", "experiments/metrics",
                   "experiments/benchmarks", "experiments/runners"):
        for path in sorted((ROOT / folder).rglob("*.py")):
            if path.name.startswith("report_"):
                continue
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    digest.update((ROOT / "experiments/common.py").read_bytes())
    return digest.hexdigest()


def old_rank(audit):
    return (audit.robustness, -audit.stability_radius, -audit.exit_tradeoff, tuple(audit.weight))


def run_case(name, cfg, output):
    y, target = case(name)
    lo, scale = y.min(axis=0), np.ptp(y, axis=0)
    if np.any(scale <= 0):
        raise ValueError(f"Collapsed normalization: {name}")
    normalized = (y - lo) / scale
    problem = FastTable(normalized)
    oracle = CheckedTableOracle(problem.y)
    anchors = np.array([problem.solve(w).objectives for w in np.eye(y.shape[1])])
    table_hash = hashlib.sha256(problem.y.tobytes()).hexdigest()
    source_hash = source_fingerprint()
    prov = provenance()
    exact_cache = {}
    for seed in cfg["seeds"]:
        path = Path(output) / "raw" / f"{name}-{seed}.json"
        if path.exists():
            saved = json.loads(path.read_text())
            if (saved["config"] != cfg or saved["table_sha256"] != table_hash
                    or saved["source_fingerprint"] != source_hash):
                raise RuntimeError(f"Stale checkpoint: {path}")
            continue
        start = perf_counter()
        engine = ColdEngine(problem, FrozenNormalizer(np.zeros(oracle.m), np.ones(oracle.m)),
                            identity_reduction(oracle.m))
        config = KneeConfig(radius=cfg["radius"], objective_tolerance=cfg["objective_tolerance"],
                            min_improvement=cfg["min_gain"], min_deterioration=cfg["min_gain"],
                            interior_samples=cfg["interior_samples"], random_seed=seed,
                            stability_iterations=cfg["stability_iterations"],
                            exit_step=cfg["exit_step"], exit_layers=cfg["exit_layers"])
        audits, exact, records = [], [], []
        lp_start = oracle.lp_calls
        for w in candidates(oracle.m, cfg["candidate_budget"], seed):
            audit = audit_prepared_weight(engine, w, anchors, config)
            key = tuple(w)
            if key not in exact_cache:
                exact_cache[key] = certify_table_weight(oracle, w, config)
            certificate = exact_cache[key]
            audits.append(audit)
            exact.append(certificate)
            records.append(dict(sampled_audit=audit.to_dict(), table_evidence=certificate))
        rows = []
        for variant in VARIANTS:
            eligible = []
            for index, (a, e) in enumerate(zip(audits, exact, strict=True)):
                keep = a.certified
                if variant == "sampled_radius_gate":
                    keep = keep and a.robustness <= config.objective_tolerance
                elif variant == "exact_radius_gate":
                    keep = keep and e["radius_status"] == "verified"
                elif variant == "exact_all_exits":
                    keep = e["accepted"]
                if keep:
                    eligible.append(index)
            def ranking(i, exact=exact, variant=variant, audits=audits):
                e = exact[i]
                if variant == "exact_all_exits":
                    return (e["R_exact"], -e["exact_stability_radius"],
                            -e["exit_tradeoff"], tuple(audits[i].weight))
                return old_rank(audits[i])
            selected = min(eligible, key=ranking) if eligible else None
            row = dict(case=name, seed=seed, variant=variant, selected=selected is not None,
                       eligible_candidates=len(eligible), selected_candidate=selected,
                       R_reported=None, R_exact=None, selected_above_epsilon=False,
                       radius_status="unverified", exit_certificate_passed=False,
                       selected_objectives=None, selected_weight=None, knee_error=None)
            if selected is not None:
                a, e = audits[selected], exact[selected]
                value = e["R_exact"]
                row.update(R_reported=a.robustness, R_exact=value,
                           selected_above_epsilon=value > config.objective_tolerance,
                           radius_status=e["radius_status"],
                           exit_certificate_passed=e["accepted"],
                           selected_objectives=a.normalized_objectives,
                           selected_weight=a.weight,
                           knee_error=None if target is None else
                           float(np.linalg.norm(a.normalized_objectives - (target - lo) / scale)))
            rows.append(row)
        # No swallowed candidate exceptions or synthetic rows: an incomplete job
        # has no checkpoint and can be restarted without replacing measurements.
        save_json(path, dict(case=name, seed=seed, config=cfg, knee_config=asdict(config),
                             table_sha256=table_hash, source_fingerprint=source_hash,
                             provenance=prov, normalization_ideal=lo, normalization_scale=scale,
                             normalized_table=problem.y, candidates=records, rows=rows,
                             sampled_solver_calls=engine.solver_calls + oracle.m,
                             oracle_lp_calls=oracle.lp_calls - lp_start,
                             seconds=perf_counter() - start,
                             cost_scope="shared across variants; exact cache shared within case"))
    return name


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/results/certification_gate")
    args = parser.parse_args()
    cfg = config_file("certification_gate")
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(run_case, name, cfg, args.output) for name in cfg["cases"]]
        for future in as_completed(futures):
            print(f"Completed gate case: {future.result()}", flush=True)


if __name__ == "__main__":
    main()
