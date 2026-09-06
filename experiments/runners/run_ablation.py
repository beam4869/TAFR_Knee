"""Paired component ablations, direct AWT selection, and independent audits."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
import json
from time import perf_counter

import numpy as np

from tafrknee import KneeConfig
from tafrknee.audit import audit_prepared_weight
from tafrknee.normalization import FrozenNormalizer
from tafrknee.reduction import identity_reduction
from tafrknee.weights import hit_and_run_samples

from experiments.benchmarks.adversarial_tabular import extreme_plateau, incomplete_vertices, interior_cell
from experiments.benchmarks.discrete_suite import portfolio
from experiments.benchmarks.negative_controls import linear_front, unsupported_front
from experiments.common import RESULTS, config_file, provenance, raw_row, save_json, write_bundle
from experiments.metrics.exact_tabular_oracle import ExactTabularOracle
from experiments.methods.tafr_variants import awt_table
from experiments.runners.run_adversarial import ColdEngine
from experiments.runners.run_pilots import FastTable, candidates


def case(name):
    if name == "A": return extreme_plateau(), np.array([4., 4.])
    if name == "B": return extreme_plateau(10000), np.array([4., 40000.])
    if name == "C": return incomplete_vertices(), None
    if name == "D": return interior_cell(), None
    if name == "flat_gain":
        return np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0],
                         [.5, .5, .5], [.5 - 1e-5, .9, .5]]), None
    if name == "linear": return linear_front(), None
    if name == "unsupported": return unsupported_front()
    if name.startswith("portfolio"): return portfolio(int(name[-1]))[0], None
    raise ValueError(name)


def worker(name, seed, cfg):
    path = RESULTS / "raw/ablation_paired" / f"{name}-{seed}.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if saved["config"] != cfg: raise RuntimeError(f"Stale ablation config: {path}")
        return saved
    t = perf_counter()
    y, target = case(name)
    lo, scale = y.min(axis=0), np.ptp(y, axis=0)
    norm = (y - lo) / scale
    m = y.shape[1]
    weights = candidates(m, cfg["candidate_budget"], seed)
    kc = KneeConfig(radius=cfg["radius"], interior_samples=cfg["interior_samples"],
                    stability_iterations=cfg["stability_iterations"], random_seed=seed,
                    min_improvement=cfg["min_gain"], min_deterioration=cfg["min_gain"])
    groups = {}
    rows = []
    prov = provenance()
    exact_cache = {}
    for variant in cfg["variants"]:
        group = variant if variant in {"no_normalization", "pairwise", "vertices", "default_gain", "AWT", "adaptive"} else "full"
        if group not in groups:
            values = y if group == "no_normalization" else norm
            problem = awt_table(values) if group == "AWT" else FastTable(values)
            engine = ColdEngine(problem, FrozenNormalizer(np.zeros(m), np.ones(m)), identity_reduction(m))
            anchors = np.array([problem.solve(w).objectives for w in np.eye(m)])
            options = kc
            if group in ("pairwise", "vertices", "adaptive"):
                options = replace(options, audit_strategy=group)
            if group == "default_gain":
                options = replace(options, min_improvement=1e-6, min_deterioration=1e-6)
            start = perf_counter()
            audits = []
            failures = []
            for w in weights:
                try: audits.append(audit_prepared_weight(engine, w, anchors, options))
                except Exception as error: failures.append(dict(weight=w, error=repr(error)))
            groups[group] = dict(engine=engine, audits=audits, failures=failures,
                                 calls=engine.solver_calls + m, seconds=perf_counter() - start,
                                 values=values)
        data = groups[group]
        audits = data["audits"]
        if variant == "no_nonextreme":
            eligible = [a for a in audits if a.has_exit and a.tradeoff_active and a.exit_tradeoff >= kc.kappa_min]
        elif variant == "no_tradeoff":
            eligible = [a for a in audits if a.non_extreme]
        else:
            eligible = [a for a in audits if a.certified]
        if variant == "no_abstention" and not eligible: eligible = audits
        def ranking(a):
            ratio = a.exit_tradeoff if a.exit_tradeoff is not None else -np.inf
            return ((a.robustness, -ratio, tuple(a.weight)) if variant == "no_stability_tiebreak"
                    else (a.robustness, -a.stability_radius, -ratio, tuple(a.weight)))
        chosen = min(eligible, key=ranking) if eligible else None
        row = raw_row(problem_family="mechanism_ablation", problem_name=name, instance=name,
                      method="TAFR", variant=variant, seed=seed, n_objectives=m,
                      git_sha=prov["git_sha"], external_commit_shas=prov["external_commit_shas"],
                      scalarization="AWT" if group == "AWT" else "weighted_sum",
                      candidate_budget=len(weights), radius=kc.radius, objective_tolerance=kc.objective_tolerance,
                      extreme_threshold=kc.extreme_threshold, kappa_min=kc.kappa_min,
                      min_improvement=1e-6 if group == "default_gain" else kc.min_improvement,
                      min_deterioration=1e-6 if group == "default_gain" else kc.min_deterioration,
                      selected=chosen is not None, abstained=chosen is None,
                      certified=bool(chosen and chosen.certified),
                      solver_calls=data["calls"], unique_solver_calls=data["calls"],
                      lower_level_failures=len(data["failures"]), wall_time_seconds=data["seconds"],
                      audit_group=group, shared_audit_cost=True,
                      displacement_units="raw" if group == "no_normalization" else "normalized")
        if chosen is not None:
            w = chosen.weight
            v = chosen.normalized_objectives
            raw = v if group == "no_normalization" else v * scale + lo
            normalized = (raw - lo) / scale
            count = max(cfg["validation_samples"], 10 * chosen.perturbation_count)
            u = hit_and_run_samples(w, kc.radius, count, seed=seed + 700001, burn_in=64)
            vv = np.array([data["engine"].solve(a).objectives for a in u])
            d = np.linalg.norm(vv - v, axis=1)
            validation, stability = float(d.max()), None
            validation_type = "independent samples; lower bound"
            if group != "AWT":
                cache_key = (group == "no_normalization", tuple(w))
                if cache_key not in exact_cache:
                    oracle = ExactTabularOracle(data["values"])
                    exact_cache[cache_key] = (oracle.audit(w, kc.radius)["robustness"],
                                              oracle.stability_radius(w, kc.objective_tolerance))
                validation, stability = exact_cache[cache_key]
                validation_type = "exact finite-table LP"
            error = None if target is None else float(np.linalg.norm(normalized - (target - lo) / scale))
            row.update(selected_weight=w, raw_objectives=raw, normalized_objectives=normalized,
                       R_reported=chosen.robustness, R_validation=validation,
                       audit_gap=validation - chosen.robustness,
                       false_reported_bound=validation > chosen.robustness + 1e-7,
                       stability_radius=chosen.stability_radius, exact_stability_radius=stability,
                       exit_tradeoff=chosen.exit_tradeoff, core_screen_reasons=chosen.reasons,
                       expected_displacement=float(d.mean()), persistence_probability=float(np.mean(d <= kc.objective_tolerance)),
                       validation_type=validation_type, validation_samples=count, knee_error=error,
                       success_001=error is not None and error <= .01,
                       success_0025=error is not None and error <= .025,
                       success_005=error is not None and error <= .05)
        rows.append(row)
    out = dict(problem=name, seed=seed, config=cfg, provenance=prov, rows=rows,
               total_wall_time_seconds=perf_counter() - t,
               candidate_failures={name: data["failures"] for name, data in groups.items()})
    save_json(path, out)
    return out


def main():
    cfg = config_file("ablation_paired")
    rows = []
    with ProcessPoolExecutor(max_workers=cfg["workers"]) as pool:
        futures = [pool.submit(worker, name, seed, cfg) for name in cfg["cases"] for seed in cfg["seeds"]]
        for i, future in enumerate(as_completed(futures), 1):
            rows.extend(future.result()["rows"])
            if i % 10 == 0:
                print(f"Paired ablations: {i}/{len(futures)}", flush=True)
                write_bundle("ablation_paired", cfg, rows)
    write_bundle("ablation_paired", cfg, rows)


if __name__ == "__main__":
    main()
