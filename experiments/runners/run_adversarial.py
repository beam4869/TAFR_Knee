"""Execute Phase 0/1 exactly as specified, including adverse default results."""
from __future__ import annotations

from dataclasses import asdict, replace
from time import perf_counter

import numpy as np
from scipy.stats import norm
from tafrknee import KneeConfig, audit_weight, select_knee
from tafrknee._engine import EvaluationEngine
from tafrknee.audit import _radius_robustness
from tafrknee.metrics import tradeoff
from tafrknee.normalization import FrozenNormalizer
from tafrknee.reduction import identity_reduction
from tafrknee.weights import perturbation_weights

from experiments.benchmarks.adversarial_tabular import as_problem, extreme_plateau, incomplete_vertices, interior_cell
from experiments.benchmarks.adversarial_smooth import smooth_companion
from experiments.benchmarks.negative_controls import linear_front, unsupported_front
from experiments.common import config_file, write_bundle
from experiments.metrics.exact_tabular_oracle import ExactTabularOracle
from experiments.methods.tafr_variants import awt_table


class ColdEngine(EvaluationEngine):
    """Analytic/tabular solvers ignore warm starts; avoid quadratic cache scans."""
    def solve(self, weight, *, allow_warm_start=True):
        return super().solve(weight, allow_warm_start=False)


def engine(problem, scale):
    return ColdEngine(problem, FrozenNormalizer(np.zeros(problem.n_objectives), np.asarray(scale)),
                      identity_reduction(problem.n_objectives))


def wilson(k, n):
    z = norm.ppf(.975)
    p = k / n
    center = (p + z*z/(2*n)) / (1+z*z/n)
    delta = z*np.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [float(center-delta), float(center+delta)]


def main():
    cfg = config_file("red_team")
    base = KneeConfig(radius=.05, objective_tolerance=1e-8, interior_samples=0,
                      audit_strategy="vertices", stability_tolerance=1e-7,
                      stability_iterations=26)
    out = {}
    # A: use actual public API for the production and ablated configurations.
    y = extreme_plateau()
    p = as_problem(y)
    full = select_knee(p, config=base)
    no_ext = select_knee(p, config=replace(base, extreme_threshold=0))
    legacy_weights = np.column_stack([np.linspace(.05,.95,21), 1-np.linspace(.05,.95,21)])
    risks = []
    for w in legacy_weights:
        idx = int(p.solve(w).decision)
        pert = perturbation_weights(w, .05, strategy="pairwise")
        alt = np.array([p.solve(u).objectives for u in pert])
        risks.append(float(np.max(np.sum((alt-y[idx])**2, axis=1))))
    oracle = ExactTabularOracle(y/10)
    out["E0-A"] = dict(full=full.to_dict(), no_nonextreme=no_ext.to_dict(),
        legacy_selected=int(p.solve(legacy_weights[int(np.argmin(risks))]).decision),
        legacy_risks=risks, exact_stability=oracle.stability_radius([.5,.5]),
        exit=asdict(tradeoff(y[1]/10,y[0]/10)))
    assert full.selected.decision == 1
    assert np.isclose(out["E0-A"]["exact_stability"], .1, atol=1e-9)
    assert np.isclose(out["E0-A"]["exit"]["ratio"], 1.5, atol=1e-9)
    print("E0-A passed", flush=True)

    out["E0-B"] = []
    for c in cfg["scales"]:
        scaled = as_problem(extreme_plateau(c))
        f = select_knee(scaled, config=base)
        raw = select_knee(scaled, config=base, ideal=[0,0], reference=[1,1])
        record = dict(scale=c, normalized=f.to_dict(include_audits=False),
                      no_normalization=raw.to_dict(include_audits=False),
                      raw_knee_interval=[4*c/(6+4*c), 6*c/(4+6*c)])
        assert np.linalg.norm(f.selected.normalized_objectives-[.4,.4]) < 1e-8
        out["E0-B"].append(record)
    print("E0-B passed", flush=True)

    y = incomplete_vertices()
    w = np.full(4,.25)
    comparisons = {}
    for strategy in ["pairwise","vertices"]:
        points = perturbation_weights(w,.1,strategy=strategy)
        indices = np.argmin(points @ y.T, axis=1)
        comparisons[strategy] = dict(weights=points, selected=indices,
            R=float(np.max(np.linalg.norm(y[indices]/10-y[4]/10,axis=1))))
    exact = ExactTabularOracle(y/10).audit(w,.1)
    out["E0-C"] = dict(**comparisons, exact=exact)
    assert comparisons["pairwise"]["R"] == 0
    assert np.isclose(comparisons["vertices"]["R"], np.sqrt(.0436))
    assert np.isclose(exact["robustness"], comparisons["vertices"]["R"])
    print("E0-C passed", flush=True)

    y = interior_cell()
    w = np.full(3,1/3)
    exact = ExactTabularOracle(y/10).audit(w,.18)
    points = perturbation_weights(w,.18,strategy="vertices")
    idx = np.argmin(points @ y.T,axis=1)
    vertex_r = float(np.max(np.linalg.norm(y[idx]/10-y[3]/10,axis=1)))
    assert np.isclose(exact["robustness"], .3)
    assert np.argmin(y @ [.48,.26,.26]) == 4
    assert np.isclose(vertex_r,.178834,atol=1e-5)
    records, summaries = [], []
    designs = [(n,0) for n in cfg["interior_samples"]]
    designs += [(16,r) for r in cfg["adaptive_rounds"] if r]
    for n, rounds in designs:
        selected_records=[]
        for seed in range(cfg["sampling_seeds"]):
            ec = replace(base, radius=.18, audit_strategy="adaptive" if rounds else "hybrid",
                         interior_samples=n, random_seed=seed, adaptive_rounds=rounds)
            e = engine(as_problem(y),[10]*3)
            t = perf_counter()
            R, calls = _radius_robustness(e,w,y[3]/10,.18,ec,seed_offset=0)
            row = dict(samples=n, adaptive_rounds=rounds, seed=seed, R_reported=R,
                       R_validation=.3, audit_gap=.3-R, detected=bool(abs(R-.3)<1e-9),
                       solver_calls=e.solver_calls, wall_time_seconds=perf_counter()-t)
            selected_records.append(row)
            records.append(row)
        hits = sum(r["detected"] for r in selected_records)
        summaries.append(dict(samples=n, adaptive_rounds=rounds, detection_probability=hits/cfg["sampling_seeds"],
                              wilson95=wilson(hits,cfg["sampling_seeds"]),
                              mean_gap=np.mean([r["audit_gap"] for r in selected_records]),
                              mean_calls=np.mean([r["solver_calls"] for r in selected_records])))
        print(f"E0-D n={n}, rounds={rounds}: {hits}/{cfg['sampling_seeds']}",flush=True)
    smooth=[]
    for tau in cfg["smooth_tau"]:
        q = smooth_companion(y,tau)
        nominal=q.solve(w).objectives
        rv=max(np.linalg.norm((q.solve(u).objectives-nominal)/10) for u in points)
        witness=np.linalg.norm((q.solve(np.array([.48,.26,.26])).objectives-nominal)/10)
        smooth.append(dict(tau=tau, vertex_R=rv, interior_witness_R=witness,
                           label="witness lower bound; not global QP worst-case certificate"))
        assert witness>rv+.1
    out["E0-D"] = dict(exact=exact, vertex_R=vertex_r, records=records, summary=summaries, smooth=smooth)

    out["E0-E"]=[]
    for eps in cfg["flat_gains"]:
        nominal=np.array([.5,.5,.5]); alternative=np.array([.5-eps,.9,.5])
        current=tradeoff(nominal,alternative)
        for threshold in [1e-6]+cfg["minimum_gains"]:
            result=tradeoff(nominal,alternative,min_improvement=threshold,min_deterioration=threshold)
            out["E0-E"].append(dict(epsilon=eps,threshold=threshold,**asdict(result)))
        assert np.isclose(current.improvement,eps,rtol=1e-8)
        assert np.isclose(current.deterioration,.4)
    print("E0-E measured: current default accepts flat gains above 1e-6",flush=True)

    front,k=unsupported_front()
    weights=np.column_stack([np.linspace(0,1,1001),1-np.linspace(0,1,1001)])
    ws=as_problem(front)
    awt=awt_table(front)
    ws_idx=[int(ws.solve(w).decision) for w in weights]
    awt_idx=[int(awt.solve(w).decision) for w in weights]
    no_knee=select_knee(as_problem(linear_front()),config=base)
    negative=dict(linear=no_knee.to_dict(include_audits=False),
                  unsupported_knee=k, ws_nearest=float(np.min(np.linalg.norm(front[ws_idx]-k,axis=1))),
                  awt_nearest=float(np.min(np.linalg.norm(front[awt_idx]-k,axis=1))),
                  awt_recovers=bool(np.min(np.linalg.norm(front[awt_idx]-k,axis=1))<.003),
                  interpretation="AWT reachability is distinct from passing a global exit-ratio threshold")
    assert negative["ws_nearest"]>.4 and negative["awt_recovers"]
    # No-bulge result is measured, not forced; kappa=1 need not imply curvature.
    out["E0-F"]=negative
    # Row permutation invariance uses a fixed objective-lexicographic secondary rule.
    perms=[]
    rng=np.random.default_rng(40)
    for _ in range(20):
        order=rng.permutation(3)
        sol=select_knee(as_problem(extreme_plateau()[order]),config=base)
        perms.append(sol.selected.normalized_objectives)
    assert np.max(np.abs(np.asarray(perms)-[.4,.4]))<1e-10
    out["acceptance"]=dict(exact_checks_passed=True, permutations_tested=20,
        defaults_frozen=True, flat_gain_default_limitation=True,
        linear_abstained=not no_knee.found,
        note="Known limitations are recorded, not silently converted into acceptance targets.")
    path=write_bundle("adversarial_phase1",cfg,out)
    print(path,flush=True)


if __name__=="__main__":
    main()
