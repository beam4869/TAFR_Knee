"""Report completed paired gates and explicitly post-hoc PMOP filtering."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.benchmarks.adversarial_tabular import incomplete_vertices, interior_cell
from experiments.common import ROOT, config_file, save_json
from experiments.methods.table_certificate import CheckedTableOracle, certify_table_weight
from experiments.runners.run_adversarial import ColdEngine
from experiments.runners.run_certification_gate import VARIANTS, source_fingerprint
from experiments.runners.run_pilots import FastTable
from tafrknee import KneeConfig
from tafrknee.audit import audit_prepared_weight
from tafrknee.normalization import FrozenNormalizer
from tafrknee.reduction import identity_reduction

LABELS = {"core_screen": "Original screens", "sampled_radius_gate": "Sampled radius gate",
          "conservative_radius_gate": "Conservative radius gate",
          "conservative_all_exits": "Conservative radius + exits"}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def mechanism_rows(output, cfg):
    rows, inputs, counts = [], {}, {}
    expected = {f"{name}-{seed}.json" for name in cfg["cases"] for seed in cfg["seeds"]}
    observed = {p.name for p in (output / "raw").glob("*.json")}
    if observed != expected:
        raise RuntimeError(f"Incomplete/unexpected checkpoints: {len(observed)}/{len(expected)}")
    baseline_mismatches = []
    for filename in sorted(expected):
        path = output / "raw" / filename
        value = json.loads(path.read_text())
        if value["config"] != cfg or value["source_fingerprint"] != source_fingerprint():
            raise RuntimeError(f"Stale source or configuration: {filename}")
        if len(value["candidates"]) != cfg["candidate_budget"]:
            raise RuntimeError(f"Missing candidates: {filename}")
        if tuple(r["variant"] for r in value["rows"]) != VARIANTS:
            raise RuntimeError(f"Missing methods: {filename}")
        if value["provenance"]["dirty"]:
            raise RuntimeError(f"Uncommitted execution source: {filename}")
        inputs[str(path.relative_to(ROOT))] = digest(path)
        rows.extend(value["rows"])
        for record in value["candidates"]:
            reasons = record["table_evidence"]["reasons"]
            for reason in reasons:
                counts[reason] = counts.get(reason, 0) + 1
        historical = ROOT / "experiments/results/raw/ablation_paired" / filename
        inputs[str(historical.relative_to(ROOT))] = digest(historical)
        old = next(r for r in json.loads(historical.read_text())["rows"] if r["variant"] == "full")
        new = value["rows"][0]
        if old["selected"] != new["selected"] or (new["selected"] and not np.allclose(
                old["selected_weight"], new["selected_weight"], atol=1e-12, rtol=0)):
            baseline_mismatches.append(filename)
    if baseline_mismatches:
        raise RuntimeError(f"Original paired control changed: {baseline_mismatches}")
    return pd.DataFrame(rows), inputs, counts


def pmop_postfilters(cfg):
    records, inputs = [], {}
    for m in (3, 5, 8):
        for number in range(1, 15):
            for seed in range(30):
                path = (ROOT / "experiments/results/raw/pmop_paired"
                        / f"PMOP{number}-{m}-{seed}.json")
                value = json.loads(path.read_text())
                if value["status"] != "completed":
                    raise RuntimeError(f"Unfinished PMOP input: {path}")
                row = next(r for r in value["rows"] if r["method"] == "TAFR-meaningful-gain")
                if row["selected"] and row["validation_type"] != "exact finite-table LP":
                    raise RuntimeError(f"Missing exact selected-output evidence: {path}")
                inputs[str(path.relative_to(ROOT))] = digest(path)
                for epsilon in cfg["pmop_postfilter_tolerances"]:
                    for gate in ("original", "sampled_postfilter", "historical_lp_postfilter"):
                        keep = row["selected"]
                        if gate != "original" and keep:
                            key = "R_reported" if gate == "sampled_postfilter" else "R_validation"
                            keep = row[key] <= epsilon
                        records.append(dict(problem=f"PMOP{number}", m=m, seed=seed,
                                            gate=gate, epsilon=epsilon, selected=keep,
                                            above_epsilon=keep and row["R_validation"] > epsilon,
                                            success_0025=keep and row["success_0025"],
                                            missed_bound=keep and row["false_reported_bound"]))
    frame = pd.DataFrame(records)
    summary = frame.groupby(["m", "epsilon", "gate"], sort=True).agg(
        trials=("selected", "size"), returns=("selected", "sum"),
        historical_lp_above_epsilon=("above_epsilon", "sum"),
        successes_0025=("success_0025", "sum"),
        reported_understatements=("missed_bound", "sum")).reset_index()
    return summary, inputs


def counterexamples():
    profiles = [
        ("C_missing_vertex", incomplete_vertices() / 10, np.full(4, .25),
         KneeConfig(radius=.1, objective_tolerance=.2, audit_strategy="pairwise")),
        ("D_interior_output", interior_cell() / 10, np.full(3, 1 / 3),
         KneeConfig(radius=.18, objective_tolerance=.28, audit_strategy="vertices")),
        ("E_inactive_exit", np.array([[0, 1, 1], [1, 0, 1], [1, 1, 0],
                                     [.5, .5, .5], [.5 - 1e-5, .9, .5]]),
         np.array([.499975, .00005, .499975]),
         KneeConfig(radius=1e-6, min_improvement=.001, min_deterioration=.001)),
    ]
    records = []
    for name, y, w, config in profiles:
        p = FastTable(y)
        engine = ColdEngine(p, FrozenNormalizer(np.zeros(len(w)), np.ones(len(w))),
                            identity_reduction(len(w)))
        anchors = np.array([p.solve(u).objectives for u in np.eye(len(w))])
        old = audit_prepared_weight(engine, w, anchors, config)
        exact = certify_table_weight(CheckedTableOracle(y), w, config)
        records.append(dict(name=name, radius=config.radius, epsilon=config.objective_tolerance,
                            old=old.to_dict(), exact=exact))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=ROOT / "experiments/results/certification_gate")
    args = parser.parse_args()
    cfg = config_file("certification_gate")
    frame, inputs, reasons = mechanism_rows(args.output, cfg)
    pmop, pmop_inputs = pmop_postfilters(cfg)
    inputs.update(pmop_inputs)
    summary = frame.groupby(["case", "variant"], sort=True).agg(
        trials=("selected", "size"), returns=("selected", "sum"),
        upper_bound_above_epsilon=("upper_bound_above_epsilon", "sum"),
        witness_above_epsilon=("witness_above_epsilon", "sum"),
        full_table_rule_passes=("full_table_rule_passed", "sum")).reset_index()
    totals = summary.groupby("variant").sum(numeric_only=True).reindex(VARIANTS).reset_index()
    frame.to_csv(args.output / "selections.csv", index=False)
    summary.to_csv(args.output / "mechanism_summary.csv", index=False)
    totals.to_csv(args.output / "method_totals.csv", index=False)
    pmop.to_csv(args.output / "pmop_postfilter.csv", index=False)
    save_json(args.output / "counterexamples.json", counterexamples())
    save_json(args.output / "input_checksums.json", inputs)
    completion = dict(expected_seed_jobs=len(cfg["cases"]) * len(cfg["seeds"]),
                      observed_seed_jobs=frame.shape[0] // len(VARIANTS),
                      candidate_audits=frame.shape[0] // len(VARIANTS) * cfg["candidate_budget"],
                      selection_records=len(frame), original_control_mismatches=0,
                      pmop_source_seed_jobs=1260, pmop_mode="postfilter only; no reselection",
                      source_fingerprint=source_fingerprint(), rejection_counts=reasons,
                      partial=False, production_core_modified=False)
    save_json(args.output / "completion.json", completion)
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})
    fig, ax = plt.subplots(1, 2, figsize=(10.8, 3.8), layout="constrained")
    labels = [LABELS[v] for v in VARIANTS]
    colors = ["#8795a4", "#487fb3", "#237964", "#be773b"]
    for a, field, title in zip(ax, ("returns", "upper_bound_above_epsilon"),
                              ("Returned selections / 300 trials",
                               "Returned upper bounds > 0.001"),
                              strict=True):
        bars = a.barh(labels, totals[field], color=colors)
        a.bar_label(bars, padding=4)
        a.set_title(title, loc="left", fontsize=11)
        a.set_xlim(0, 300 if field == "returns" else max(8, totals[field].max() * 1.3))
        a.invert_yaxis()
        a.spines[["top", "right"]].set_visible(False)
    fig.savefig(args.output / "gate_comparison.png", dpi=180)
    fig.savefig(args.output / "gate_comparison.svg")
    plt.close(fig)
    print(json.dumps(completion, indent=2))
    print(totals.to_string(index=False))


if __name__ == "__main__":
    main()
