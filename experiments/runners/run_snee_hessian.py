"""Paired GRV2 derivative correction; retain the original implementation as a control."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np

from experiments.common import ROOT, RESULTS, config_file, jsonable, save_json, write_bundle
from experiments.methods.snee_adapter import official_core


def child(algorithm, normalization, hessian, seed):
    start = (np.random.default_rng(seed).dirichlet(np.ones(2))
             if algorithm == "NM" else None)
    t = perf_counter()
    try:
        row = official_core("GRV2", algorithm, seed=seed,
                            normalized=normalization == "normalized", compat=True,
                            start=start, corrected_grv2_hessian=hessian == "corrected_diagonal")
        row["status"] = "completed"
        # Analytic symmetric target, independent of SNEE's reported MCF.
        row["distance_to_analytic_knee"] = float(
            np.linalg.norm(np.asarray(row["objectives"]) - np.array([2., 2.])))
    except Exception as error:
        row = dict(status="failed", error=repr(error))
    row.update(problem="GRV2", algorithm=algorithm, normalization=normalization,
               hessian=hessian, seed=seed, initial_weight=start,
               wall_time_seconds=perf_counter() - t)
    print(json.dumps(jsonable(row)))


def worker(job):
    algorithm, normalization, hessian, seed, timeout = job
    path = RESULTS / "raw/snee_hessian" / f"{algorithm}-{normalization}-{hessian}-{seed}.json"
    if path.exists():
        return json.loads(path.read_text())
    try:
        process = subprocess.run(
            [sys.executable, "-m", "experiments.runners.run_snee_hessian", "--child",
             algorithm, normalization, hessian, str(seed)], cwd=ROOT,
            capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"})
        if process.returncode:
            raise RuntimeError(process.stderr[-2000:])
        row = json.loads(process.stdout.strip().splitlines()[-1])
    except Exception as error:
        row = dict(problem="GRV2", algorithm=algorithm, normalization=normalization,
                   hessian=hessian, seed=seed, error=repr(error),
                   status="timeout" if isinstance(error, subprocess.TimeoutExpired) else "failed")
    save_json(path, row)
    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--child", nargs=4)
    args = parser.parse_args()
    if args.child:
        child(*args.child[:3], int(args.child[3]))
        return
    cfg = config_file("snee_hessian")
    jobs = [(a, n, h, s, cfg["timeout_seconds"])
            for a in cfg["algorithms"] for n in cfg["normalizations"]
            for h in cfg["hessians"] for s in (cfg["seeds"] if a == "NM" else [0])]
    rows = []
    with ThreadPoolExecutor(max_workers=cfg["workers"]) as pool:
        for row in pool.map(worker, jobs):
            rows.append(row)
            if len(rows) % 12 == 0:
                print(f"GRV2 Hessian comparison: {len(rows)}/{len(jobs)}", flush=True)
                write_bundle("snee_hessian", cfg, rows)
    write_bundle("snee_hessian", cfg, rows)


if __name__ == "__main__":
    main()
