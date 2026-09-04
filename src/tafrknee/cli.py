"""Command-line interface for tabular objective data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .api import discover_knees, select_knee
from .config import KneeConfig
from .problem import TabularProblem


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="tafr-knee")
    parser.add_argument("command", choices=("select", "discover"))
    parser.add_argument("csv", type=Path, help="CSV containing one objective vector per row")
    parser.add_argument("--delimiter", default=",")
    parser.add_argument("--skiprows", type=int, default=0)
    parser.add_argument("--radius", type=float, default=0.05)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--eta-ext", type=float, default=0.05)
    parser.add_argument("--kappa-min", type=float, default=1.0)
    parser.add_argument("--resolution", type=int, default=20)
    parser.add_argument("--mode", choices=("fixed_radius", "max_stability"), default="fixed_radius")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--compact", action="store_true", help="omit per-weight audits")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    objectives = np.loadtxt(args.csv, delimiter=args.delimiter, skiprows=args.skiprows, ndmin=2)
    problem = TabularProblem(objectives)
    config = KneeConfig(
        radius=args.radius,
        objective_tolerance=args.epsilon,
        extreme_threshold=args.eta_ext,
        kappa_min=args.kappa_min,
        candidate_resolution=args.resolution,
    )
    function = select_knee if args.command == "select" else discover_knees
    result = function(problem, config=config, mode=args.mode)
    payload = json.dumps(result.to_dict(include_audits=not args.compact), indent=2)
    if args.output is None:
        print(payload)
    else:
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0 if result.found else 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
