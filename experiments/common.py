"""Result provenance and serialization shared by all experiment runners."""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import subprocess
from dataclasses import asdict, is_dataclass
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "experiments/results"


def git_sha(path=ROOT):
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=path, text=True).strip()


def provenance():
    versions = {}
    for package in ("numpy", "scipy", "matplotlib", "pandas", "PyYAML", "scikit-learn"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    from scipy.optimize._highspy import _core
    versions["HiGHS"] = ".".join(str(getattr(_core,"HIGHS_VERSION_"+v)) for v in ("MAJOR","MINOR","PATCH"))
    cpu = platform.processor()
    if Path("/proc/cpuinfo").exists():
        cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                    if line.startswith("model name")), cpu)
    return dict(git_sha=git_sha(), external_commit_shas={
        name: git_sha(ROOT / "external" / name) for name in ("snee", "pmops", "ammonia") if (ROOT / "external" / name).exists()
    }, python=platform.python_version(), platform=platform.platform(), cpu=cpu,
                versions=versions, dirty=bool(subprocess.check_output(
                    ["git", "status", "--porcelain", "--untracked-files=no"], cwd=ROOT, text=True).strip()))


def jsonable(value):
    if is_dataclass(value):
        value = asdict(value)
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable(value.tolist())
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def config_file(name):
    path = ROOT / "experiments/configs" / (name + ".yaml")
    return yaml.safe_load(path.read_text())


def write_bundle(name, config, results):
    prov = provenance()
    bundle = dict(provenance=prov, config=config, results=results)
    path = RESULTS / "raw" / (name + ".json")
    save_json(path, bundle)
    path.with_suffix(".yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    return path


RAW_FIELDS = """run_id git_sha external_commit_shas problem_family problem_name instance
n_objectives n_variables method variant scalarization seed candidate_budget radius
objective_tolerance extreme_threshold min_improvement min_deterioration kappa_min
selected certified abstained selected_weight raw_objectives normalized_objectives knee_error
success_001 success_0025 success_005 R_reported R_validation audit_gap stability_radius
exact_stability_radius exit_improvement exit_deterioration exit_tradeoff activity_coverage
solver_calls unique_solver_calls cache_hits objective_calls gradient_calls hessian_calls
lower_level_failures wall_time_seconds""".split()


def raw_row(**values):
    # Null means unmeasured/not applicable, never a fabricated zero count.
    row = dict.fromkeys(RAW_FIELDS)
    row.update(values)
    row["run_id"] = hashlib.sha256(json.dumps(jsonable(values), sort_keys=True).encode()).hexdigest()[:20]
    return row
