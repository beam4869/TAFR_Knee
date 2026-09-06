"""Execute unmodified upstream MATLAB function bodies with Octave.

Only the function wrapper and PF random-input generation/filter are replaced:
the supplied input is shared by both languages, and we compare every objective
before nondominance filtering. No formula is translated for the reference side.
"""
import argparse
import hashlib
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

import numpy as np
from scipy.io import loadmat, savemat

from experiments.benchmarks.pmop_decision import objectives, distance_minimizer
from experiments.benchmarks.pmop_suite import SOURCE, front
from experiments.common import write_bundle


def source_wrappers(number, directory):
    path = SOURCE / f"PMOP{number}.m"
    source = path.read_text(errors="replace")
    cal = re.split(r"function\s+PopObj\s*=\s*CalObj\(obj,PopDec\)", source)[1]
    cal, pf = re.split(r"function\s+P\s*=\s*PF\(obj,N\)", cal)
    pf = pf.split("[FrontNo,")[0]
    pf, replaced = re.subn(r"PopDec\s*=\s*rand\([^;]+;", "", pf)
    if replaced != 1:
        raise ValueError(f"Unexpected PF input generation in {path}")
    (directory / f"cal_{number}.m").write_text(
        f"function PopObj = cal_{number}(obj,PopDec)\n" + cal)
    (directory / f"pf_{number}.m").write_text(
        f"function PopObj = pf_{number}(obj,PopDec)\nN=rows(PopDec);\n" + pf + "\nend\n")
    initialization = source.split("function PopObj")[0]
    constants = re.findall(r"obj\.Global\.(A|B|S|p|l)\s*=\s*(-?\d+(?:\.\d+)?)\s*;",
                           initialization)
    setup = "\n".join(f"obj.Global.{key}={value};" for key, value in constants)
    return setup, hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--octave", default=shutil.which("octave"))
    parser.add_argument("--samples", type=int, default=128)
    args = parser.parse_args()
    if not args.octave:
        raise RuntimeError("An Octave executable is required for independent MATLAB parity")
    seed = 271828
    rng = np.random.default_rng(seed)
    inputs = {}
    for m in (3, 5, 8):
        x = rng.uniform(size=(args.samples, m + 9))
        x[:, m - 1:] *= 10
        inputs[f"X{m}"] = np.vstack([x, np.zeros(m + 9),
                                      np.r_[np.ones(m - 1), np.full(10, 10.)]])
    rows = []
    with tempfile.TemporaryDirectory(prefix="pmop_parity_") as work:
        directory = Path(work)
        savemat(directory / "inputs.mat", inputs)
        script = ["load('inputs.mat');", "version_string=version;"]
        hashes = {}
        for number in range(1, 15):
            setup, hashes[f"PMOP{number}.m"] = source_wrappers(number, directory)
            for m in (3, 5, 8):
                key = f"p{number}m{m}"
                script += [f"obj=struct(); obj.Global=struct(); M={m}; D=M+9;",
                           "obj.Global.M=M; obj.Global.D=D; obj.Global.Linkage=0;",
                           "obj.Global.lower=zeros(1,D); obj.Global.upper=[ones(1,M-1),10*ones(1,10)];",
                           setup, f"Y_{key}=cal_{number}(obj,X{m});",
                           f"P_{key}=pf_{number}(obj,X{m}(:,1:M-1));",
                           "obj.Global.Linkage=1;",
                           f"try; L_{key}=cal_{number}(obj,X{m}); catch err; E_{key}=err.message; end;"]
        script.append("save('-mat7-binary','outputs.mat');")
        (directory / "parity.m").write_text("\n".join(script))
        result = subprocess.run([args.octave, "--no-gui", "--quiet", "parity.m"],
                                cwd=directory, capture_output=True, text=True, timeout=120)
        if result.returncode:
            raise RuntimeError(result.stderr[-5000:])
        data = loadmat(directory / "outputs.mat")
        for number in range(1, 15):
            for m in (3, 5, 8):
                key = f"p{number}m{m}"
                x = inputs[f"X{m}"]
                comparisons = [("CalObj", objectives(number, x, m), data[f"Y_{key}"]),
                               ("PF", front(number, x[:, :m - 1]), data[f"P_{key}"])]
                comparisons.append(("CalObj_Linkage1", objectives(number, x, m, True),
                                    data[f"L_{key}"]))
                for kind, python, reference in comparisons:
                    error = np.abs(python - reference)
                    rows.append(dict(problem=f"PMOP{number}", m=m, function=kind,
                                     vectors=len(x), max_absolute_error=float(error.max()),
                                     max_scaled_error=float((error / np.maximum(1, np.abs(reference))).max()),
                                     passed=bool(np.allclose(python, reference, rtol=1e-10, atol=1e-10))))
                # Check whether the usual distance-minimizer witness reproduces PF.
                witness = x.copy()
                witness[:, m - 1:] = distance_minimizer(number, 10)
                error = np.abs(objectives(number, witness, m) - front(number, x[:, :m - 1]))
                rows.append(dict(problem=f"PMOP{number}", m=m, function="PF_distance_witness",
                                 status="diagnostic", max_absolute_error=float(error.max()),
                                 passed=bool(np.allclose(error, 0, atol=1e-9))))
        config = dict(seed=seed, samples=args.samples, objectives=[3, 5, 8],
                      source_sha256=hashes, octave_version=str(data["version_string"].ravel()[0]),
                      reference="Unmodified pinned CalObj and PF formula bodies executed by Octave",
                      PF_adjustments="Supply shared positions and compare before NDSort",
                      rtol=1e-10, atol=1e-10)
    write_bundle("pmop_numeric_parity", config, rows)
    checks = [row for row in rows if row["function"] != "PF_distance_witness"]
    failures = [row for row in checks if not row["passed"]]
    print(f"MATLAB/Python parity: {len(checks) - len(failures)}/{len(checks)} checks pass")
    for row in failures:
        print(row)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
