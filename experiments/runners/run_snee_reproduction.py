"""Run isolated upstream knee-search reproductions with retained failures."""
import argparse
import json
import os
import subprocess
import sys
import traceback
from time import perf_counter
import numpy as np
from experiments.common import ROOT,RESULTS,config_file,save_json,write_bundle
from experiments.methods.snee_adapter import official_core


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--child",nargs=3)
    args=parser.parse_args()
    if args.child:
        name,algorithm,variant=args.child; t=perf_counter()
        try:
            row=official_core(name,algorithm,normalized=variant=="normalized",compat=variant!="published")
            row["status"]="completed"
        except Exception as exc:
            row=dict(status="failed",error=type(exc).__name__+": "+str(exc),traceback=traceback.format_exc())
        row.update(problem=name,algorithm=algorithm,variant=variant,wall_time_seconds=perf_counter()-t)
        save_json(RESULTS/"raw/snee_individual_v2"/f"{name}_{algorithm}_{variant}.json",row)
        return
    cfg=config_file("snee_reproduction"); rows=[]
    for name in cfg["problems"]:
        for algorithm in cfg["algorithms"]:
            for variant in cfg["variants"]:
                path=RESULTS/"raw/snee_individual_v2"/f"{name}_{algorithm}_{variant}.json"
                if not path.exists():
                    try:
                        proc=subprocess.run([sys.executable,"-m",__name__.replace("__main__","experiments.runners.run_snee_reproduction"),
                                            "--child",name,algorithm,variant],cwd=ROOT,capture_output=True,text=True,
                                            timeout=cfg["timeout_seconds"])
                        if not path.exists():
                            save_json(path,dict(problem=name,algorithm=algorithm,variant=variant,status="failed",
                                                error=proc.stderr[-2500:],returncode=proc.returncode))
                    except subprocess.TimeoutExpired:
                        save_json(path,dict(problem=name,algorithm=algorithm,variant=variant,status="timeout",
                                            timeout_seconds=cfg["timeout_seconds"]))
                row=json.loads(path.read_text());rows.append(row)
                print(name,algorithm,variant,row["status"],row.get("mcf"),flush=True)
                write_bundle("snee_reproduction_v2",{**cfg,"adapter_version":2},rows)


if __name__=="__main__": main()
