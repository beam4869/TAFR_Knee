"""Shared-solver TAFR pilot plus paired multi-start official SNEE NM."""
import json,subprocess,sys,os
from concurrent.futures import ThreadPoolExecutor,as_completed
from time import perf_counter
import numpy as np
from experiments.common import RESULTS,config_file,write_bundle,save_json,ROOT
from experiments.methods.snee_adapter import SneeProblem,official_core
from experiments.runners.run_pilots import evaluate

NAMES=['ZLT1','GRV1','VFM1','ZLT1q','GRV2','DAS1','DO2DK','DO2DKtight','VFM1constr']


def child(name,seed,variant):
    t=perf_counter()
    try:
        m=SneeProblem(name).n_objectives
        w=np.random.default_rng(seed).dirichlet(np.ones(m))
        r=official_core(name,'NM',seed,normalized=variant=='normalized',compat=True,start=w)
        r.update(status='completed',start=w)
    except Exception as e:r=dict(status='failed',error=repr(e))
    r.update(problem=name,seed=seed,algorithm='NM',variant=variant,wall_time_seconds=perf_counter()-t)
    print(json.dumps(__import__('experiments.common',fromlist=['jsonable']).jsonable(r)))


def worker(job):
    name,seed,variant=job;path=RESULTS/'raw/snee_multistart_v2'/f'{name}-{seed}-{variant}.json'
    if path.exists():return json.loads(path.read_text())
    try:
        r=subprocess.run([sys.executable,'-m','experiments.runners.run_snee_comparison','--child',name,str(seed),variant],
             cwd=ROOT,capture_output=True,text=True,timeout=60,env={**os.environ,'OPENBLAS_NUM_THREADS':'1'})
        row=json.loads(r.stdout.strip().splitlines()[-1])
    except Exception as e:row=dict(problem=name,seed=seed,variant=variant,status='timeout' if isinstance(e,subprocess.TimeoutExpired) else 'failed',error=repr(e))
    save_json(path,row);return row


def main():
    cfg=config_file('pmop_oracle');cfg.update(candidate_budget=17,validation_samples=256,phase='shared-solver pilot')
    rows=[];details=[]
    for name in ([] if '--multistart-only' in sys.argv else NAMES):
        try:
            p=SneeProblem(name)
            # Only symmetric cases have a justified analytic target here.
            target=p.solve(np.ones(p.n_objectives)).objectives if name in ('ZLT1','ZLT1q','GRV2') else None
            r,d=evaluate(p,name,'SNEE',cfg,None if target is None else [target]);rows.extend(r);details.append(dict(problem=name,status='completed',**d))
        except Exception as e:details.append(dict(problem=name,status='failed',error=repr(e)))
        print(name,details[-1]['status'],flush=True)
        write_bundle('snee_comparison_pilot',cfg,dict(rows=rows,instances=details))
    results=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        futures=[pool.submit(worker,(name,seed,variant)) for name in NAMES for seed in range(30) for variant in ['scipy_shape_compatibility','normalized']]
        for f in as_completed(futures):
            results.append(f.result())
            if len(results)%30==0:
                print('multi-start completed',len(results),flush=True)
                write_bundle('snee_multistart_v2',dict(seeds=list(range(30)),variants=['scipy_shape_compatibility','normalized'],per_job_timeout_seconds=60),results)
    write_bundle('snee_multistart_v2',dict(seeds=list(range(30)),variants=['scipy_shape_compatibility','normalized'],per_job_timeout_seconds=60),results)


if __name__=='__main__':
    if '--child' in sys.argv:child(sys.argv[2],int(sys.argv[3]),sys.argv[4])
    else:main()
