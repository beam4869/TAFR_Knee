"""Complete exact baseline table audits and enforce 10x total audit-point validation."""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from time import perf_counter
import numpy as np
from tafrknee.weights import perturbation_weights,hit_and_run_samples
from experiments.common import ROOT,RESULTS,save_json,write_bundle
from experiments.benchmarks.pmop_suite import load_instance
from experiments.metrics.exact_tabular_oracle import ExactTabularOracle


def job(args):
    number,m,rows=args;t=perf_counter();y,*_=load_instance(number,m);out=[]
    for r in rows:
        if r['selected_weight'] is None or not r['selected']:continue
        z=(y-np.array(r['normalization_ideal']))/(np.array(r['normalization_reference'])-r['normalization_ideal'])
        oracle=ExactTabularOracle(z);a=oracle.audit(r['selected_weight'],r['radius'])
        out.append(dict(bundle='pmop_pilot_table_range',problem=r['problem_name'],m=m,method=r['method'],
            R_validation=a['robustness'],validation_type='exact finite table',lp_calls=oracle.lp_calls,
            wall_time_seconds=perf_counter()-t))
    print('exact baseline validation',number,m,flush=True);return out


def main():
    b=json.loads((RESULTS/'raw/pmop_pilot_table_range.json').read_text());rows=b['results']['rows'];jobs=[]
    for m in (3,5,8):
        for number in range(1,15):
            rr=[r for r in rows if r['problem_name']==f'PMOP{number}' and r['n_objectives']==m and r['method'] in ('equal','TPE-2024-grid')]
            jobs.append((number,m,rr))
    existing=RESULTS/'raw/validation_extensions.json'
    results=json.loads(existing.read_text())['results'] if existing.exists() else []
    done={(r['bundle'],r['problem'],r['m'],r['method']) for r in results}
    jobs=[j for j in jobs if not all(('pmop_pilot_table_range',f'PMOP{j[0]}',j[1],r['method']) in done for r in j[2])]
    with ProcessPoolExecutor(max_workers=3) as pool:
        for out in pool.map(job,jobs):
            results.extend(out);write_bundle('validation_extensions',dict(exact_baselines=True,continuous_multiplier=10),results)
    from experiments.benchmarks.discrete_suite import portfolio
    b=json.loads((RESULTS/'raw/discrete_pilot.json').read_text())
    for rec in b['results']:
        y,_=portfolio(rec['instance']['seed'])
        for r in rec['rows']:
            if r['selected_weight'] is None or r['method']=='TAFR-meaningful-gain':continue
            key=('discrete_pilot',r['problem_name'],3,r['method'])
            if key in done:continue
            z=(y-r['normalization_ideal'])/(np.array(r['normalization_reference'])-r['normalization_ideal'])
            oracle=ExactTabularOracle(z);a=oracle.audit(r['selected_weight'],r['radius'])
            results.append(dict(bundle=key[0],problem=key[1],m=3,method=key[3],R_validation=a['robustness'],validation_type='exact finite table',lp_calls=oracle.lp_calls))
    # Independent continuous validation extends stochastic budget when necessary.
    for name in ('snee_comparison_pilot','ammonia_pilot','ammonia_pilot_global_range'):
        b=json.loads((RESULTS/'raw'/f'{name}.json').read_text());records=b['results'] if isinstance(b['results'],list) else [b['results']]
        cfg=b['config']
        for record in records:
            if name.startswith('ammonia'):
                from experiments.benchmarks.ammonia import AmmoniaProblem
                win=record['window'];problem=AmmoniaProblem(win['price'],win['emission'])
            for r in record.get('rows',[]):
                if not r['selected'] or r['selected_weight'] is None:continue
                if (name,r['problem_name'],r['n_objectives'],r['method']) in done:continue
                w=np.array(r['selected_weight']);training=perturbation_weights(w,r['radius'],strategy='hybrid',interior_samples=cfg['interior_samples'],seed=r['seed'])
                count=max(256,10*len(training))
                if count<=r['validation_samples']:continue
                if name=='snee_comparison_pilot':
                    from experiments.methods.snee_adapter import SneeProblem
                    problem=SneeProblem(r['problem_name'])
                ideal=np.array(r['normalization_ideal']);scale=np.array(r['normalization_reference'])-ideal
                u=hit_and_run_samples(w,r['radius'],count,seed=r['seed']+100000,burn_in=64)
                alt=np.array([(problem.solve(v/scale).objectives-ideal)/scale for v in u])
                distances=np.linalg.norm(alt-np.array(r['normalized_objectives']),axis=1)
                results.append(dict(bundle=name,problem=r['problem_name'],m=r['n_objectives'],method=r['method'],
                    R_validation=float(distances.max()),expected_displacement=float(distances.mean()),
                    persistence_probability=float(np.mean(distances<=r['objective_tolerance'])),
                    validation_type='independent hit-and-run lower bound',validation_samples=count,
                    training_audit_points=len(training)))
                write_bundle('validation_extensions',dict(exact_baselines=True,continuous_multiplier=10),results)
    write_bundle('validation_extensions',dict(exact_baselines=True,continuous_multiplier=10),results)


if __name__=='__main__':main()
