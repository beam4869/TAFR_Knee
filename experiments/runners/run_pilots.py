"""Bounded pilots using the unchanged production audit and declared configurations."""
from dataclasses import replace,asdict
from time import perf_counter
import numpy as np
from scipy.stats import qmc
from tafrknee import KneeConfig,SolveResult
from tafrknee.audit import audit_prepared_weight
from tafrknee.normalization import fit_normalizer
from tafrknee.reduction import identity_reduction
from tafrknee.weights import hit_and_run_samples
from experiments.runners.run_adversarial import ColdEngine
from experiments.common import config_file,write_bundle,raw_row,provenance,save_json,RESULTS
from experiments.metrics.knee_matching import metrics
from experiments.metrics.exact_tabular_oracle import ExactTabularOracle,canonical_priorities
from experiments.methods.nbi import chim_select
from experiments.methods.mavrotas_ri import profile
from experiments.methods.tpe_2024 import select as tpe_select


class FastTable:
    def __init__(self,y):
        # Sort once; lexicographic objective order determines exact ties.
        a=np.asarray(y);self.y=a[np.argsort(canonical_priorities(a))];self.n_objectives=a.shape[1]
    def solve(self,c,warm_start=None):
        scores=self.y@c;idx=int(np.flatnonzero(scores<=scores.min()+1e-12)[0])
        return SolveResult(idx,self.y[idx])


def candidates(m,budget,seed):
    u=qmc.Sobol(m,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(max(1,budget-1)))))[:budget-1]
    v=-np.log(np.maximum(u,1e-12));v/=v.sum(axis=1,keepdims=True)
    return np.vstack([np.full(m,1/m),v])


def evaluate(problem,name,family,cfg,knees=None,region=None,seed=0,table_exact=False):
    t0=perf_counter()
    if 'fixed_ideal' in cfg:
        normalizer,anchors=fit_normalizer(problem,ideal=cfg['fixed_ideal'],reference=cfg['fixed_reference'])
    elif cfg.get('normalization_policy')=='oracle_table_range':
        normalizer,anchors=fit_normalizer(problem,ideal=problem.y.min(axis=0),reference=problem.y.max(axis=0))
    else:normalizer,anchors=fit_normalizer(problem)
    payoff=np.array([a.objectives for a in anchors]);an=normalizer.transform(payoff)
    e=ColdEngine(problem,normalizer,identity_reduction(problem.n_objectives))
    ws=candidates(problem.n_objectives,cfg['candidate_budget'],seed)
    kc=KneeConfig(radius=cfg['radius'],objective_tolerance=cfg['objective_tolerance'],
        min_improvement=cfg['min_improvement'],min_deterioration=cfg['min_deterioration'],
        interior_samples=cfg['interior_samples'],stability_iterations=cfg['stability_iterations'],
        random_seed=seed,extreme_threshold=.05,audit_strategy='hybrid')
    audits=[];failures=[]
    for i,w in enumerate(ws):
        try:audits.append(audit_prepared_weight(e,w,an,kc))
        except Exception as error:failures.append(dict(candidate=i,error=repr(error)))
    valid=[a for a in audits if a.certified]
    chosen=min(valid,key=lambda a:(a.robustness,-a.stability_radius,-a.exit_tradeoff,tuple(a.weight))) if valid else None
    selection_time=perf_counter()-t0;selection_calls=e.solver_calls+len(anchors)
    original_cache_hits=e.cache_hits
    selected=[('TAFR-meaningful-gain',None if chosen is None else chosen.weight,chosen,selection_time,selection_calls)]
    selected.append(('equal',ws[0],None,None,None))
    if isinstance(problem,FastTable):
        y=normalizer.transform(problem.y);idx,meta=chim_select(y,an)
        selected.append(('CHIM-posthoc',None,None,None,None))
    else:y=None;idx=None;meta=None
    ts=perf_counter();calls_before=e.solver_calls
    legacy=tpe_select(lambda w:problem.solve(w).objectives,ws,kc.radius)
    selected.append(('TPE-2024-grid',legacy['weight'],None,perf_counter()-ts,None))
    # MC-RI remains a post-analysis diagnostic of equal and TAFR selections.
    # Turning RI into a search ranking is a distinct extension, not the source method.
    rows=[]
    for method,w,a,elapsed,calls in selected:
        values=None;yn=None;Rval=None;Rreported=None;expected=None;persistence=None;oracle_result=None
        if w is not None:
            if method=='TPE-2024-grid':
                values=legacy['raw_objectives'];w=w*normalizer.scale;w/=w.sum()
            else:values=e.solve(w).objectives
            yn=normalizer.transform(values)
            validation=hit_and_run_samples(w,kc.radius,cfg['validation_samples'],seed=seed+cfg['validation_seed_offset'],burn_in=64)
            vv=np.array([normalizer.transform(e.solve(u).objectives) for u in validation]);d=np.linalg.norm(vv-yn,axis=1)
            Rval=float(d.max());expected=float(d.mean());persistence=float(np.mean(d<=kc.objective_tolerance))
            if a:Rreported=a.robustness
            if table_exact and method=='TAFR-meaningful-gain':
                oracle_result=ExactTabularOracle(normalizer.transform(problem.y)).audit(w,kc.radius)
                Rval=oracle_result['robustness']
        elif method=='CHIM-posthoc':values=problem.y[idx];yn=y[idx]
        resultmetrics=metrics([] if yn is None else [yn],normalizer.transform(knees),normalizer.transform(region) if region is not None else None) if knees is not None else {}
        row=raw_row(problem_family=family,problem_name=name,n_objectives=problem.n_objectives,
            n_variables=getattr(problem,'n_variables',None),method=method,variant='pilot',scalarization='weighted_sum',seed=seed,
            candidate_budget=len(ws),radius=kc.radius,objective_tolerance=kc.objective_tolerance,extreme_threshold=kc.extreme_threshold,
            min_improvement=kc.min_improvement,min_deterioration=kc.min_deterioration,kappa_min=kc.kappa_min,
            selected=yn is not None,certified=bool(a and a.certified),abstained=yn is None,
            selected_weight=w,raw_objectives=values,normalized_objectives=yn,R_reported=Rreported,R_validation=Rval,
            audit_gap=None if Rreported is None else Rval-Rreported,
            stability_radius=None if a is None else a.stability_radius,
            exit_tradeoff=None if a is None else a.exit_tradeoff,
            solver_calls=calls,unique_solver_calls=calls,cache_hits=original_cache_hits if a else None,
            lower_level_failures=len(failures),wall_time_seconds=elapsed,
            expected_displacement=expected,persistence_probability=persistence,
            validation_type='exact finite table' if oracle_result else 'independent hit-and-run lower bound',
            validation_samples=cfg['validation_samples'],normalization_ideal=normalizer.ideal,normalization_reference=normalizer.reference,
            **resultmetrics)
        if a:
            # Persist absolute exits, including inactive ones; reported K is the
            # production first-active-shell statistic and is not a global ratio.
            from tafrknee.weights import perturbation_weights
            from tafrknee.metrics import tradeoff
            out=[]
            for u in perturbation_weights(w,min(1,a.stability_radius+kc.exit_step),strategy='hybrid',interior_samples=32,seed=seed+30001):
                alt=normalizer.transform(e.solve(u).objectives)
                if np.linalg.norm(alt-yn)>kc.objective_tolerance:out.append(asdict(tradeoff(yn,alt,min_improvement=kc.min_improvement,min_deterioration=kc.min_deterioration)))
            row['independent_exit_pairs']=out
            active=[r for r in out if r['active']]
            if active:
                low=min(active,key=lambda r:r['ratio']);row['exit_improvement']=low['improvement'];row['exit_deterioration']=low['deterioration']
            ri,curve=profile(lambda v:normalizer.transform(e.solve(v).objectives),w,yn,count=32,seed=seed+200000)
            row['MC_RI_postanalysis']=ri;row['MC_RI_curve']=curve
        rows.append(row)
    return rows,dict(config=asdict(kc),normalization_ideal=normalizer.ideal,normalization_reference=normalizer.reference,
        selection_audits=len(audits),audits=[dict(weight=a.weight,certified=a.certified,reasons=a.reasons,R=a.robustness,stability=a.stability_radius,exit_tradeoff=a.exit_tradeoff) for a in audits],candidate_failures=failures,CHIM=meta,all_solver_calls_including_validation=e.solver_calls)


def pmop():
    from experiments.benchmarks.pmop_suite import load_instance
    cfg=config_file('pmop_oracle');rows=[];instances=[]
    import sys
    suffix='_table_range' if '--table-range' in sys.argv else ''
    if suffix:cfg['normalization_policy']='oracle_table_range'
    for m in cfg['objectives']:
        for number in cfg['problems']:
            key=f'PMOP{number}-{m}';path=RESULTS/f'raw/pmop_individual{suffix}'/f'{key}.json'
            if path.exists():
                import json
                saved=__import__('json').loads(path.read_text());rows.extend(saved['rows']);instances.append(saved['instance']);continue
            t=perf_counter()
            try:
                y,k,region,meta=load_instance(number,m,cfg['front_samples']);p=FastTable(y)
                rr,info=evaluate(p,f'PMOP{number}','PMOP',cfg,k,region,table_exact=True)
                norm=(k-info['normalization_ideal'])/(info['normalization_reference']-info['normalization_ideal'])
                pool=(y-info['normalization_ideal'])/(info['normalization_reference']-info['normalization_ideal'])
                oracle=ExactTabularOracle(pool);support=oracle.supported(norm)
                # supported() returns bool; all-knee and supported-knee errors remain separate.
                support=np.array(support,dtype=bool)
                for r in rr:
                    r['supported_knee_count']=int(support.sum());r['all_knee_count']=len(k)
                    pred=[] if r['normalized_objectives'] is None else [r['normalized_objectives']]
                    r['supported_metrics']=metrics(pred,norm[support]) if support.any() else None
                meta.update(info,status='completed',supported_mask=support,wall_time_seconds=perf_counter()-t)
            except Exception as error:rr=[];meta=dict(problem=key,status='failed',error=repr(error))
            save_json(path,dict(rows=rr,instance=meta));rows.extend(rr);instances.append(meta)
            write_bundle('pmop_pilot'+suffix,cfg,dict(rows=rows,instances=instances))
            print(key,meta['status'],round(perf_counter()-t,2),flush=True)


if __name__=='__main__':pmop()
