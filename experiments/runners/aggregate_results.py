"""Create analysis-ready rows, figure inputs and LaTeX tables from saved bundles."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from experiments.common import RESULTS,raw_row,save_json
from experiments.metrics.statistical_tests import compare
from experiments.benchmarks.adversarial_tabular import extreme_plateau,interior_cell


def read(name):return json.loads((RESULTS/'raw'/f'{name}.json').read_text())

def export(df,name):
    df.to_csv(RESULTS/'processed'/f'{name}.csv',index=False)
    return df


def table(df,name):
    df.to_latex(RESULTS/'tables'/f'{name}.tex',index=False,escape=True,float_format=lambda x:f'{x:.4g}',na_rep='NA')
    df.to_csv(RESULTS/'tables'/f'{name}.csv',index=False)


def main():
    rows=[];bundles={}
    extpath=RESULTS/'raw/validation_extensions.json'
    extensions=json.loads(extpath.read_text())['results'] if extpath.exists() else []
    extension_index={(r['bundle'],r['problem'],r['m'],r['method']):r for r in extensions}
    from experiments.benchmarks.pmop_suite import load_instance
    from experiments.runners.run_pilots import FastTable
    chim_valid={}
    for m in (3,5,8):
        for num in range(1,15):
            y,*_=load_instance(num,m);p=FastTable(y);anchors=np.array([p.solve(w).objectives for w in np.eye(m)])
            chim_valid[(f'PMOP{num}',m)]=int(np.linalg.matrix_rank(anchors[1:]-anchors[0]))==m-1
    names=['pmop_pilot','pmop_pilot_table_range','snee_comparison_pilot','ammonia_pilot','ammonia_pilot_global_range','discrete_pilot']
    for name in names:
        path=RESULTS/'raw'/f'{name}.json'
        if not path.exists():continue
        b=read(name);bundles[name]=b;r=b['results'];blocks=r if isinstance(r,list) else [r]
        for block in blocks:
            for row in block.get('rows',[]):
                v=raw_row(**{**row,'bundle':name,'git_sha':b['provenance']['git_sha'],
                    'external_commit_shas':b['provenance']['external_commit_shas']})
                ext=extension_index.get((name,v['problem_name'],v['n_objectives'],v['method']))
                if ext:
                    v.update({k:ext[k] for k in ['R_validation','validation_type','validation_samples','expected_displacement','persistence_probability'] if k in ext})
                    if v['R_reported'] is not None:v['audit_gap']=v['R_validation']-v['R_reported']
                    v['validation_extension']='validation_extensions.json'
                v['available']=True
                if v['method']=='CHIM-posthoc' and v['problem_family']=='PMOP' and not chim_valid[(v['problem_name'],v['n_objectives'])]:
                    v['available']=False;v['availability_reason']='payoff anchors are affinely dependent'
                    v['diagnostic_only_knee_error']=v['knee_error'];v['knee_error']=None;v['selected']=False
                v['false_reported_bound']=bool(v['certified'] and v['audit_gap'] is not None and v['audit_gap']>1e-7)
                # This is a bound-understatement flag, not a claim that the API
                # screens R <= objective_tolerance (the production API does not).
                for key,t in [('success_001',.01),('success_0025',.025),('success_005',.05)]:
                    v[key]=bool(v['knee_error'] is not None and v['knee_error']<=t)
                rows.append(v)
    corrections=json.loads((RESULTS.parents[1]/'experiments/PROVENANCE_CORRECTIONS.json').read_text())['execution_start_revisions']
    for v in rows:
        fix=corrections.get(v['bundle'],v['git_sha'])
        if isinstance(fix,dict):fix=fix.get(v['problem_name'],v['git_sha'])
        v['recorded_bundle_git_sha']=v['git_sha'];v['execution_start_sha']=fix
    save_json(RESULTS/'processed/all_rows.json',rows)
    df=pd.DataFrame(rows);serial=df.copy()
    for c in serial:
        if serial[c].map(lambda x:isinstance(x,(list,dict,np.ndarray))).any():serial[c]=serial[c].map(lambda x:json.dumps(x) if isinstance(x,(list,dict)) else x)
    export(serial,'all_rows')
    p=df[df.bundle=='pmop_pilot_table_range'].copy()
    summary=[]
    for method,g in p.groupby('method',sort=False):
        vals=g.knee_error.dropna().to_numpy(float)
        summary.append(dict(Method=method,Available=int(g.available.sum()),Returned=int(g.selected.sum()),Instances=len(g),
            Median=float(np.median(vals)) if len(vals) else None,Q1=float(np.quantile(vals,.25)) if len(vals) else None,
            Q3=float(np.quantile(vals,.75)) if len(vals) else None,
            Success0025=int(g.success_0025.sum()),BoundUnderstatements=int(g.false_reported_bound.sum())))
    table(pd.DataFrame(summary),'pmop_summary')
    p['instance_key']=p.problem_name+'-'+p.n_objectives.astype(str)
    pivot=p[p.available].pivot(index='instance_key',columns='method',values='knee_error')
    save_json(RESULTS/'processed/pilot_statistics.json',dict(label='exploratory one-seed finite-table blocks; no final significance claim',
        method_order=list(pivot.columns),tests=compare(pivot.to_numpy(float))))
    table(p[['problem_name','n_objectives','method','selected','knee_error','KD','KGD','KIGD','supported_knee_count','all_knee_count','R_reported','R_validation','audit_gap']],'pmop_per_instance')
    # All failed runs remain available separately, never numeric zero.
    failures=[]
    for name,b in bundles.items():
        result=b['results'];items=result.get('instances',[]) if isinstance(result,dict) else result
        for item in items:
            if item.get('status')=='failed':failures.append(dict(bundle=name,problem=item.get('problem'),error=item.get('error')))
    for name in ['snee_reproduction_v2','snee_multistart_v2']:
        path=RESULTS/'raw'/f'{name}.json'
        if not path.exists():continue
        b=read(name)
        for row in b['results']:
            if row['status']!='completed':failures.append(dict(bundle=name,problem=row['problem'],method=row.get('algorithm'),variant=row['variant'],seed=row.get('seed',0),error=row.get('error')))
        compact=pd.DataFrame([{k:r.get(k) for k in ['problem','algorithm','variant','seed','status','mcf','solver_calls','objective_calls','gradient_calls','hessian_calls','constraint_violation','wall_time_seconds']} for r in b['results']])
        export(compact,name)
        if name=='snee_reproduction_v2':table(compact[['problem','algorithm','variant','status','mcf','solver_calls']],'snee_reproduction')
        else:
            entries=[]
            for (problem,variant),g in compact.groupby(['problem','variant']):
                vals=g.loc[g.status=='completed','mcf'].dropna().to_numpy(float)
                entries.append(dict(Problem=problem,Variant=variant,Completed=len(vals),Runs=len(g),
                    Median=np.median(vals) if len(vals) else None,Q1=np.quantile(vals,.25) if len(vals) else None,Q3=np.quantile(vals,.75) if len(vals) else None))
            table(pd.DataFrame(entries),'snee_multistart_summary')
    export(pd.DataFrame(failures),'failures')
    phase1=read('adversarial_phase1')['results']
    export(pd.DataFrame(phase1['E0-D']['records']),'interior_trials')
    table(pd.DataFrame(phase1['E0-D']['summary']),'interior_detection')
    export(pd.DataFrame(phase1['E0-E']),'flat_gain')
    # Analytical curve/cell data are generated here from committed benchmark
    # definitions, so plotting consumes processed data without hand-entered results.
    y=extreme_plateau();w=np.linspace(0,1,1001)
    envelope=dict(w=w,costs=np.c_[w,1-w]@y.T,objectives=y)
    yi=interior_cell();simplex=[]
    for a in np.linspace(0,1,301):
        for b in np.linspace(0,1-a,301-int(round(300*a))):simplex.append([a,b,1-a-b])
    ww=np.array(simplex);cells=np.argmin(ww@yi.T,axis=1)
    save_json(RESULTS/'processed/phase1_plot_data.json',dict(results=phase1,envelope=envelope,
        cell_map=dict(weights=ww,selected=cells,objectives=yi)))
    a=df[df.bundle.str.startswith('ammonia_pilot')]
    table(a[['bundle','problem_name','method','selected','raw_objectives','R_reported','R_validation','exit_tradeoff']],'ammonia_results')
    standardized=list(rows)
    for name in ['snee_reproduction_v2','snee_multistart_v2']:
        b=read(name)
        for r in b['results']:
            standardized.append(raw_row(git_sha=corrections.get(name,b['provenance']['git_sha']),
                external_commit_shas=b['provenance']['external_commit_shas'],problem_family='SNEE',
                problem_name=r['problem'],method='SNEE-'+r.get('algorithm','NM'),variant=r['variant'],
                scalarization='weighted_sum',seed=r.get('seed',0),selected=r['status']=='completed',
                abstained=False,selected_weight=r.get('weight'),raw_objectives=r.get('objectives'),
                solver_calls=r.get('solver_calls'),objective_calls=r.get('objective_calls'),
                gradient_calls=r.get('gradient_calls'),hessian_calls=r.get('hessian_calls'),
                wall_time_seconds=r.get('wall_time_seconds'),status=r['status'],MCF=r.get('mcf'),
                error=r.get('error'),bundle=name,config=b['config']))
    b=read('adversarial_phase1')
    for r in phase1['E0-D']['records']:
        standardized.append(raw_row(git_sha=corrections['adversarial_phase1'],
            external_commit_shas=b['provenance']['external_commit_shas'],problem_family='adversarial',
            problem_name='E0-D',method='TAFR-audit',variant='adaptive' if r['adaptive_rounds'] else 'hybrid',
            n_objectives=3,n_variables=7,scalarization='weighted_sum',radius=.18,bundle='adversarial_phase1',
            config=b['config'],**r))
    with (RESULTS/'processed/standardized_runs.jsonl').open('w') as f:
        for r in standardized:f.write(json.dumps(r,allow_nan=False)+'\n')
    for name in ['scalability_pilot','ammonia_pilot','ammonia_pilot_global_range']:
        plot_bundle=read(name)
        if name.startswith('ammonia'):
            for rec in plot_bundle['results']:
                rec['rows']=[r for r in rows if r['bundle']==name and r['problem_name']==rec['problem']]
        save_json(RESULTS/'processed'/f'plot_bundle_{name}.json',plot_bundle)
    save_json(RESULTS/'processed/analysis_summary.json',dict(PMOP=summary,raw_rows=len(df),failure_rows=len(failures),
        source_bundles={name:b['provenance']['git_sha'] for name,b in bundles.items()}))


if __name__=='__main__':main()
