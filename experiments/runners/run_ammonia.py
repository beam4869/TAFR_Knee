"""Source-traceable 48h window inventory and two-window direct MILP pilot."""
import hashlib
import numpy as np
import pandas as pd
from experiments.common import ROOT,RESULTS,config_file,write_bundle,save_json
from experiments.benchmarks.ammonia import AmmoniaProblem
from experiments.runners.run_pilots import evaluate


def inventory():
    base=ROOT/'external/ammonia';windows=[]
    for month in ('Jan','Apr','Aug','Oct'):
        path=base/f'datasets for ML/dataset of Jan Apr Aug Oct 2022 LA DA/{month} LA 2022 price and emission.xlsx'
        df=pd.read_excel(path);price=df.iloc[:,0].to_numpy(float);emission=df.iloc[:,1].to_numpy(float)
        for start in range(0,len(df)-47,48):
            p=price[start:start+48];e=emission[start:start+48]
            windows.append(dict(market='CAISO_LA',month=month,year=2022,row=start,price=p,emission=e,
                correlation=float(np.corrcoef(p,e)[0,1]),source=str(path.relative_to(ROOT)),
                source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    path=base/'datasets for ML/dataset of 2022 New England Hub DA/Dataset_of_2022_New_England_Hub_for_SVM.csv'
    df=pd.read_csv(path)
    # x1..x48 are the 48 prices; x49..x96 the 48 emission intensities.
    for start in range(0,len(df)-47,48):
        p=df.loc[start,[f'x{i}' for i in range(1,49)]].to_numpy(float)
        e=df.loc[start,[f'x{i}' for i in range(49,97)]].to_numpy(float)
        windows.append(dict(market='ISO_NE',year=2022,row=start,price=p,emission=e,
            correlation=float(np.corrcoef(p,e)[0,1]),source=str(path.relative_to(ROOT)),
            source_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    corr=np.array([w['correlation'] for w in windows]);q1,q3=np.quantile(corr,[.25,.75])
    for w in windows:w['stratum']='low' if w['correlation']<=q1 else 'high' if w['correlation']>=q3 else 'middle'
    selected=[]
    for stratum in ('low','middle','high'):
        pool=[w for w in windows if w['stratum']==stratum]
        # Two windows per market per stratum if both markets are represented.
        for market in ('CAISO_LA','ISO_NE'):
            subset=sorted([w for w in pool if w['market']==market],key=lambda w:w['correlation'])
            for i in np.linspace(0,len(subset)-1,2,dtype=int):selected.append(subset[i])
    return windows,selected


def main():
    cfg=config_file('ammonia');windows,chosen=inventory()
    import sys
    suffix='_global_range' if '--global-range' in sys.argv else ''
    if suffix:cfg.update(candidate_budget=17,interior_samples=8,stability_iterations=6,normalization_policy='global_objective_bounds',variant='global_bounds_with_larger_search_budget')
    all_windows = '--all-windows' in sys.argv
    bundle_name = ('ammonia_all_windows' if all_windows else 'ammonia_pilot') + suffix
    if all_windows:
        cfg.update(phase='12-window direct-MILP extension', enforce_validation_ratio=10)
    save_json(RESULTS/f'raw/ammonia_window_inventory{suffix}.json',dict(config=cfg,windows=windows,selected_12=chosen,
        units=['USD/kWh','kgCO2/kWh'],dates='source year/month and row identifiers; no inferred timestamps'))
    pilots=[next(w for w in chosen if w['stratum']=='low' and w['market']=='CAISO_LA'),
            next(w for w in chosen if w['stratum']=='high' and w['market']=='ISO_NE')]
    if all_windows:pilots=chosen
    results=[]
    existing=RESULTS/'raw'/f'{bundle_name}.json'
    if (suffix or all_windows) and existing.exists():
        import json
        results=json.loads(existing.read_text())['results']
    for w in pilots:
        name=f"{w['market']}-{w.get('month','year')}-row{w['row']}"
        if any(r['problem']==name for r in results):continue
        p=AmmoniaProblem(w['price'],w['emission'],cfg['lower_level_time_limit'],cfg['mip_relative_gap'])
        try:
            if suffix:
                cfg['fixed_ideal']=np.array([p.solve(v).objectives[i] for i,v in enumerate(np.eye(4))])
                cfg['fixed_reference']=np.array([p.solve(-v).objectives[i] for i,v in enumerate(np.eye(4))])
            rr,info=evaluate(p,name,'ammonia',cfg)
            schedules=[]
            for row in rr:
                if row['selected_weight'] is not None:
                    try:
                        scale=info['normalization_reference']-info['normalization_ideal']
                        sol=p.solve(np.array(row['selected_weight'])/scale)
                        schedules.append(dict(method=row['method'],objectives=sol.objectives,
                            schedule={k:sol.decision[v] for k,v in p.index.items()},solver_metadata=sol.metadata))
                    except Exception as e:schedules.append(dict(method=row['method'],error=repr(e)))
            out=dict(problem=name,status='completed',window=w,rows=rr,info=info,schedules=schedules)
        except Exception as e:out=dict(problem=name,status='failed',window=w,error=repr(e))
        out['solver_logs']=p.logs;out['total_MILP_calls']=p.calls
        results.append(out);write_bundle(bundle_name,cfg,results);print(name,out['status'],p.calls,flush=True)


if __name__=='__main__':main()
