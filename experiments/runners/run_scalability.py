"""Geometry, coverage, grouped-objective counterexample and exact discrete pilot."""
from math import comb
from time import perf_counter
import numpy as np
from tafrknee.weights import perturbation_weights
from experiments.common import config_file,write_bundle
from experiments.runners.run_pilots import candidates,evaluate,FastTable
from experiments.benchmarks.adversarial_tabular import incomplete_vertices
from experiments.metrics.exact_tabular_oracle import ExactTabularOracle
from experiments.benchmarks.discrete_suite import portfolio
from experiments.methods.tafr_variants import awt_table
from experiments.methods.mavrotas_ri import profile


def main():
    cfg=config_file('scalability');counts=[];coverage=[]
    for m in cfg['objectives']:
        n=comb(m,m//2) if m%2==0 else m*comb(m-1,(m-1)//2)
        t=perf_counter();actual=None
        if m<=cfg['enumerate_through']:
            v=perturbation_weights(np.full(m,1/m),.2/m,strategy='vertices')
            actual=int(np.sum(np.max(np.abs(v-1/m),axis=1)>1e-12));assert actual==n
        counts.append(dict(m=m,lattice_H20=comb(20+m-1,m-1),vertices=n,enumerated=actual,
            enumeration_seconds=perf_counter()-t if actual else None,rho_scaled=.2/m,
            max_TV_interior=.2*(m//2)/m))
        if m>10:continue
        for seed in cfg['seeds']:
            w=candidates(m,10001,seed)[1:] # explicitly exclude injected equal weights
            f=(np.sum(w*w,axis=1)[:,None]+1-2*w)/2
            target=np.full(m,(1-1/m)/2)
            error=np.linalg.norm(f-target,axis=1)
            for B in cfg['candidate_budgets']:
                coverage.append(dict(m=m,seed=seed,budget=B,nearest_analytic_knee_error=float(error[:B].min()),
                    interpretation='Sobol candidate coverage only; not TAFR selection or solve-budget efficiency'))
    y=incomplete_vertices()/10;w=np.full(4,.25)
    correct=np.array([[.5,.5,0,0],[0,0,.5,.5]])
    wrong=np.array([[.5,0,.5,0],[0,.5,0,.5]])
    reductions=[]
    for name,A in [('correct',correct),('wrong',wrong)]:
        z=y@A.T
        r=ExactTabularOracle(z).audit([.5,.5],.2)
        original=ExactTabularOracle(y).audit(w,.1)
        reductions.append(dict(grouping=name,reduced_R=r['robustness'],original_R=original['robustness'],
            hidden_original_R=original['robustness'] if r['robustness']<1e-9 else 0,
            interpretation='different domains: reduced group weights vs full original perturbation box'))
    write_bundle('scalability_pilot',cfg,dict(counts=counts,coverage=coverage,reduction=reductions))
    ec=config_file('pmop_oracle');ec['candidate_budget']=17;records=[]
    for seed in (0,1,2):
        y,meta=portfolio(seed);p=FastTable(y)
        rr,info=evaluate(p,f'portfolio15-{seed}','discrete',ec,table_exact=True)
        # Same frozen range for WS and AWT; compare reachability and RI on equal weights.
        z=(p.y-info['normalization_ideal'])/(info['normalization_reference']-info['normalization_ideal'])
        awt=awt_table(z);w=np.full(3,1/3);nom=awt.solve(w).objectives
        ri,curve=profile(lambda u:awt.solve(u).objectives,w,nom,count=256,seed=seed)
        records.append(dict(instance=meta,rows=rr,info=info,AWT_MC_RI_equal=ri,AWT_MC_curve=curve))
        write_bundle('discrete_pilot',ec,records);print('portfolio',seed,'rows',len(y),flush=True)


if __name__=='__main__':main()
