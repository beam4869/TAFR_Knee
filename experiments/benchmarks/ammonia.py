"""48h MILP port of beam4869/ML_scheduling MHS_cost_emiss.py.

Eliminated variables are equality-defined or unused in the four objectives:
energy, grid buy/sell (unbounded with zero wind), start counts, NH3 deviations.
Original startup counts are NOT part of this source's safety objective.
"""
from pathlib import Path
import numpy as np
from scipy.optimize import Bounds,LinearConstraint,milp
from scipy.sparse import coo_matrix
from tafrknee import SolveResult
from experiments.common import ROOT


class AmmoniaProblem:
    n_objectives=4
    def __init__(self,price,emission,time_limit=10,mip_gap=.001):
        self.price=np.asarray(price);self.emission=np.asarray(emission);T=len(price)
        if T!=48:raise ValueError('This case preserves the original 48-hour target')
        self.T=T;self.n_variables=8*T;self.time_limit=time_limit;self.mip_gap=mip_gap;self.calls=0;self.logs=[]
        self.blocks=['H2','N2','NH3','fossil','stock_H2','stock_N2','electrolyzers','ramp']
        self.index={key:np.arange(i*T,(i+1)*T) for i,key in enumerate(self.blocks)}
        ix=self.index;n=self.n_variables;lo=np.zeros(n);hi=np.full(n,np.inf)
        for key,l,u in [('H2',0,499.5),('N2',0,2250),('NH3',750,1100),('fossil',0,100),
                        ('stock_H2',4035,807197),('stock_N2',22601,4.52e6),('electrolyzers',0,15),('ramp',0,1)]:lo[ix[key]]=l;hi[ix[key]]=u
        for key,z in [('stock_H2',4035),('stock_N2',22601)]:lo[ix[key][-1]]=hi[ix[key][-1]]=z
        integers=np.zeros(n);integers[ix['electrolyzers']]=1;integers[ix['ramp']]=1
        entries=[];lbs=[];ubs=[]
        def add(coefs,l=-np.inf,u=np.inf):
            r=len(lbs);entries.extend((r,c,v) for c,v in coefs.items());lbs.append(l);ubs.append(u)
        for t in range(T):
            add({ix['H2'][t]:1,ix['electrolyzers'][t]:-9.99},l=0)
            add({ix['H2'][t]:1,ix['electrolyzers'][t]:-33.3},u=0)
            for stock,prod,ratio,initial in [('stock_H2','H2',3/17,4035),('stock_N2','N2',14/17,22601)]:
                c={ix[stock][t]:1,ix[prod][t]:-1,ix['NH3'][t]:ratio}
                if stock=='stock_H2':c[ix['fossil'][t]]=-1
                if t:c[ix[stock][t-1]]=-1
                rhs=initial if t==0 else 0;add(c,rhs,rhs)
            for factor,isupper in [(-40,True),(100,False)]:
                c={ix['NH3'][t]:1,ix['ramp'][t]:factor}
                if t:c[ix['NH3'][t-1]]=-1
                rhs=1000 if t==0 else 0
                add(c,u=rhs) if isupper else add(c,l=rhs)
            add({ix['ramp'][k]:1 for k in range(max(0,t-3),t+1)},u=1)
        add({j:1 for j in ix['NH3']},l=45000)
        rr,cc,vv=zip(*entries);self.A=coo_matrix((vv,(rr,cc)),shape=(len(lbs),n)).tocsr()
        self.lbs=np.array(lbs);self.ubs=np.array(ubs);self.bounds=Bounds(lo,hi);self.integrality=integers
        self.constraint=LinearConstraint(self.A,self.lbs,self.ubs)
        self.C=np.zeros((4,n))
        for key,rho,extra in [('H2',60,.42),('N2',.8,.10),('NH3',2.12,0)]:
            self.C[0,ix[key]]=rho*self.price+extra;self.C[1,ix[key]]=rho*self.emission
        self.C[0,ix['fossil']]=2.3;self.C[1,ix['fossil']]=9.3
        self.C[2,ix['H2']]=9;self.C[2,ix['fossil']]=4.5;self.C[3,ix['H2']]=1/33.3
    def solve(self,c,warm_start=None):
        self.calls+=1;cost=np.asarray(c)@self.C
        # Positive rescaling improves HiGHS numerics without changing minimizers.
        cost/=max(np.max(np.abs(cost)),1e-12)
        r=milp(cost,integrality=self.integrality,bounds=self.bounds,constraints=self.constraint,
               options={'time_limit':self.time_limit,'mip_rel_gap':self.mip_gap})
        log=dict(status=int(r.status),message=r.message,mip_gap=getattr(r,'mip_gap',None),nodes=getattr(r,'mip_node_count',None))
        self.logs.append(log)
        if r.x is None or r.status!=0:raise RuntimeError(f'MILP not certified to requested gap: {log}')
        v=self.A@r.x;violation=max(0,float(np.max(self.lbs-v)),float(np.max(v-self.ubs)))
        log['constraint_violation']=violation
        return SolveResult(r.x,self.C@r.x,metadata=log)
