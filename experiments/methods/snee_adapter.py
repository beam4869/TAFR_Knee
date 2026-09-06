"""Pinned SNEE code with explicit runtime/diagnostic adaptations."""
from __future__ import annotations

import importlib.util
import sys
import types
import numpy as np
import scipy
import sklearn
from scipy.optimize import brentq, minimize, NonlinearConstraint
from tafrknee import SolveResult
from experiments.common import ROOT


def load_upstream():
    # Upstream uses torch solely to seed an unused backend; do not require a
    # several-hundred-MB numerical package for those no-op seed calls.
    if importlib.util.find_spec("torch") is None and "torch" not in sys.modules:
        torch=types.ModuleType("torch")
        torch.Tensor=type("Tensor",(),{})
        torch.manual_seed=lambda seed: None
        torch.cuda=types.SimpleNamespace(manual_seed=lambda seed:None,manual_seed_all=lambda seed:None)
        torch.backends=types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False,deterministic=True))
        sys.modules["torch"]=torch
    path=str(ROOT/"external/snee")
    if path not in sys.path:
        sys.path.insert(0,path)
    from functions import SyntheticMultiobjProblem
    from snee import Snee
    return SyntheticMultiobjProblem,Snee


def make_upstream(name, algorithm="NM", seed=0, normalized=False, flatten_jacobian=False):
    Problem,Snee=load_upstream()
    f=Problem(name,seed=42) # fixed instance; optimizer seeds do not change GRV1
    run=Snee(f,1,True,algorithm,False,False,iprint=0,seed=seed)
    initial=np.random.RandomState(seed).uniform(0,1,(f.prob.dim,1))
    # SciPy now rejects (n,1) SLSQP objective Jacobians. This optional adapter
    # flattens that array only; upstream files and mathematical functions stay intact.
    if flatten_jacobian and f.prob.constrained:
        original=f.grad_f_weighted_vars
        f.grad_f_weighted_vars=lambda w,x: np.asarray(original(w,x)).flatten()
    scale=np.ones(f.prob.num_obj); ideal=np.zeros(f.prob.num_obj)
    if normalized:
        anchors=[]
        for w in np.eye(f.prob.num_obj):
            values,_=run.weighted_sum_method(initial,w)
            anchors.append([float(values[i]) for i in range(f.prob.num_obj)])
        payoff=np.array(anchors); ideal=np.diag(payoff); scale=payoff.max(axis=0)-ideal
        if np.any(scale<=0):
            raise ValueError("Degenerate SNEE payoff scale")
        for i in range(f.prob.num_obj):
            original_f=f.prob.f_dict[i]; original_g=f.prob.grad_f_dict[i]; original_h=f.prob.hess_f_dict[i]
            f.prob.f_dict[i]=lambda x,fn=original_f,j=i:(fn(x)-ideal[j])/scale[j]
            f.prob.grad_f_dict[i]=lambda x,fn=original_g,j=i:fn(x)/scale[j]
            f.prob.hess_f_dict[i]=lambda x,fn=original_h,j=i:fn(x)/scale[j]
    return f,run,initial,ideal,scale


def mcf(f,w,x):
    B=(f.jacob_F_lam_constr(w,x) if f.prob.constrained else f.jacob_F_lam(w,x))
    norms=np.linalg.norm(B,axis=0)
    ratios=norms[:,None]/np.maximum(norms[None,:],np.finfo(float).eps)
    np.fill_diagonal(ratios,-np.inf)
    return float(ratios.max())


def official_core(name,algorithm="NM",seed=0,normalized=False,compat=False,start=None):
    f,run,x0,ideal,scale=make_upstream(name,algorithm,seed,normalized,compat)
    # MCM/most-changing-front diagnostics never feed back into NM/DIRECT.
    # Disable ONLY those callbacks to isolate knee-search work and avoid the
    # upstream O(50**(q-1)) visualization grid for q=5.
    run.compute_mostchanging_subfronts=lambda *a,**k:(None,{},None,None,None,None,None)
    run.MCM=lambda *a,**k: float("nan")
    counts={"solver_calls":0,"objective_calls":0,"gradient_calls":0,"hessian_calls":0}
    lower=run.weighted_sum_method
    def solve(*args,**kwargs):
        counts["solver_calls"]+=1
        return lower(*args,**kwargs)
    run.weighted_sum_method=solve
    for attribute,key in [("f_dict","objective_calls"),("grad_f_dict","gradient_calls"),("hess_f_dict","hessian_calls")]:
        mapping=getattr(f.prob,attribute)
        for i,fn in list(mapping.items()):
            def wrapped(*a,fn=fn,key=key,**k):
                counts[key]+=1
                return fn(*a,**k)
            mapping[i]=wrapped
    guess=f.prob.weight_user[0] if start is None else np.asarray(start).reshape(-1,1)
    w,history,values,_,_=run.compute_knee_solutions(x0,guess,np.empty((f.prob.num_obj,0)),{})
    vals,x=run.weighted_sum_method(x0,w)
    transformed=np.array([float(vals[i]) for i in range(f.prob.num_obj)])
    violation=0.
    if f.prob.constrained:
        violation=max(0.,float(np.max(f.prob.inequality_constraints(x.flatten()))))
        if f.prob.num_eq_constr:
            violation=max(violation,float(np.max(np.abs(f.prob.equality_constraints(x.flatten())))))
    return dict(weight=w,decision=x.flatten(),objectives=transformed*scale+ideal,
                mcf=mcf(f,w,x),iterations=len(history),history=history,objective_history=values,
                normalization_ideal=ideal,normalization_scale=scale,
                constraint_violation=violation,**counts)


class SneeProblem:
    """Shared lower level using published objectives and checked solver status.

    Quadratic problems use their closed-form stationary solution. GRV2 uses a
    scalar monotone root. These are explicit lower-level variants for comparison;
    official reproduction always runs the authors' original BFGS/SLSQP path.
    """
    def __init__(self,name,dimension=None):
        P,_=load_upstream(); self.f=P(name,seed=42); self.name=name
        self.n_objectives=self.f.prob.num_obj
        if dimension is not None:
            if name=="ZLT1q":
                self.f.prob.dim=dimension; self.f.prob.num_obj=dimension
                self.f.prob.f_dict={i:(lambda y,i=i:self.f.prob.f_i(y,i)) for i in range(dimension)}
                self.f.prob.grad_f_dict={i:(lambda y,i=i:self.f.prob.grad_fi(y,i)) for i in range(dimension)}
                self.f.prob.hess_f_dict={i:(lambda y,i=i:self.f.prob.hess_fi(y,i)) for i in range(dimension)}
                self.n_objectives=dimension
            elif name=="GRV2": self.f.prob.dim=dimension
        self.n_variables=self.f.prob.dim
        self.calls=0

    def solve(self,c,warm_start=None):
        self.calls+=1
        c=np.asarray(c,float); f=self.f; n=self.n_variables
        if self.name in {"ZLT1","VFM1","GRV1","ZLT1q"}:
            z=np.zeros((n,1)); h=f.hess_f_weighted_vars(c,z); g=f.grad_f_weighted_vars(c,z)
            x=np.linalg.solve(h,-g).flatten()
        elif self.name=="GRV2":
            a=c[0]/c.sum()
            t=brentq(lambda t:a*(2*t/n+2*t**3)+(1-a)*(2*(t-2)/n+2*(t-2)**3),0,2)
            x=np.full(n,t)
        else:
            constraints=[NonlinearConstraint(lambda x:f.prob.inequality_constraints(x).flatten(),-np.inf,0,
                         jac=lambda x:f.prob.inequality_constraints_jacob(x).T)]
            if f.prob.num_eq_constr:
                constraints.append(NonlinearConstraint(lambda x:f.prob.equality_constraints(x).flatten(),0,0,
                                   jac=lambda x:f.prob.equality_constraints_jacob(x).T))
            result=minimize(lambda x:f.f_weighted(c,x.reshape(-1,1)),np.full(n,.25),
                            jac=lambda x:f.grad_f_weighted_vars(c,x.reshape(-1,1)).flatten(),
                            constraints=constraints,method="SLSQP",options={"ftol":1e-9,"maxiter":300})
            if not result.success:
                raise RuntimeError(f"SLSQP failed: {result.message}")
            x=result.x
        values=np.array([f.prob.f_dict[i](x.reshape(-1,1)) for i in range(self.n_objectives)]).flatten()
        return SolveResult(x,values,metadata={"solver":"analytic" if not f.prob.constrained else "SLSQP"})
