"""Mavrotas-style relative-weight Monte Carlo persistence, DOI 10.1016/j.ejor.2014.06.039.

Joint uniform draws on the relative box intersected with the simplex are obtained
by rejection in m-1 coordinates, not by renormalizing independent cube samples.
Continuous persistence uses a declared normalized objective tolerance; it is an
experimental tolerance choice, not an assertion of exact published equality.
"""
import numpy as np


def relative_samples(w,alpha,count,seed):
    w=np.asarray(w,float);lo=np.maximum(0,w*(1-alpha));hi=np.minimum(1,w*(1+alpha))
    active=np.flatnonzero(hi-lo>1e-14)
    if len(active)<2:return np.tile(w,(count,1))
    residual=active[np.argmax((hi-lo)[active])];draw=active[active!=residual]
    rng=np.random.default_rng(seed);accepted=[];total=0
    for _ in range(10000):
        v=np.tile(w,(max(64,2*(count-total)),1))
        v[:,draw]=rng.uniform(lo[draw],hi[draw],(len(v),len(draw)))
        v[:,residual]=1-v[:,np.arange(len(w))!=residual].sum(axis=1)
        v=v[(v[:,residual]>=lo[residual])&(v[:,residual]<=hi[residual])]
        accepted.append(v);total+=len(v)
        if total>=count:return np.vstack(accepted)[:count]
    raise RuntimeError('Relative-simplex rejection budget exhausted')


def profile(solve,w,nominal,radii=np.linspace(0,1,6),count=32,seed=0,tolerance=.001):
    rows=[]
    for i,a in enumerate(radii):
        u=relative_samples(w,a,count,seed+1009*i)
        y=np.asarray([solve(v) for v in u]);d=np.linalg.norm(y-nominal,axis=1)
        rows.append(dict(relative_radius=float(a),persistence=float(np.mean(d<=tolerance)),
            expected_displacement=float(d.mean()),R_sampled=float(d.max()),solver_calls=len(u)))
    ri=float(np.trapezoid([r['persistence'] for r in rows],radii)/(radii[-1]-radii[0]))
    return ri,rows
