"""Explicit proposal reconstruction: raw-space squared displacement, pairwise audit.

Grid outer search is an experiment adapter, not the proposal's KKT implementation.
No normalization, non-extreme, meaningful-gain, or abstention screen is added.
"""
import numpy as np
from tafrknee.weights import perturbation_weights


def select(solve,weights,radius):
    rows=[]
    for w in weights:
        y=solve(w);u=perturbation_weights(w,radius,strategy='pairwise')
        risk=max(np.sum((solve(v)-y)**2) for v in u)
        rows.append(dict(weight=w,raw_objectives=y,raw_squared_R=float(risk)))
    return rows[int(np.argmin([r['raw_squared_R'] for r in rows]))]
