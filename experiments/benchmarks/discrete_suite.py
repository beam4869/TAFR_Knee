"""Seeded 15-item project portfolios with exactly enumerated feasible vectors."""
import numpy as np
from experiments.benchmarks.pmop_suite import nondominated


def portfolio(seed=0,n=15,m=3):
    rng=np.random.default_rng(seed);cost=rng.integers(1,16,n)
    benefits=rng.integers(1,30,(n,m));capacity=int(.4*cost.sum())
    bits=((np.arange(2**n)[:,None]>>np.arange(n))&1).astype(float)
    feasible=bits[bits@cost<=capacity];y=-(feasible@benefits)
    return nondominated(np.unique(y,axis=0)),dict(seed=seed,n=n,m=m,capacity=capacity,item_costs=cost,
        benefits=benefits,feasible_count=len(feasible),enumerated_count=2**n)
