"""Set metrics with explicit empty-set behavior and two KGD conventions."""
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


def metrics(predicted,knees,region=None,thresholds=(.01,.025,.05)):
    k=np.atleast_2d(knees);p=np.asarray(predicted).reshape(-1,k.shape[1]);q=k if region is None else np.atleast_2d(region)
    if not len(p):
        return dict(knee_error=None,KD=None,KGD=None,KIGD=None,KGD_code=None,
                    **{f'success_{str(t)}':False for t in thresholds},precision=0.,recall=0.,F1=0.)
    d=cdist(p,k);dq=cdist(p,q);nearest=d.min(axis=1)
    r,c=linear_sum_assignment(d);hits=int(np.sum(d[r,c]<=.025))
    precision=hits/len(p);recall=hits/len(k)
    return dict(knee_error=float(nearest.min()),KD=float(d.min(axis=0).mean()),
        KGD=float(dq.min(axis=1).mean()),KGD_code=float(np.linalg.norm(dq.min(axis=1))/len(p)),
        KIGD=float(dq.min(axis=0).mean()),precision=precision,recall=recall,
        F1=2*precision*recall/(precision+recall) if precision+recall else 0.,
        **{f'success_{str(t)}':bool(nearest.min()<=t) for t in thresholds})
