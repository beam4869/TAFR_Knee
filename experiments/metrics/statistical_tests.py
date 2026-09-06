"""Paired, failure-aware statistical reporting; never turn missing into wins."""
import numpy as np
from scipy.stats import friedmanchisquare,wilcoxon


def holm(pvalues):
    p=np.asarray(pvalues);order=np.argsort(p); adjusted=np.empty(len(p));last=0.
    for i,j in enumerate(order):last=max(last,min(1.,(len(p)-i)*p[j]));adjusted[j]=last
    return adjusted


def a12(a,b):
    a=np.asarray(a);b=np.asarray(b)
    return float(((a[:,None]>b).sum()+.5*(a[:,None]==b).sum())/(len(a)*len(b)))


def compare(matrix):
    x=np.asarray(matrix);complete=np.all(np.isfinite(x),axis=1);x=x[complete]
    out={'complete_blocks':len(x),'excluded_blocks':int((~complete).sum()),'pairs':[]}
    if len(x)<2:return out
    if x.shape[1]>=3 and not np.all(x==x[:,[0]]):out['friedman_p']=float(friedmanchisquare(*x.T).pvalue)
    for j in range(1,x.shape[1]):
        p=1. if np.allclose(x[:,0],x[:,j]) else float(wilcoxon(x[:,0],x[:,j],zero_method='pratt').pvalue)
        out['pairs'].append({'versus':j,'p':p,'A12_lower_is_better':1-a12(x[:,0],x[:,j])})
    for row,p in zip(out['pairs'],holm([r['p'] for r in out['pairs']])):row['holm_p']=float(p)
    return out
