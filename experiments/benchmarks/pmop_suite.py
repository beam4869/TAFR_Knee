# SPDX-License-Identifier: EPL-2.0
"""Analytic PF mapping transcribed from pinned PMOP{1,...,14}.m PF methods.

This is an oracle-front track, not a full-decision global optimizer. Official
knee/reference arrays are loaded without regeneration. Source license remains
in external/pmops. All objectives are minimized.
"""
from pathlib import Path
import numpy as np
from scipy.io import loadmat
from scipy.stats import qmc
ROOT = Path(__file__).resolve().parents[2]

SOURCE=ROOT/'external/pmops/PlatEMO/Problems/PMOPs'
A=[4,4,4,6,1,2,4,4,2,1,4,4,2,2]
S=[-1,2,2,-1,2,2,2,2,2,2,2,2,-2,-1]


def front(number, positions):
    x=np.atleast_2d(positions); m=x.shape[1]+1; a=A[number-1]; s=S[number-1]
    if number in (1,13): r=5+10*(x-.5)**2+np.cos(a*np.pi*x)/(a*2.**s)
    elif number in (2,7,11): r=1+np.exp(np.cos(a*np.pi*x+np.pi/2))/(a*2.**s)
    elif number in (3,8,12,14): r=1+np.exp(np.sin(a*np.pi*x+np.pi/2))/(a*2.**s)
    elif number==4: r=2+abs(np.sin(a*x)-np.cos(a*x-np.pi/4))/(a*2.**s)
    elif number in (5,10): r=2+np.minimum(np.sin(2*a*np.pi*x),np.cos(2*a*np.pi*x-np.pi/12))/2.**s
    else: r=2-np.exp(np.cos(a*np.pi*x)+.5*(np.cos(a*np.pi*x)-.5)**4)/(a*2.**s)
    dim=m-2 if number in (13,14) else m-1
    base=np.prod(r[:,:dim],axis=1)/dim
    transform={1:np.log,2:np.sqrt,3:lambda z:2**z,4:np.sqrt,5:lambda z:z**.4,
        6:lambda z:2**z,7:lambda z:3**z,8:lambda z:z,9:lambda z:z,
        10:lambda z:z**.2,11:lambda z:np.log(1/z+1),12:lambda z:z*z,13:np.sqrt,14:np.sqrt}
    k=transform[number](base)
    if number in (1,5,7,10,13,14): lead=x; tail=1-x
    elif number in (2,4,8,11): lead=np.cos(np.pi*x/2);tail=np.sin(np.pi*x/2)
    else: lead=1-np.cos(np.pi*x/2);tail=1-np.sin(np.pi*x/2)
    y=np.empty((len(x),m))
    for i in range(m):
        y[:,i]=k*np.prod(lead[:,:m-i-1],axis=1)
        if i: y[:,i]*=tail[:,m-i-1]
    if number==3:y*=2
    if number==10:y[:,::2]*=2
    return y


def matrix(path):
    d=loadmat(path)
    arrays=[v for k,v in d.items() if not k.startswith('_') and isinstance(v,np.ndarray) and v.ndim==2]
    if len(arrays)!=1:raise ValueError(f'Ambiguous MAT variables in {path}')
    return arrays[0]


def nondominated(y):
    # Bounded-memory, exact minimization dominance filtering.
    keep=np.ones(len(y),bool)
    for i in range(len(y)):
        if keep[i]:
            dominates=np.all(y<=y[i]+1e-12,axis=1)&np.any(y<y[i]-1e-12,axis=1)
            if np.any(dominates):keep[i]=False
    return y[keep]


def load_instance(number,m,n_front=512,seed=1729):
    knees=matrix(SOURCE/f'trueKnee/PMOP{number}-{m}.mat')
    region=matrix(SOURCE/f'KneeRefPoF/RefPMOP{number}_{m}_{m}.mat')
    x=qmc.Sobol(m-1,scramble=True,seed=seed).random_base2(int(np.ceil(np.log2(n_front))))[:n_front]
    # Include shape boundaries, supplied true knees and an evenly indexed
    # reference-region subset. This is deliberately a criterion-isolation pool.
    boundaries=np.array(list(__import__('itertools').product([0.,1.],repeat=m-1)))
    y=np.vstack([front(number,x),front(number,boundaries),knees,region[np.linspace(0,len(region)-1,min(128,len(region)),dtype=int)]])
    y=nondominated(np.unique(y,axis=0))
    return y,knees,region,dict(problem=f'PMOP{number}',m=m,n_variables=m+9,
        track='finite analytic/reference-front criterion isolation',front_draws=n_front,
        n_feasible_vectors=len(y),official_knee_count=len(knees),seed=seed,
        reference_injection='all methods receive the same table including official knees',
        python_port_validation='formula review and independent boundary checks; MATLAB runtime parity pending')
