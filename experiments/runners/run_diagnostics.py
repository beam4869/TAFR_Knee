"""Independent derivative, metric and source-data diagnostics."""
import numpy as np
from experiments.common import write_bundle
from experiments.methods.snee_adapter import SneeProblem,make_upstream
from experiments.metrics.knee_matching import metrics
from experiments.methods.mavrotas_ri import relative_samples
from experiments.benchmarks.pmop_suite import front


def main():
    f,_,_,_,_=make_upstream('GRV2');x=np.array([[.3],[1.2]]);step=1e-5
    out={}
    for i in range(2):
        analytic=f.prob.hess_f_dict[i](x)
        numerical=np.column_stack([(f.prob.grad_f_dict[i](x+step*np.eye(2)[:,[j]])-f.prob.grad_f_dict[i](x-step*np.eye(2)[:,[j]])).flatten()/(2*step) for j in range(2)])
        out[f'GRV2_H{i+1}']=dict(upstream=analytic,finite_difference=numerical,max_error=float(np.abs(analytic-numerical).max()),
            interpretation='upstream np.diag on a column vector produces broadcasting; retained in published reproduction')
    extensions=[]
    for name,dims in [('ZLT1q',[3,5,8,10]),('GRV2',[2,10,50,100])]:
        for d in dims:
            p=SneeProblem(name,d);w=np.full(p.n_objectives,1/p.n_objectives);r=p.solve(w)
            grad=p.f.grad_f_weighted_vars(w,r.decision.reshape(-1,1))
            extensions.append(dict(problem=name,dimension=d,equal_objectives=r.objectives,stationarity=float(np.linalg.norm(grad))))
    out['dimension_extension_sanity']=extensions
    checks=[]
    for number in range(1,15):
        y=front(number,np.array([[0.,0.],[1.,1.],[.3,.7]]))
        checks.append(dict(problem=number,finite=bool(np.isfinite(y).all()),nonnegative=bool((y>=0).all())))
    out['PMOP_boundary_checks']=checks
    # Hungarian matching limits duplicate predictions to one match.
    mm=metrics([[.4,.4],[.4,.4]],[[.4,.4]])
    assert mm['precision']==.5 and mm['recall']==1
    out['duplicate_prediction_check']=mm
    w=np.array([.2,.3,.5]);u=relative_samples(w,.4,10000,33)
    assert np.max(abs(u.sum(axis=1)-1))<1e-12
    assert np.all(u>=.6*w) and np.all(u<=1.4*w)
    out['relative_sampler_check']=dict(sample_mean=u.mean(axis=0),samples=len(u),simplex_error=float(np.abs(u.sum(axis=1)-1).max()))
    write_bundle('diagnostics',dict(finite_difference_step=step),out)


if __name__=='__main__':main()
