"""Execute Track B without injecting reference knees or fixing distance variables."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
from time import perf_counter

from experiments.common import RESULTS, config_file, provenance, save_json, write_bundle
from experiments.methods.pmop_solver import PMOPProblem
from experiments.runners.run_paired_pmop import require_parity
from experiments.runners.run_pilots import evaluate


def worker(number, m, seed, cfg):
    path = RESULTS / 'raw/pmop_direct_pilot' / f'PMOP{number}-{m}-{seed}.json'
    if path.exists(): return json.loads(path.read_text())
    start = perf_counter()
    problem = PMOPProblem(number, m, seed=seed, maxiter=cfg['lower_maxiter'],
                          popsize=cfg['lower_popsize'])
    prov = provenance()
    try:
        rows, details = evaluate(problem, f'PMOP{number}', 'PMOP-direct', cfg, seed=seed)
        for row in rows:
            row.update(git_sha=prov['git_sha'], external_commit_shas=prov['external_commit_shas'],
                       instance=f'PMOP{number}-{m}', variant='full-space DE pilot',
                       core_screen_pass=row['certified'], certified=False,
                       certification_status='unavailable: heuristic lower-level map',
                       global_optimality_proven=False)
        outcome = dict(status='completed', rows=rows, details=details)
    except Exception as error:
        outcome = dict(status='failed', rows=[], error=repr(error))
    outcome.update(problem=f'PMOP{number}', m=m, seed=seed, config=cfg, provenance=prov,
                   lower_solver_logs=problem.logs, solver_calls=problem.calls,
                   objective_vector_calls=problem.objective_calls,
                   lower_budget_exhaustions=sum(not a['converged'] for a in problem.logs),
                   total_wall_time_seconds=perf_counter()-start)
    save_json(path, outcome)
    return outcome


def main():
    require_parity()
    cfg = config_file('pmop_direct_pilot')
    rows = []
    with ProcessPoolExecutor(max_workers=cfg['workers']) as pool:
        jobs = [pool.submit(worker, n, m, s, cfg) for n in cfg['problems']
                for m in cfg['objectives'] for s in cfg['seeds']]
        for job in as_completed(jobs):
            row = job.result(); rows.append(row)
            print(row['problem'], row['status'], row['solver_calls'], flush=True)
            write_bundle('pmop_direct_pilot', cfg, rows)


if __name__ == '__main__': main()
