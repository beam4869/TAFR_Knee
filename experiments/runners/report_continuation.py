"""Generate continuation tables and scientific figures from retained measurements."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.common import RESULTS, jsonable, save_json
from experiments.metrics.statistical_tests import compare

OUT = RESULTS / 'continuation'


def read(name):
    return json.loads((RESULTS / 'raw' / f'{name}.json').read_text())


def figure(fig, name):
    fig.savefig(OUT / f'{name}.svg', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=180, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('--allow-partial', action='store_true')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    jobs = [json.loads(p.read_text()) for p in sorted((RESULTS / 'raw/pmop_paired').glob('*.json'))]
    expected = 14 * 3 * 30
    if len(jobs) != expected and not args.allow_partial:
        raise RuntimeError(f'PMOP campaign incomplete: {len(jobs)}/{expected}; use --allow-partial only for diagnostics')
    all_rows = []
    pmop_rows = []
    failures = []
    for job in jobs:
        if job['status'] != 'completed':
            failures.append({k: job.get(k) for k in ('problem', 'm', 'seed', 'status', 'error')})
            for method in ('TAFR-meaningful-gain', 'equal', 'TPE-2024-grid'):
                pmop_rows.append(dict(method=method, instance=f"{job['problem']}-{job['m']}",
                                      seed=job['seed'], selected=False, success_0025=False,
                                      knee_error=None, method_failure=True))
        for row in job['rows']:
            all_rows.append(row)
            if row['method'] != 'CHIM-posthoc' or row.get('baseline_valid', False):
                pmop_rows.append({**row, 'method_failure': False})
    pmop = pd.DataFrame(pmop_rows)
    pmop.to_csv(OUT / 'pmop_all_runs.csv', index=False)
    # Inferential blocks are problem/dimension instances, not repeated copies of equal weights.
    per_instance = pmop.groupby(['instance', 'method'], as_index=False).agg(
        trials=('selected', 'size'), returns=('selected', 'sum'),
        successes=('success_0025', 'sum'), median_selected_error=('knee_error', 'median'),
        failures=('method_failure', 'sum'))
    per_instance['success_rate'] = per_instance['successes'] / per_instance['trials']
    per_instance.to_csv(OUT / 'pmop_per_instance.csv', index=False)
    summary = pmop.groupby('method', as_index=False).agg(
        trials=('selected', 'size'), returns=('selected', 'sum'),
        successes=('success_0025', 'sum'), median_selected_error=('knee_error', 'median'),
        failures=('method_failure', 'sum'))
    summary.to_csv(OUT / 'pmop_summary.csv', index=False)
    methods = ['TAFR-meaningful-gain', 'equal', 'TPE-2024-grid']
    blocks = per_instance.pivot(index='instance', columns='method', values='success_rate').reindex(columns=methods)
    statistics = compare(1 - blocks.to_numpy())
    statistics.update(metric='1 - within-instance Success@0.025 rate', methods=methods,
                      block='problem and objective count; 42 final blocks',
                      partial=len(jobs) != expected)
    save_json(OUT / 'paired_statistics.json', statistics)
    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    colors = ['#177e89', '#a44a3f', '#526777']
    for i, method in enumerate(methods):
        row = summary[summary.method == method].iloc[0]
        ax.bar(i - .16, row.returns / row.trials, width=.3, color=colors[i], alpha=.3)
        ax.bar(i + .16, row.successes / row.trials, width=.3, color=colors[i])
    ax.set_xticks(range(3), ['TAFR: meaningful gain', 'Equal weights', 'TPE: grid'])
    ax.set(ylabel='Fraction of scheduled seed trials', ylim=(0, 1.04),
           title='PMOP oracle tables: return rate (light) and Success@0.025 (dark)')
    fig.tight_layout(); figure(fig, 'pmop_paired_success')

    ablation = pd.DataFrame(read('ablation_paired')['results'])
    all_rows.extend(ablation.to_dict('records'))
    ablation.groupby(['problem_name', 'variant'], as_index=False).agg(
        trials=('selected', 'size'), returns=('selected', 'sum'),
        reported_understatements=('false_reported_bound', 'sum'),
        median_selected_error=('knee_error', 'median')).to_csv(OUT / 'ablation_summary.csv', index=False)
    matrix = ablation.pivot_table(index='problem_name', columns='variant', values='selected', aggfunc='mean')
    fig, ax = plt.subplots(figsize=(11.4, 4.8))
    im = ax.imshow(matrix, vmin=0, vmax=1, cmap='YlGnBu', aspect='auto')
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(matrix)), matrix.index)
    ax.set_title('Component ablations: fraction of seeds returning a selection')
    fig.colorbar(im, ax=ax, label='Return rate'); fig.tight_layout(); figure(fig, 'ablation_return_rates')
    audited = ablation[(ablation.validation_type == 'exact finite-table LP') & ablation.selected].copy()
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    audit = audited[audited.displacement_units == 'normalized']
    ax.scatter(audit.R_reported, audit.R_validation, s=15, alpha=.35, color='#177e89')
    end = max(audit.R_reported.max(), audit.R_validation.max()) * 1.03
    ax.plot([0, end], [0, end], '--', color='#526777', linewidth=1)
    ax.set(xlabel='Sampled reported displacement', ylabel='Exact finite-table displacement',
           title='Independent validation of the selected weights')
    fig.tight_layout(); figure(fig, 'ablation_exact_validation')

    snee = pd.DataFrame(read('snee_hessian')['results'])
    snee.groupby(['algorithm', 'normalization', 'hessian'], as_index=False).agg(
        trials=('status', 'size'), completed=('status', lambda s: (s == 'completed').sum()),
        median_knee_distance=('distance_to_analytic_knee', 'median'),
        median_mcf=('mcf', 'median')).to_csv(OUT / 'snee_hessian_summary.csv', index=False)
    nm = snee[snee.algorithm == 'NM'].pivot(index=['normalization', 'seed'],
        columns='hessian', values='distance_to_analytic_knee')
    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    ax.scatter(np.maximum(nm.pinned_upstream, 1e-14), np.maximum(nm.corrected_diagonal, 1e-14),
               color='#177e89', s=20)
    ends = [max(1e-14, nm.min().min() * .5), nm.max().max() * 2]
    ax.plot(ends, ends, '--', color='#526777')
    ax.set(xscale='log', yscale='log', xlabel='Original Hessian: distance to analytic knee',
           ylabel='Corrected Hessian: distance to analytic knee', title='GRV2: 60 pairs of NM runs')
    fig.tight_layout(); figure(fig, 'snee_hessian_parity')

    ammonia = read('ammonia_all_windows')
    ammonia_rows, schedule_checks = [], []
    for window in ammonia['results']:
        if window['status'] != 'completed':
            failures.append({k: window.get(k) for k in ('problem', 'status', 'error')}); continue
        for row in window['rows']:
            row = {**row, 'git_sha': ammonia['provenance']['git_sha'],
                   'external_commit_shas': ammonia['provenance']['external_commit_shas']}
            all_rows.append(row)
            raw = row['raw_objectives']
            ammonia_rows.append(dict(window=window['problem'], method=row['method'],
                stratum=window['window']['stratum'], selected=row['selected'], certified=row['certified'],
                cost=None if raw is None else raw[0], emissions=None if raw is None else raw[1],
                water=None if raw is None else raw[2], safety_throughput=None if raw is None else raw[3],
                R_reported=row['R_reported'], R_validation=row['R_validation']))
        for schedule in window['schedules']:
            if 'schedule' not in schedule: continue
            total = sum(schedule['schedule']['NH3'])
            water, safety = schedule['objectives'][2:]
            schedule_checks.append(dict(window=window['problem'], method=schedule['method'],
                total_NH3=total, water=water, safety_throughput=safety,
                affine_identity_residual=water - (4.5 * 33.3 * safety + 4.5 * 3 / 17 * total)))
    pd.DataFrame(ammonia_rows).to_csv(OUT / 'ammonia_twelve_windows.csv', index=False)
    checks = pd.DataFrame(schedule_checks)
    checks.to_csv(OUT / 'ammonia_objective_dependence.csv', index=False)
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ax.scatter(checks.safety_throughput, checks.water - 4.5 * 3 / 17 * checks.total_NH3,
               color='#a44a3f', s=24)
    grid = np.linspace(checks.safety_throughput.min(), checks.safety_throughput.max(), 100)
    ax.plot(grid, 4.5 * 33.3 * grid, '--', color='#526777')
    ax.set(xlabel='Safety proxy: electrolyzer throughput / 33.3',
           ylabel='Water minus the ammonia production term',
           title='Water-safety identity in the source model')
    fig.tight_layout(); figure(fig, 'ammonia_objective_dependence')

    direct = read('pmop_direct_pilot')['results']
    pd.DataFrame([{k: r.get(k) for k in ('problem', 'm', 'seed', 'status', 'solver_calls',
        'objective_vector_calls', 'lower_budget_exhaustions', 'total_wall_time_seconds', 'error')}
        for r in direct]).to_csv(OUT / 'pmop_direct_status.csv', index=False)
    for r in direct: all_rows.extend(r['rows'])
    pd.DataFrame(failures, columns=['problem', 'm', 'seed', 'status', 'error']).to_csv(OUT / 'failures.csv', index=False)
    with (OUT / 'standardized_runs.jsonl').open('w') as f:
        for row in all_rows: f.write(json.dumps(jsonable(row), allow_nan=False) + '\n')
    status = dict(pmop_expected_seed_jobs=expected, pmop_observed_seed_jobs=len(jobs),
        pmop_failed_seed_jobs=sum(j['status'] != 'completed' for j in jobs),
        ablation_rows=len(ablation), snee_hessian_rows=len(snee),
        ammonia_windows=len(ammonia['results']), pmop_direct_instances=len(direct),
        partial=len(jobs) != expected, raw_failure_records=len(failures),
        maximum_ammonia_affine_residual=float(checks.affine_identity_residual.abs().max()),
        lower_solver_warning='Track B DE iteration limits retained; no global certificates',
        timing_warning='PMOP checkpoints include both original LP and LP-validated accelerated stability; do not pool timing as a uniform implementation benchmark')
    save_json(OUT / 'completion.json', status)
    print(json.dumps(status, indent=2))


if __name__ == '__main__': main()
