"""One explicit entry point; unfinished tracks never emit synthetic results."""
import argparse,subprocess,sys,os
from experiments.common import ROOT,RESULTS


def main():
    p=argparse.ArgumentParser();p.add_argument('phase',choices=['smoke','adversarial','snee','pmop','scalability','ammonia','report']);args=p.parse_args()
    env={**os.environ,'PYTHONPATH':str(ROOT/'src'),'OPENBLAS_NUM_THREADS':'1'}
    commands={
      'smoke':[['-m','pytest','-q','tests/test_exact_oracle.py'],['-m','experiments.runners.run_diagnostics']],
      'adversarial':[['-m','experiments.runners.run_adversarial']],
      'snee':[['-m','experiments.runners.run_snee_reproduction'],['-m','experiments.runners.run_snee_comparison']],
      'pmop':[['-m','experiments.runners.run_pilots'],['-m','experiments.runners.run_pilots','--table-range']],
      'scalability':[['-m','experiments.runners.run_scalability']],
      'ammonia':[['-m','experiments.runners.run_ammonia'],['-m','experiments.runners.run_ammonia','--global-range']],
      'report':[['-m','experiments.runners.aggregate_results'],['-m','experiments.plots.make_figures'],['-m','experiments.note.make_note_assets']]}
    if args.phase not in ('smoke','adversarial','report'):
        import json
        gate=RESULTS/'raw/adversarial_phase1.json'
        if not gate.exists() or not json.loads(gate.read_text())['results']['acceptance']['exact_checks_passed']:
            raise RuntimeError('Phase-1 exact gate has not passed')
    for command in commands[args.phase]:subprocess.run([sys.executable,*command],cwd=ROOT,env=env,check=True)


if __name__=='__main__':main()
