"""Generate all quantitative note macros/tables from measured results."""
import json
import numpy as np
import pandas as pd
from experiments.common import ROOT,RESULTS

DIR=ROOT/'experiments/note'

def read(name):return json.loads((RESULTS/'raw'/f'{name}.json').read_text())

def main():
    r=read('adversarial_phase1')['results'];mac={
        'RpairC':r['E0-C']['pairwise']['R'],'RfullC':r['E0-C']['vertices']['R'],
        'RvertexD':r['E0-D']['vertex_R'],'RexactD':r['E0-D']['exact']['robustness'],
        'RadiusA':r['E0-A']['exact_stability'],'ExitA':r['E0-A']['exit']['ratio']}
    for row in r['E0-D']['summary']:
        n=row['samples'];rnd=row['adaptive_rounds']
        if n in (16,128) and rnd==0:mac['Detect'+('Sixteen' if n==16 else 'OneTwoEight')]=int(round(row['detection_probability']*100))
    lines=[]
    for k,v in mac.items():lines.append('\\newcommand{\\'+k+'}{'+(str(v) if isinstance(v,int) else f'{v:.6g}')+'}')
    (DIR/'numbers.tex').write_text('\n'.join(lines)+'\n')
    p=pd.read_csv(RESULTS/'tables/pmop_summary.csv');out=[]
    for _,row in p.iterrows():
        if row.Method=='CHIM-posthoc':continue
        name={'TAFR-meaningful-gain':'TAFR gain variant','equal':'Equal normalized weights','TPE-2024-grid':'TPE-2024-grid'}[row.Method]
        out.append({'Method':name,'Returned':f'{row.Returned}/42','Error median [Q1, Q3]':f'{row.Median:.4f} [{row.Q1:.4f}, {row.Q3:.4f}]','Success@0.025':f'{row.Success0025}/42'})
    pd.DataFrame(out).to_latex(DIR/'table_pmop.tex',index=False,escape=True)
    ms=pd.read_csv(RESULTS/'tables/snee_multistart_summary.csv');out=[]
    for name,g in ms.groupby('Problem'):
        a=g[g.Variant=='scipy_shape_compatibility'].iloc[0];b=g[g.Variant=='normalized'].iloc[0]
        out.append({'Problem':name,'Published-scale MCF':f'{a.Median:.4f} [{a.Q1:.4f}, {a.Q3:.4f}]',
            'Normalized MCF':f'{b.Median:.4f} [{b.Q1:.4f}, {b.Q3:.4f}]','Runs per variant':int(a.Runs)})
    pd.DataFrame(out).to_latex(DIR/'table_snee.tex',index=False,escape=True)
    out=[]
    for file,variant in [('ammonia_pilot','Payoff'),('ammonia_pilot_global_range','Global')]:
        for record in json.loads((RESULTS/'processed'/f'plot_bundle_{file}.json').read_text())['results']:
            if record['status']!='completed':continue
            market='LA' if record['problem'].startswith('CAISO') else 'NE'
            for row in record['rows']:
                name={'TAFR-meaningful-gain':'TAFR','equal':'Equal','TPE-2024-grid':'TPE'}[row['method']]
                y=row['raw_objectives'];rv=row['R_validation']
                out.append({'Case':market+'/'+variant,'Method':name,'Cost':None if y is None else y[0],
                    'CO2':None if y is None else y[1],'Water':None if y is None else y[2],
                    'Safety':None if y is None else y[3], 'Validation R':rv})
    pd.DataFrame(out).to_latex(DIR/'table_ammonia.tex',index=False,escape=True,float_format=lambda x:f'{x:.3g}',na_rep='NA')
    # Compact first-seed exact portfolio table.
    out=[]
    for rec in read('discrete_pilot')['results']:
        a=next(r for r in rec['rows'] if r['method']=='TAFR-meaningful-gain')
        out.append({'Seed':rec['instance']['seed'],'Feasible portfolios':rec['instance']['feasible_count'],
            'TAFR selected':a['selected'],'Exact R':a['R_validation'],'Exit ratio':a['exit_tradeoff'],
            'AWT equal-weight RI':rec['AWT_MC_RI_equal']})
    pd.DataFrame(out).to_latex(DIR/'table_discrete.tex',index=False,escape=True,float_format=lambda x:f'{x:.4f}')
    # Claim ledger points to source files and their analysis functions.
    ledger={'exact_numbers':{'values':mac,'raw':'adversarial_phase1.json','analysis':'make_note_assets.main'},
        'PMOP_table':{'raw':'pmop_pilot_table_range.json','analysis':'aggregate_results.main'},
        'SNEE_table':{'raw':'snee_multistart_v2.json','analysis':'aggregate_results.main'},
        'ammonia_table':{'raw':['ammonia_pilot.json','ammonia_pilot_global_range.json'],'analysis':'make_note_assets.main'},
        'discrete_table':{'raw':'discrete_pilot.json','analysis':'make_note_assets.main'}}
    (DIR/'claim_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')


if __name__=='__main__':main()
