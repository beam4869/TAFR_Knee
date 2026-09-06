"""Publication figures from processed records; no hand-transcribed measurements."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from experiments.common import RESULTS

plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
    'pdf.fonttype':42,'ps.fonttype':42,'savefig.dpi':220,'axes.titleweight':'bold'})
COLORS=['#0f766e','#d97706','#2563eb','#be123c','#7c3aed','#475569','#65a30d']


def save(fig,name):
    # Write complete byte streams atomically so preview/indexing readers never
    # encounter a partially written scientific figure.
    import io,cairosvg
    svg_buffer=io.BytesIO();fig.savefig(svg_buffer,format='svg',bbox_inches='tight')
    svg_data=svg_buffer.getvalue()
    pdf_data=cairosvg.svg2pdf(bytestring=svg_data)
    assert len(pdf_data)>1000 and pdf_data.rstrip().endswith(b'%%EOF')
    png_buffer=io.BytesIO();fig.savefig(png_buffer,format='png',bbox_inches='tight')
    for suffix,data in [('svg',svg_data),('pdf',pdf_data),('png',png_buffer.getvalue())]:
        path=RESULTS/'figures'/f'{name}.{suffix}';temp=path.with_suffix(path.suffix+'.tmp')
        temp.write_bytes(data);temp.replace(path)
    plt.close(fig)


def read(name,folder='processed'):
    path=RESULTS/folder/f'{name}.json'
    if not path.exists():path=RESULTS/folder/f'plot_bundle_{name}.json'
    return json.loads(path.read_text())


def main():
    d=read('phase1_plot_data','processed');r=d['results'];env=d['envelope'];w=np.array(env['w']);cost=np.array(env['costs'])
    fig,ax=plt.subplots(1,2,figsize=(9,3.2),layout='constrained')
    for i,name in enumerate(['A','K','B']):ax[0].plot(w,cost[:,i],label=name,color=COLORS[i])
    ax[0].axvspan(.4,.6,color=COLORS[1],alpha=.12);ax[0].set(xlabel='$w_1$',ylabel='Scalarized cost',title='A: stable cells include anchors');ax[0].legend(ncol=3)
    y=np.array(env['objectives']);ax[1].plot(y[:,0],y[:,1],'o--',color='#9ca3af',markersize=7)
    for point,label in zip(y,['A: TPE tie-break','K: TAFR','B']):ax[1].annotate(label,point,xytext=(5,8),textcoords='offset points',fontsize=9)
    ax[1].set(xlabel='$f_1$',ylabel='$f_2$',xlim=(-1,12),ylim=(-1,12),title='Interior compromise survives screening');save(fig,'case_A')
    fig,ax=plt.subplots(1,2,figsize=(9,3.2),layout='constrained');scales=[];err=[];weights=[];rawerr=[]
    for row in r['E0-B']:
        scales.append(np.log10(row['scale']));s=row['normalized']['selected'];err.append(np.linalg.norm(np.array(s['normalized_objectives'])-[.4,.4]));weights.append(s['weight'][0]);t=row['no_normalization']['selected'];rawerr.append(np.nan if t is None else np.linalg.norm(np.array(t['objectives'])/np.array([10,10*row['scale']])-[.4,.4]))
    ax[0].plot(scales,err,'o-',label='TAFR frozen normalization',color=COLORS[0]);ax[0].plot(scales,rawerr,'s--',label='No normalization',color=COLORS[1])
    for x,y0 in zip(scales,rawerr):
        if np.isnan(y0):ax[0].text(x,.05,'abstain',ha='center',rotation=40,fontsize=8,color=COLORS[1])
    ax[0].set(xlabel=r'$\log_{10} c$',ylabel='Selected normalized error',ylim=(-.01,.12),title='B: normalized selection is invariant');ax[0].legend(fontsize=8)
    ax[1].plot(scales,weights,'o-',label='Selected normalized $w_1$',color=COLORS[0]);interval=np.array([x['raw_knee_interval'] for x in r['E0-B']]);ax[1].fill_between(scales,interval[:,0],interval[:,1],color=COLORS[1],alpha=.3,label='Raw-weight knee cell');ax[1].set(xlabel=r'$\log_{10} c$',ylabel='Weight',title='Raw preference cell moves to the edge');ax[1].legend(fontsize=8);save(fig,'case_B')
    fig,ax=plt.subplots(1,3,figsize=(10,3.6),layout='constrained')
    for a,key,title in zip(ax[:2],['pairwise','vertices'],['Pairwise transfers','Complete vertices']):
        v=np.array(r['E0-C'][key]['weights'])-.25;mask=np.max(abs(v),axis=1)>1e-8;v=v[mask]
        a.imshow(v,cmap='coolwarm',vmin=-.1,vmax=.1,aspect='auto');a.set(xticks=range(4),xticklabels=['1','2','3','4'],xlabel='Objective weight',ylabel='Perturbation',title=title)
    vals=[r['E0-C']['pairwise']['R'],r['E0-C']['vertices']['R'],r['E0-C']['exact']['robustness']]
    ax[2].bar(['Pairwise','Vertices','Exact LP'],vals,color=COLORS[:3]);ax[2].set(ylabel='$R_{0.1}$',ylim=(0,.26),title='C: omitted group direction')
    for i,v in enumerate(vals):ax[2].text(i,v+.008,f'{v:.4f}',ha='center');
    save(fig,'case_C')
    cell=d['cell_map'];ww=np.array(cell['weights']);xx=ww[:,1]+.5*ww[:,2];yy=np.sqrt(3)/2*ww[:,2]
    fig,ax=plt.subplots(figsize=(6,5),layout='constrained');sc=ax.scatter(xx,yy,c=cell['selected'],s=4,cmap=ListedColormap(COLORS),vmin=-.5,vmax=6.5,rasterized=True)
    for i,label in enumerate(['$w_1=1$','$w_2=1$','$w_3=1$']):
        pt=[(0,0),(1,0),(.5,np.sqrt(3)/2)][i];ax.annotate(label,pt,xytext=(0,-15 if i<2 else 8),textcoords='offset points',ha='center')
    from itertools import permutations
    vertices=np.unique(np.array(list(permutations([1/3-.18,1/3,1/3+.18]))),axis=0)
    vx=vertices[:,1]+.5*vertices[:,2];vy=np.sqrt(3)/2*vertices[:,2];ang=np.argsort(np.arctan2(vy-vy.mean(),vx-vx.mean()));ax.plot(np.r_[vx[ang],vx[ang[0]]],np.r_[vy[ang],vy[ang[0]]],color='black',lw=1);ax.scatter(vx,vy,c='white',edgecolors='black',s=55,zorder=3)
    ax.scatter([.26+.5*.26],[np.sqrt(3)/2*.26],marker='*',s=140,c='black',label='Interior Q witness');ax.legend(loc='upper right',fontsize=8);ax.set(aspect='equal',title='D: Q has a cell missed by every vertex');ax.axis('off');cb=fig.colorbar(sc,ax=ax,ticks=range(7),shrink=.7);cb.ax.set_yticklabels(['A1','A2','A3','P','Q','R1','R2']);save(fig,'case_D_cells')
    summary=pd.read_csv(RESULTS/'tables/interior_detection.csv');plain=summary[summary.adaptive_rounds==0]
    fig,ax=plt.subplots(1,2,figsize=(9,3.3),layout='constrained');cis=np.array([json.loads(s) for s in plain.wilson95])
    ax[0].errorbar(plain.samples,plain.detection_probability,yerr=np.maximum(0,[plain.detection_probability-cis[:,0],cis[:,1]-plain.detection_probability]),fmt='o-',capsize=3,color=COLORS[0]);ax[0].set(xlabel='Interior samples',ylabel='Detection probability',ylim=(-.04,1.04),title='D: 100 seeds, Wilson 95% interval')
    ax[1].plot(plain.mean_calls,plain.mean_gap,'o-',label='Hybrid',color=COLORS[0]);ad=summary[(summary.samples==16)];ax[1].plot(ad.mean_calls,ad.mean_gap,'s--',label='Adaptive, 16 initial samples',color=COLORS[1]);ax[1].set(xlabel='Mean solver calls',ylabel='Mean exact audit gap',title='Audit accuracy has a sampling cost');ax[1].legend(fontsize=8);save(fig,'case_D_detection')
    flat=pd.read_csv(RESULTS/'processed/flat_gain.csv');fig,ax=plt.subplots(1,2,figsize=(9,3.2),layout='constrained');base=flat[flat.threshold==1e-6].sort_values('epsilon')
    ax[0].loglog(base.epsilon,base.deterioration/base.improvement,'o-',color=COLORS[1]);ax[0].set(xlabel='Absolute improvement $I$',ylabel='$D/I$',title='E: ratio diverges as gain vanishes')
    pv=flat.pivot(index='threshold',columns='epsilon',values='active').sort_index(ascending=False);ax[1].imshow(pv.astype(float),vmin=0,vmax=1,cmap=ListedColormap(['#e5e7eb',COLORS[0]]),aspect='auto');ax[1].set(xticks=range(len(pv.columns)),xticklabels=[f'{v:.0e}' for v in pv.columns],yticks=range(len(pv.index)),yticklabels=[f'{v:g}' for v in pv.index],xlabel='Improvement $I$',ylabel='Minimum gain threshold',title='Green = passes activity screen');ax[1].tick_params(axis='x',labelsize=8);save(fig,'case_E')
    rows=pd.DataFrame(read('all_rows','processed'));p=rows[rows.bundle=='pmop_pilot_table_range'].copy();p['instance']=p.problem_name+' / '+p.n_objectives.astype(str)
    order=[f'PMOP{i} / {m}' for m in (3,5,8) for i in range(1,15)];methods=['TAFR-meaningful-gain','TPE-2024-grid','equal','CHIM-posthoc'];pv=p.pivot(index='instance',columns='method',values='knee_error').reindex(index=order,columns=methods).astype(float)
    fig,ax=plt.subplots(figsize=(7,10),layout='constrained');cm=plt.get_cmap('viridis').copy();cm.set_bad('#d1d5db');im=ax.imshow(pv,cmap=cm,vmin=0,vmax=min(1,float(np.nanmax(pv))),aspect='auto');ax.set(yticks=range(len(order)),yticklabels=order,xticks=range(4),xticklabels=['TAFR gain','TPE grid','Equal','CHIM'],title='PMOP pilot: nearest-knee error\nFixed finite-table ranges; gray = abstention or undefined CHIM');ax.tick_params(axis='y',labelsize=8);fig.colorbar(im,ax=ax,shrink=.5,label='Normalized Euclidean distance');save(fig,'pmop_heatmap')
    fig,ax=plt.subplots(figsize=(5.3,4.2),layout='constrained');valid=rows[(rows.R_reported.notna())&(rows.R_validation.notna())]
    for exact,color,label in [(True,COLORS[0],'Finite-table LP'),(False,COLORS[1],'Independent samples')]:
        g=valid[valid.validation_type.str.startswith('exact')==exact];ax.scatter(g.R_reported,g.R_validation,s=35,alpha=.7,c=color,label=label)
    lim=max(valid.R_reported.max(),valid.R_validation.max())*1.08;ax.plot([0,lim],[0,lim],'--',c='#64748b');ax.set(xlabel='Reported $R$',ylabel='Validation $R$',title='Reported bounds require independent checks');ax.legend(fontsize=8);save(fig,'audit_validity')
    scale=read('scalability_pilot')['results'];ct=pd.DataFrame(scale['counts']);cov=pd.DataFrame(scale['coverage']);fig,ax=plt.subplots(1,2,figsize=(9,3.4),layout='constrained')
    ax[0].semilogy(ct.m,ct.vertices,'o-',label='Complete vertices');ax[0].semilogy(ct.m,ct.lattice_H20,'s-',label='H=20 candidate lattice');ax[0].set(xlabel='Objectives $m$',ylabel='Count',title='Exact enumeration grows combinatorially');ax[0].legend(fontsize=8)
    for i,(m,g) in enumerate(cov.groupby('m')):
        med=g.groupby('budget').nearest_analytic_knee_error.median();ax[1].loglog(med.index,med.values,'o-',label=f'm={m}',lw=1)
    ax[1].set(xlabel='Sobol candidate budget',ylabel='Median closest-point error',title='Coverage pilot, 30 seeds');ax[1].legend(ncol=2,fontsize=8);save(fig,'scalability')
    multi_path=RESULTS/'processed/snee_multistart_v2.csv'
    if multi_path.exists():
        ms=pd.read_csv(multi_path);fig,axs=plt.subplots(3,3,figsize=(10,8),layout='constrained')
        for ax,name in zip(axs.flat,sorted(ms.problem.unique())):
            ss=ms[ms.problem==name];data=[ss[(ss.variant==v)&(ss.status=='completed')].mcf.dropna().to_numpy() for v in ['scipy_shape_compatibility','normalized']]
            if all(len(x) for x in data):ax.boxplot(data,tick_labels=['Published scale','Normalized'],showfliers=True)
            ax.set(title=name,ylabel='MCF');ax.tick_params(axis='x',labelsize=8)
        save(fig,'snee_multistart')
    for bname in ['ammonia_pilot','ammonia_pilot_global_range']:
        path=RESULTS/'raw'/f'{bname}.json'
        if not path.exists():continue
        ar=read(bname)['results'];fig,ax=plt.subplots(len(ar),2,figsize=(10,3.1*len(ar)),squeeze=False,layout='constrained')
        for j,record in enumerate(ar):
            if record['status']!='completed':continue
            for i,row in enumerate(record['rows']):
                if row['selected']:ax[j,0].plot(range(4),row['normalized_objectives'],'o-',label=row['method'],color=COLORS[i])
            ax[j,0].set(xticks=range(4),xticklabels=['Cost','Emissions','Water','Safety'],ylabel='Frozen normalized objectives',title=record['problem']);ax[j,0].legend(fontsize=7)
            for i,s in enumerate(record['schedules']):
                if 'schedule' in s:ax[j,1].step(range(1,49),s['schedule']['NH3'],where='mid',label=s['method'],color=COLORS[i])
            ax[j,1].set(xlabel='Hour',ylabel='NH3 production (kg/h)',title='Selected schedules, forecasts fixed');ax[j,1].legend(fontsize=7)
        save(fig,bname)
    print('figures generated',len(list((RESULTS/'figures').glob('*.pdf'))))


if __name__=='__main__':main()
