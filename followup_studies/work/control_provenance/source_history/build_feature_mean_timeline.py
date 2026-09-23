"""Descriptive full-grid feature-mean timeline from archived Fourier summaries."""
import os,csv,hashlib,json,math,re
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter
B=Path(__file__).resolve().parent;R=Path(os.environ.get('MUON_SOURCE_REPO',str(B.parent/'muon-grokking')))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def rows(p):return list(csv.DictReader(p.open()))
def dumpcsv(p,a):
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(a[0]));w.writeheader();w.writerows(a)
p=R/'runs/seedstudy_fourier_mode_layer_summary.csv';raw=rows(p);data=[];sources={str(p.relative_to(R)):sha(p)}
for line,r in enumerate(raw,2):
 if r['stage_name']!='layer_1_post_block_residual' or r['regime']=='stable_muon':continue
 total=float(r['total_spectral_power']);frac=float(r['dc_power_fraction_total']);dc=total*frac
 data.append(dict(run=r['run'],seed=int(re.match(r'seed(\d+)',r['run'])[1]),regime=r['regime'],step=int(r['step']),readout_input='layer_1_post_block_residual',population='full operand grid, uniform113x113',feature_mean_norm=math.sqrt(dc)/113,dc_spectral_power=dc,total_spectral_power=total,dc_power_fraction=frac,checkpoint=r['checkpoint'],source_csv=str(p.relative_to(R)),source_row=line,training_backend='CPU reported in repository README; runtime manifest unavailable',measurement_backend='CPU, float32 orthonormal2D FFT per source'))
events=[]
for regime,pref in [('muon','muon'),('adamw','adamw_stable')]:
 for seed in range(5):
  path=R/'runs'/f'seedstudy_{pref}_seed_{seed}.csv';a=rows(path);sources[str(path.relative_to(R))]=sha(path)
  g=next(int(a[i]['step']) for i in range(len(a)-4) if all(float(x['test_accuracy'])>=.95 for x in a[i:i+5]))
  ev=[dict(run=f'seed{seed}_{pref}',seed=seed,regime=regime,step=g,event='original_grokking_first_of_five',train_accuracy='',test_accuracy='',source_csv=str(path.relative_to(R)),sampling_interval=100)]
  for r in a:
   if int(r['step'])>g and float(r['train_accuracy'])<.9 and float(r['test_accuracy'])<.9:
    ev.append(dict(run=f'seed{seed}_{pref}',seed=seed,regime=regime,step=int(r['step']),event='sampled_joint_failure',train_accuracy=r['train_accuracy'],test_accuracy=r['test_accuracy'],source_csv=str(path.relative_to(R)),sampling_interval=100))
  events+=ev
dumpcsv(B/'feature_mean_timeline.csv',data);dumpcsv(B/'feature_mean_events.csv',events)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8.2,'axes.titlesize':9,'axes.labelsize':8.5,'xtick.labelsize':8,'ytick.labelsize':8,'pdf.fonttype':42})
fig=plt.figure(figsize=(6.5,4.2));gs=fig.add_gridspec(2,2,height_ratios=[2.5,1],hspace=.10,wspace=.22,left=.10,right=.985,top=.87,bottom=.18)
colors=['#0072B2','#D55E00','#009E73','#CC79A7','#E69F00']
for j,(regime,title) in enumerate([('muon','Muon hidden / AdamW interface'),('adamw','Selected AdamW baseline')]):
 ax=fig.add_subplot(gs[0,j]);evax=fig.add_subplot(gs[1,j],sharex=ax)
 for seed,col in enumerate(colors):
  a=sorted([x for x in data if x['regime']==regime and x['seed']==seed],key=lambda x:x['step'])
  ax.plot([r['step'] for r in a],[r['feature_mean_norm'] for r in a],color=col,marker='o',markersize=3,linewidth=.9,alpha=.95,label=f'Seed {seed}')
  for e in [e for e in events if e['regime']==regime and e['seed']==seed]:
   evax.scatter(e['step'],seed,color=col,marker='^' if e['event'].startswith('original') else 'x',s=24,linewidths=1,zorder=3)
 ax.set_title(title,pad=8);ax.set_yscale('log');ax.set_ylim(8,30000);ax.set_xlim(-2000,102000);ax.grid(axis='y',alpha=.18);ax.tick_params(axis='x',labelbottom=False)
 if j==0:ax.set_ylabel(r'Full-grid $\|\bar h\|_2$')
 ax.text(.98,.04,f'{sum(x["regime"]==regime for x in data)} sampled checkpoints',ha='right',va='bottom',transform=ax.transAxes,fontsize=7.5)
 evax.set_ylim(-.65,4.65);evax.set_yticks(range(5));evax.set_yticklabels([str(i) for i in range(5)]);evax.set_xlabel('Training step');evax.xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{int(x/1000)}k'));evax.set_xticks([0,25000,50000,75000,100000]);evax.grid(axis='x',alpha=.12)
 if j==0:evax.set_ylabel('Seed')
 for z in [ax,evax]:z.spines[['top','right']].set_visible(False)
fig.text(.5,.98,'Training backend: CPU (reported); spectral measurements: CPU',ha='center',va='top',fontsize=8)
handles=[Line2D([0],[0],marker='o',color=c,lw=.8,markersize=3,label=str(i)) for i,c in enumerate(colors)]+[Line2D([0],[0],marker='^',color='#333333',lw=0,markersize=5,label='Grokking'),Line2D([0],[0],marker='x',color='#333333',lw=0,markersize=5,label='Sampled joint failure')]
fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.52,.02),ncol=7,frameon=False,handlelength=1,columnspacing=1,fontsize=8)
fig.savefig(B/'feature_mean_timeline.pdf');fig.savefig(B/'feature_mean_timeline.png',dpi=180);plt.close(fig)
manifest={'status':'complete','definition':'Euclidean norm of arithmetic mean of actual pre-readout residual features over all12769 operand pairs. Gain-free unnormalized depth1 models. Derived from preserved float32 ortho-FFT summaries: mean_norm=sqrt(total_spectral_power*dc_power_fraction_total)/113. FourierDC coefficient=113*mean_feature.','population':'Full operand grid, including training and held-out examples. Descriptive measurement only.','sampling':'49 Muon and39 AdamW selected checkpoints, irregular sampling exactly shown by circles; joining segments aid visual tracking and do not measure intermediate states. Source checkpoint selection includes phase landmarks and low-accuracy checkpoints.','events':'Grokking uses first of5successive100-step scheduled test evaluations>=95%. Crosses show every recorded post-grokking joint train/test<90% evaluation in the100-step archived CSVs. Brief intervening events may be missed.','historical_event_counts':{'muon_evaluations':4,'muon_seeds':3,'adamw_evaluations':3,'adamw_seeds':3},'cohort_distinction':'Original archived100-step seed-study trajectories, separate from fresh densely captured5seed failures. Absence of an event marker does not establish event-free training between observations.','comparison_scope':'Selected configurations and five historical seed-study trajectories; backend CPU as reported by repository README, no independently archived runtime manifest. No inference of upstream causality or population optimizer comparison.','optimizer_configs':{'muon_hidden':'lr.03,wd.1,momentum.95,NS5','muon_embeddings':'AdamWlr.001wd1','muon_readout':'AdamWlr.00025wd1','adamw_all':'lr.001wd3'},'source_sha256':sources,'rows':len(data),'events':len(events),'files':{x.name:sha(x) for x in [B/'feature_mean_timeline.csv',B/'feature_mean_events.csv',B/'feature_mean_timeline.pdf',B/'feature_mean_timeline.png']}}
(B/'feature_mean_timeline_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
print('points',len(data),'jointfailuremarkers',sum(x['event']=='sampled_joint_failure' for x in events))
