"""Standalone figures from completed CSVs; never run on the benchmark host."""
from pathlib import Path
import csv,sys,collections,statistics
sys.path.insert(0,str(Path(__file__).resolve().parent/'plot-deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
analysis=root/'analysis';figures=root/'figures';figures.mkdir(exist_ok=True)
def read(name):
 with (analysis/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def save(fig,name):
 fig.savefig(figures/(name+'.png'),dpi=180,bbox_inches='tight')
 fig.savefig(figures/(name+'.svg'),bbox_inches='tight');plt.close(fig)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
normal=read('ordinary-summary.csv')
splits=['pd0','pd1','layer','row'];labels=['Single PD0','Single PD1','Layer','Row'];colors=['#4477AA','#228833']
fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
for i,model in enumerate(['tiny','7b']):
 for j,phase in enumerate(['pp','tg']):
  ax=axes[i,j]
  for glu in [0,1]:
   rows=[next(r for r in normal if r['kind']=='standard' and r['model']==model and r['split']==s and r['batch']=='128' and r['ubatch']=='128' and r['swiglu']==str(glu) and r['phase']==phase) for s in splits]
   x=np.arange(4)+(glu-.5)*.34;y=[float(r['tokens_s_mean']) for r in rows];sd=[float(r['tokens_s_std']) for r in rows]
   ax.bar(x,y,width=.32,yerr=sd,capsize=3,label=f'SwiGLU {glu}',color=colors[glu])
   ax.scatter(x,[float(r['tokens_s_median']) for r in rows],marker='D',color='black',s=13,zorder=4,label='Median' if glu==0 else None)
   for a,b in zip(x,y):ax.annotate(f'{b:.2f}',(a,b),xytext=(0,5),textcoords='offset points',ha='center',fontsize=8)
  ax.set_xticks(np.arange(4),labels);ax.set_ylim(bottom=0);ax.margins(y=.15);ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
  ax.set_title(('TinyLlama 1.1B' if model=='tiny' else 'Llama 2 7B')+' / '+('PP512' if phase=='pp' else 'TG128, empty KV'))
  ax.set_ylabel('tokens / second')
handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=3,fontsize=8)
fig.suptitle('0.7.6 normal llama-bench — batch/ubatch 128/128, 5 trials, mean ± sample SD')
save(fig,'normal-standard-128')

fig,axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
prefixes=[128,512,1024,2048];configurations=[(128,128),(512,128),(512,512)]
for i,model in enumerate(['tiny','7b']):
 for j,phase in enumerate(['pp','tg']):
  ax=axes[i,j]
  for color,(batch,ubatch) in zip(['#4477AA','#228833','#CC6677'],configurations):
   rows=[next(r for r in normal if r['kind']=='prefix' and r['model']==model and r['split']=='pd0' and r['batch']==str(batch) and r['ubatch']==str(ubatch) and r['swiglu']=='1' and r['context']==str(p) and r['phase']==phase) for p in prefixes]
   ax.errorbar(prefixes,[float(r['tokens_s_mean']) for r in rows],yerr=[float(r['tokens_s_std']) for r in rows],capsize=3,marker='o',color=color,label=f'batch/ubatch {batch}/{ubatch}')
  ax.set_xscale('log',base=2);ax.set_xticks(prefixes,[str(p) for p in prefixes]);ax.grid(alpha=.2)
  ax.set_xlabel('Prompt prefix tokens (context capacity 4096)');ax.set_ylabel('tokens / second')
  ax.set_title(('TinyLlama 1.1B' if model=='tiny' else 'Llama 2 7B')+' / '+('PP' if phase=='pp' else 'prefix TG, 31 decodes'))
axes[0,0].legend(fontsize=8)
fig.suptitle('0.7.6 normal prefix workload — Single PD0, SwiGLU 1, 3 trials, mean ± sample SD')
save(fig,'normal-prefix-configurations')

if not (analysis/'wall-partition.csv').exists():
 print('NORMAL_FIGURES_COMPLETE');raise SystemExit(0)
partition=read('wall-partition.csv')
def family(category):
 if category=='Q4 GEMV/GEMM':return 'Q4'
 if category=='FlashAttention XPU':return 'Flash'
 if category=='SwiGLU XPU':return 'SwiGLU'
 if category=='KV SET_ROWS XPU':return 'KV update'
 if category=='RoPE XPU':return 'RoPE'
 if category in ['RMSNorm XPU','Elementwise/residual XPU']:return 'Norm / elementwise'
 if category=='Q6_K output CPU graph':return 'CPU Q6_K head'
 if 'CPU graph' in category or category=='CPU metadata graph':return 'Other CPU'
 if category.startswith('External'):return 'External copy / wait'
 return 'Host / metadata / observer'
families=['Q4','Flash','SwiGLU','KV update','RoPE','Norm / elementwise','CPU Q6_K head','Other CPU','External copy / wait','Host / metadata / observer']
palette=['#4477AA','#66CCEE','#228833','#999933','#AA3377','#DDCC77','#CC6677','#882255','#BBBBBB','#555555']
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for i,model in enumerate(['tiny','7b']):
 for j,phase in enumerate(['pp','tg']):
  ax=axes[i,j];bottom=np.zeros(2)
  for f,color in zip(families,palette):
   means=[]
   for ctx in [128,2048]:
    rows=[r for r in partition if r['name']==f'diagrep-{model}-pd0-u128-g1' and r['phase']==phase and r['context']==str(ctx) and r['category']!='TOTAL']
    per_trial=collections.defaultdict(float)
    for r in rows:
     if family(r['category'])==f:per_trial[r['trial']]+=float(r['ms_per_token'])
    means.append(sum(per_trial.values())/3)
   ax.bar([0,1],means,bottom=bottom,color=color,label=f);bottom+=means
  totals=[];sds=[]
  for ctx in [128,2048]:
   v=[float(r['ms_per_token']) for r in partition if r['name']==f'diagrep-{model}-pd0-u128-g1' and r['phase']==phase and r['context']==str(ctx) and r['category']=='TOTAL']
   assert len(v)==3;totals.append(statistics.mean(v));sds.append(statistics.stdev(v))
  assert np.allclose(bottom,totals)
  ax.errorbar([0,1],totals,yerr=sds,fmt='none',color='black',capsize=4)
  ax.set_xticks([0,1],['prefix 128','prefix 2048']);ax.set_ylabel('ms / prompt token' if phase=='pp' else 'ms / decode');ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
  ax.set_title(('TinyLlama 1.1B' if model=='tiny' else 'Llama 2 7B')+' / '+phase.upper())
handles,legend=axes[0,0].get_legend_handles_labels();fig.legend(handles,legend,loc='outside lower center',ncol=5,fontsize=8)
fig.suptitle('Diagnostic exclusive wall time — no added XPU waits; observer overhead is separate\nSingle PD0, batch/ubatch 128/128, SwiGLU 1, 3 trials')
save(fig,'diagnostic-exclusive-wall')
print('FIGURES_COMPLETE')
