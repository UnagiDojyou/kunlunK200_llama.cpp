from pathlib import Path
import sys,csv,statistics,collections
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
sys.path.insert(0,str(R/'plot-deps'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
with (R/'analysis/light-wall-partition.csv').open(newline='',encoding='utf-8') as f:rs=list(csv.DictReader(f))
def family(k):
 if k=='Q4 GEMV/GEMM':return 'Q4'
 if k=='FlashAttention XPU':return 'Flash'
 if k=='SwiGLU XPU':return 'SwiGLU'
 if k=='KV SET_ROWS XPU':return 'KV update'
 if k=='RoPE XPU':return 'RoPE'
 if k in ['RMSNorm XPU','Elementwise/residual XPU']:return 'Norm / elementwise'
 if k=='Q6_K output CPU graph':return 'CPU Q6_K head'
 if 'CPU graph' in k or k=='CPU metadata graph':return 'Other CPU'
 if k.startswith('External'):return 'External tensor / sync'
 return 'Host / metadata / observer'
names=['Q4','Flash','SwiGLU','KV update','RoPE','Norm / elementwise','CPU Q6_K head','Other CPU','External tensor / sync','Host / metadata / observer'];colors=['#4477AA','#66CCEE','#228833','#999933','#AA3377','#DDCC77','#CC6677','#882255','#BBBBBB','#555555']
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
for i,m in enumerate(['tiny','7b']):
 for j,p in enumerate(['pp','tg']):
  ax=axes[i,j];bottom=np.zeros(2)
  for name,col in zip(names,colors):
   vals=[]
   for ctx in [128,2048]:
    v=[r for r in rs if r['model']==m and r['context']==str(ctx) and r['phase']==p and r['category']!='TOTAL' and family(r['category'])==name]
    vals.append(sum(float(r['ms_per_token']) for r in v)/3)
   ax.bar([0,1],vals,bottom=bottom,color=col,label=name);bottom+=vals
  total=[];std=[]
  for ctx in [128,2048]:
   v=[float(r['ms_per_token']) for r in rs if r['model']==m and r['context']==str(ctx) and r['phase']==p and r['category']=='TOTAL'];assert len(v)==3;total.append(statistics.mean(v));std.append(statistics.stdev(v))
  assert np.allclose(bottom,total)
  ax.errorbar([0,1],total,yerr=std,fmt='none',color='black',capsize=4)
  ax.set_xticks([0,1],['prefix 128','prefix 2048']);ax.set_ylabel('ms / prompt token' if p=='pp' else 'ms / decode');ax.set_title(('TinyLlama 1.1B' if m=='tiny' else 'Llama 2 7B')+' / '+p.upper());ax.grid(axis='y',alpha=.15);ax.set_axisbelow(True)
h,l=axes[0,0].get_legend_handles_labels();fig.legend(h,l,loc='outside lower center',ncol=5,fontsize=8)
fig.suptitle('Light observer: exclusive wall time, no added device waits\nSingle PD0, batch/ubatch 128/128, SwiGLU 1, 3 trials; overhead calibrated separately')
for ext in ['png','svg']:fig.savefig(R/'figures'/('light-exclusive-wall.'+ext),dpi=180,bbox_inches='tight')
plt.close(fig);print('LIGHT_FIGURES_COMPLETE')
