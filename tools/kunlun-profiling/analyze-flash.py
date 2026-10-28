"""Existing intrusive Flash probes: do not mix with optimized normal timing."""
from pathlib import Path
from collections import defaultdict
import re,csv,statistics,json,sys
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=R/'analysis'
def write(n,rs):
 if not rs:return
 with (A/n).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
stage=[];work=[];timeline=[]
for p in (R/'logs').glob('flash-clock-*.txt'):
 model=p.stem.split('-')[-1];s=p.read_text(errors='replace')
 for name,ms in re.findall(r'(?:FLASH|DECODE)_STAGE stage=(\S+) ms=([\d.]+)',s):stage.append(dict(model=model,stage=name,ms=float(ms)))
 for line in s.splitlines():
  if not line.startswith('FLASH_WORK '):continue
  d=dict(re.findall(r'(\w+)=([^ ]+)',line));k=d['stage'];base=dict(model=model,stage=k,kind=d['kind'],cluster=int(d['cluster']))
  if d['kind']=='sd':
   t=[int(d['t'+str(i)]) for i in range(16)]
   if not t[9]:continue
   for i,label in enumerate(['Q_or_P_DMA','Q_or_P_DS','K_or_V_DMA','K_or_V_DS','MAC','EW','RS','output_DMA','remainder','total']):
    work.append(dict(**base,component=label,ticks=t[i],own_kernel_percent=100*t[i]/t[9]))
  else:
   work.append(dict(**base,core=int(d['core']),component='total',ticks=int(d['ticks']),rows=int(d['rows']),elements=int(d['elems'])))
p=R/'logs/flash-pipeline-clock-7b.txt'
if p.exists():
 call=-1
 for line in p.read_text(errors='replace').splitlines():
  if not line.startswith('FLASH_TIMELINE '):continue
  d={k:int(v) for k,v in re.findall(r'(\w+)=(\d+)',line)}
  if d['slot']==0 and d['worker']==0:call+=1
  assert d['ticks']==(d['end']-d['begin'])%(2**32)
  timeline.append(dict(call=call,**d))
write('flash-intrusive-stages.csv',stage);write('flash-intrusive-work.csv',work);write('flash-pipeline-timeline.csv',timeline)
out=[]
for key in sorted({(r['model'],r['stage']) for r in stage}):
 vals=[r['ms'] for r in stage if (r['model'],r['stage'])==key]
 out.append(dict(model=key[0],stage=key[1],calls=len(vals),mean_ms=statistics.mean(vals),median_ms=statistics.median(vals),std_ms=statistics.stdev(vals) if len(vals)>1 else 0))
write('flash-intrusive-summary.csv',out)
print(json.dumps(out,indent=2));print('FLASH_ANALYSIS_COMPLETE')
