from pathlib import Path
from collections import defaultdict
import sys,re,statistics,csv
from analyze import read_events,union,subtract,length,writecsv,summary,role,cpu_role
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=R/'analysis'
def perf(text):return [(int(t),int(c)) for t,c in re.findall(r'K200PERF trial=(\d+) pp_ms=[\d.]+ pp_tokens=(\d+)',text)]
cal=defaultdict(list)
for path in (R/'logs').glob('lightcal-*.txt'):
 m=re.match(r'lightcal-(tiny|7b)-([ab][12])',path.stem);model,tag=m.groups()
 for t,pp,c,tg,n in re.findall(r'K200PERF trial=(\d+) pp_ms=([\d.]+) pp_tokens=(\d+) tg_ms=([\d.]+) tg_steps=(\d+)',path.read_text(errors='replace')):
  cal[(model,tag[0],'pp')].append(float(pp)/int(c));cal[(model,tag[0],'tg')].append(float(tg)/int(n))
calrows=[]
for (m,o,p),v in sorted(cal.items()):
 row=dict(model=m,observer='light' if o=='b' else 'none',phase=p,**{k+'_ms_per_token':x for k,x in summary(v).items()})
 if o=='b':row['mean_overhead_percent']=100*(statistics.mean(v)/statistics.mean(cal[(m,'a',p)])-1)
 calrows.append(row)
writecsv(A/'light-calibration.csv',calrows)
out=[]
for model in ['tiny','7b']:
 path=R/f'lightrep-{model}.events.csv';events=read_events(path);roots=sorted([r for r in events if r['category']=='harness' and r['name']=='process'],key=lambda r:r['start_ns'])
 periods=[];last=None
 for r in roots:
  if r['phase']=='pp' and last!='pp':periods.append(dict(pp=[],tg=[]))
  periods[-1][r['phase']].append(r);last=r['phase']
 measured=perf((R/'logs'/f'lightrep-{model}.txt').read_text(errors='replace'));assert len(periods)==len(measured)==6
 for i,(period,(trial,ctx)) in enumerate(zip(periods,measured)):
  lo=period['pp'][0]['start_ns'];hi=periods[i+1]['pp'][0]['start_ns']-1 if i+1<len(periods) else max(r['end_ns'] for r in events)
  selected=[r for r in events if lo<=r['start_ns'] and r['end_ns']<=hi]
  for phase,prs in period.items():
   rs=[r for r in selected if r['phase']==phase];rootspans=[(r['start_ns'],r['end_ns']) for r in prs];total=length(rootspans);units=ctx if phase=='pp' else len(prs)
   c=dict(name=f'lightrep-{model}',model=model,context=ctx,trial=trial,phase=phase,tokens=units)
   candidates=defaultdict(list)
   for r in rs:
    if r['category']=='xpu.node':candidates[role(r['name'],r['tensor'])].append((r['start_ns'],r['end_ns']))
   cnodes=[r for r in rs if r['category']=='cpu.node']
   for r in rs:
    if r['category']=='backend.graph' and r['name']=='CPU':candidates[cpu_role([n for n in cnodes if r['start_ns']<=n['start_ns'] and n['end_ns']<=r['end_ns']])].append((r['start_ns'],r['end_ns']))
   for r in rs:
    if r['category'].startswith('tensor.'):k='External tensor access/copy'
    elif r['category']=='backend.sync':k='External backend synchronization'
    elif r['category']=='scheduler' and r['name']!='compute_splits':k='Scheduler split/allocation'
    else:continue
    candidates[k].append((r['start_ns'],r['end_ns']))
   covered=[];used=0
   for k,spans in candidates.items():
    clipped=subtract(subtract(spans,subtract(spans,rootspans)),covered);ns=length(clipped);covered=union(covered+clipped);used+=ns
    out.append(dict(**c,category=k,ms=ns/1e6,ms_per_token=ns/1e6/units,percent=ns/total*100,calls=len(spans)))
   assert used<=total
   for k,ns in [('Remaining host/scheduler/measurement',total-used),('TOTAL',total)]:out.append(dict(**c,category=k,ms=ns/1e6,ms_per_token=ns/1e6/units,percent=ns/total*100,calls=len(prs)))
writecsv(A/'light-wall-partition.csv',out)
amd=[]
for model in ['tiny','7b']:
 for ctx in [128,2048]:
  for phase in ['pp','tg']:
   rs=[r for r in out if r['model']==model and r['context']==ctx and r['phase']==phase];total=statistics.mean(r['ms_per_token'] for r in rs if r['category']=='TOTAL')
   for cat in sorted({r['category'] for r in rs}-{'TOTAL'}):
    v=statistics.mean(r['ms_per_token'] for r in rs if r['category']==cat);f=v/total
    amd.append(dict(model=model,context=ctx,phase=phase,candidate=cat,fraction=f,ms_per_token=v,total_ms_per_token=total,hypothetical_2x_speedup=1/(1-f/2),zero_cost_upper_speedup=1/(1-f)))
writecsv(A/'light-amdahl.csv',amd)
print('LIGHT_ANALYSIS_COMPLETE')
