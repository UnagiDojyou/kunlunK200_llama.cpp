from pathlib import Path
from collections import defaultdict
import json,re,csv,statistics,sys,hashlib
from analyze import read_events,length,writecsv,summary
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
O=R/'analysis';O.mkdir(exist_ok=True)
def perf(text):
 return [dict(trial=int(a),pp_ms=float(b),context=int(c),tg_ms=float(d),tg_steps=int(e)) for a,b,c,d,e in re.findall(r'K200PERF trial=(\d+) pp_ms=([\d.]+) pp_tokens=(\d+) tg_ms=([\d.]+) tg_steps=(\d+)',text)]
def periods(events):
 roots=sorted([r for r in events if r['category']=='harness' and r['name']=='process'],key=lambda r:r['start_ns'])
 result=[];last=None
 for r in roots:
  if r['phase']=='pp' and last!='pp':result.append(dict(pp=[],tg=[]))
  result[-1][r['phase']].append(r);last=r['phase']
 out=[]
 for i,p in enumerate(result):
  lo=min(r['start_ns'] for r in p['pp']);hi=min(r['start_ns'] for r in result[i+1]['pp'])-1 if i+1<len(result) else max(r['end_ns'] for r in events)
  out.append((p,[r for r in events if lo<=r['start_ns'] and r['end_ns']<=hi]))
 return out

cal=defaultdict(list)
for p in (R/'logs').glob('cal-*.txt'):
 m=re.match(r'cal-(tiny|7b)-g([01])-([ab][12])',p.stem)
 if not m:continue
 model,glu,tag=m.groups()
 for row in perf(p.read_text(errors='replace')):
  for phase in ['pp','tg']:cal[(model,int(glu),tag[0],phase)].append(row[phase+'_ms']/(row['tg_steps'] if phase=='tg' else row['context']))
calrows=[]
for key,values in sorted(cal.items()):
 model,glu,observer,phase=key
 row=dict(model=model,swiglu=glu,observer='full' if observer=='b' else 'none',phase=phase,**{k+'_ms_per_token':v for k,v in summary(values).items()})
 if observer=='b':row['mean_overhead_percent']=100*(statistics.mean(values)/statistics.mean(cal[(model,glu,'a',phase)])-1)
 calrows.append(row)
writecsv(O/'calibration.csv',calrows)

head=[]
for p in R.glob('head-*.events.csv'):
 model=p.name.split('-')[1].split('.')[0];events=read_events(p);ps=periods(events);measurements=perf((R/'logs'/(p.name.removesuffix('.events.csv')+'.txt')).read_text(errors='replace'))
 assert len(ps)==len(measurements)
 for (roots,rows),measurement in zip(ps,measurements):
  for phase in ['pp','tg']:
   rs=[r for r in rows if r['phase']==phase];q6=[r for r in rs if r['category']=='cpu.node' and r['name']=='MUL_MAT' and r['tensor']=='result_output']
   q6graphs=[]
   for g in rs:
    if g['category']=='backend.graph' and g['name']=='CPU' and any(g['start_ns']<=r['start_ns'] and r['end_ns']<=g['end_ns'] for r in q6):q6graphs.append(g)
   pp=[r for r in rs if r['category']=='harness' and r['name']=='greedy_finite_dump_detokenize']
   getter=[r for r in rs if r['category']=='harness' and r['name']=='logits_get']
   calls=sum(r['worker']==0 for r in q6);assert calls==len(roots[phase])
   head.append(dict(model=model,context=measurement['context'],trial=measurement['trial'],phase=phase,q6_calls=calls,process_calls=len(roots[phase]),
      process_ms=length([(r['start_ns'],r['end_ns']) for r in roots[phase]])/1e6,
      q6_worker_union_ms=length([(r['start_ns'],r['end_ns']) for r in q6])/1e6,
      q6_worker_cpu_ms=sum(r['cpu_ns'] for r in q6)/1e6,q6_graph_ms=length([(r['start_ns'],r['end_ns']) for r in q6graphs])/1e6,
      logits_get_ms=sum(r['end_ns']-r['start_ns'] for r in getter)/1e6,postprocessing_calls=len(pp),postprocessing_ms=sum(r['end_ns']-r['start_ns'] for r in pp)/1e6))
writecsv(O/'head-low-overhead.csv',head)

def kernel_kind(s):
 if 'quant_' in s:return 'Q4 Activation quantization'
 if 'q40_gemm' in s:return 'Q4 GEMM MAC+block scales+DMA'
 if 'ds_dual16' in s:return 'Q4 GEMV MAC+DMA'
 if 'q4_sum_partial' in s:return 'Q4 GEMV FP32 scale/sum'
 if 'join_slices' in s:return 'Q4 GEMM slice join'
 if 'flash_' in s:
  if 'kq' in s:return 'Flash KQ+DMA'
  if 'pv' in s:return 'Flash PV+DMA'
  if 'online' in s or 'reduce' in s:return 'Flash Online/normalize/reduce'
  if 'range' in s:return 'Flash Q ranges'
 if 'attention_copy' in s:return 'Attention pack/copy'
 if 'SDK.unmapped' in s:return 'SDK cast/findmax (unmapped)'
 if 'swiglu' in s:return 'SwiGLU'
 if 'rms_' in s:return 'RMSNorm'
 if 'rope' in s:return 'RoPE'
 if 'set_rows' in s:return 'SET_ROWS'
 if 'copy_rows' in s:return 'Device copy_rows'
 if 'elementwise' in s:return 'Other elementwise'
 return 'Other kernel'
serial=[]
for p in R.glob('serial-*.events.csv'):
 m=re.match(r'serial-(tiny|7b)-g([01])',p.name);model,glu=m.groups();events=read_events(p)
 for context,(roots,rs) in zip([128,2048],periods(events)):
  for phase in ['pp','tg']:
   agg=defaultdict(list)
   launch=None
   for r in sorted(rs,key=lambda r:r['start_ns']):
    if r['phase']!=phase or r['category']!='audit':continue
    if r['name'].startswith('runtime.launch.'):launch=r
    elif r['name'].startswith('kernel.serial_wait.'):
     assert launch is not None and launch['end_ns']<=r['start_ns']
     agg[kernel_kind(r['name'])].append((launch,r));launch=None
   for kind,vals in agg.items():serial.append(dict(model=model,swiglu=int(glu),context=context,phase=phase,kind=kind,kernels=len(vals),ms=sum(r['end_ns']-l['start_ns'] for l,r in vals)/1e6,wait_only_ms=sum(r['end_ns']-r['start_ns'] for l,r in vals)/1e6,tokens=context if phase=='pp' else len(roots['tg'])))
writecsv(O/'serialized-kernels.csv',serial)

# Compare first nine greedy predictions along identical teacher-forced inputs.
greedy=[]
for p in (R/'logs').glob('diag-*.txt'):
 normal=p.with_name(p.name.replace('diag-','prefix-',1))
 if not normal.exists():continue
 ds=[r.split() for r in re.findall(r'TOKENS mode=\S+ ([^\n]+)',p.read_text(errors='replace'))]
 ns=[r.split() for r in re.findall(r'TOKENS mode=\S+ ([^\n]+)',normal.read_text(errors='replace'))]
 assert len(ds)==4 and len(ns)==12,(p.name,len(ds),len(ns))
 for i,ctx in enumerate([128,512,1024,2048]):
  matches=[d==ds[i] for d in [v[:len(ds[i])] for v in ns[i*3:i*3+3]]]
  greedy.append(dict(name=p.stem,context=ctx,positions=len(ds[i]),trials=3,all_matched=all(matches)))
writecsv(O/'matched-greedy.csv',greedy)
assert all(r['all_matched'] for r in greedy),'Observed greedy difference; inspect matched logits before concluding'
print('EXTRA_ANALYSIS_COMPLETE')
