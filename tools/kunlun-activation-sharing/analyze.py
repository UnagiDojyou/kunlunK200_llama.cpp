from pathlib import Path
import re,json,statistics as st,csv,collections,sys
def summary(values):
 v=list(values)
 return dict(n=len(v),mean=st.mean(v),median=st.median(v),stddev=st.stdev(v) if len(v)>1 else 0,min=min(v),max=max(v),samples=v)
def bench_rows(text):
 found=[]
 for match in re.finditer(r'\{\s*"build_commit"',text):
  try:value=json.JSONDecoder().raw_decode(text[match.start():])[0]
  except ValueError:continue
  if isinstance(value,dict) and 'samples_ns' in value:found.append(value)
 return found
def kv_values(text):
 return {k:v for k,v in re.findall(r'(\w+)=([^\s]+)',text)}
def scan(root=None):
 ROOT=Path(root) if root else Path(__file__).resolve().parent
 LOGS=ROOT/'logs';DEST=ROOT/'analysis'
 if not LOGS.is_dir():raise FileNotFoundError('Pass a results directory containing logs/')
 DEST.mkdir(exist_ok=True)
 standards=[];prefix=[];quant=[];transfers=[];memory=[];activation=[];safety=[];fallback=[];paths=[];weights=[];library=[];cpu_gold=[];api=[]
 for f in sorted(LOGS.glob('*.txt')):
  meta=f.with_suffix('.json')
  if not meta.exists():continue
  meta=json.loads(meta.read_text())
  if meta['returncode']!=0:raise RuntimeError(f'failed {f.stem}')
  name=f.stem;s=f.read_text(errors='replace')
  m=re.fullmatch(r'standard-(tiny|7b)-(pd0|pd1|layer|row)-u(\d+)-g(\d+)-a([01])',name)
  series='primary';round_=0
  if not m:
   final=re.fullmatch(r'standard-final-(tiny|7b)-(pd0|pd1|layer|row)-u(\d+)-g(\d+)-a([01])-r(\d+)',name)
   if final:m=final;series='final-abba';round_=int(final.group(6))
  if m:
   model,split,u,glu,share=m.groups()[:5];rows=bench_rows(s);assert len(rows)==2,(name,len(rows))
   for row in rows:
    length=row['n_prompt'] or row['n_gen'];ts=[length*1e9/x for x in row['samples_ns']]
    standards.append(dict(name=name,series=series,round=round_,model=model,split=split,ubatch=int(u),glu=int(glu),share=int(share),phase='pp' if row['n_prompt'] else 'tg',length=length,ts=summary(ts),ms=summary(x/1e6 for x in row['samples_ns']),raw=row))
  for trial,pp_ms,p,tg_ms,steps,first in re.findall(r'K200PERF trial=(\d+) pp_ms=([0-9.]+) pp_tokens=(\d+) tg_ms=([0-9.]+) tg_steps=(\d+) first_process_ms=([0-9.]+)',s):
   prefix.append(dict(name=name,trial=int(trial),pp_tokens=int(p),pp_ms=float(pp_ms),tg_steps=int(steps),tg_ms=float(tg_ms),pp_ts=int(p)*1000/float(pp_ms),tg_ts=int(steps)*1000/float(tg_ms) if int(steps) else None))
  for pd,body in re.findall(r'KUNLUN activation phase PD([01]): ([^\n]+)',s):
   v=kv_values(body);quant.append(dict(name=name,pd=int(pd),phase=v.pop('phase'),**{k:float(x) if k.endswith('_ms') else int(x) for k,x in v.items()}))
  for pd,body in re.findall(r'KUNLUN activation PD([01]): ([^\n]+)',s):
   v=kv_values(body);activation.append(dict(name=name,pd=int(pd),**{k:float(x) if k.endswith('_ms') else int(x) for k,x in v.items()}))
  for pd,body in re.findall(r'KUNLUN common phase PD([01]): ([^\n]+)',s):
   values=kv_values(body);paths.append(dict(name=name,pd=int(pd),phase=values.pop('phase'),**{k:int(v) for k,v in values.items()}))
  for pd,body in re.findall(r'KUNLUN common totals PD([01]): ([^\n]+)',s):
   values=kv_values(body);weights.append(dict(name=name,pd=int(pd),**{k:float(v) if k.endswith('_ms') else int(v) for k,v in values.items()}))
  for line in s.splitlines():
   if name.startswith('lease-v2') and line.startswith('ACTIVATION_RESULT '):library.append(dict(name=name,**json.loads(line.split(' ',1)[1])))
   if name.startswith('cpu-gold-') and line.startswith('COMMON_RESULT '):cpu_gold.append(dict(name=name,**json.loads(line.split(' ',1)[1])))
   if line.startswith('K200PROFILE\t'):
    _,phase,kind,count,ms,bytes_=line.split('\t')
    if kind.startswith('q4.'):api.append(dict(name=name,phase=phase,kind=kind,calls=int(count),ms=float(ms),bytes=int(bytes_)))
    if kind.startswith('runtime.copy.') or kind.startswith('runtime.wait.') or kind.startswith('runtime.launch.'):
     transfers.append(dict(name=name,phase=phase,kind=kind,calls=int(count),ms=float(ms),bytes=int(bytes_)))
   if line.startswith(('K200HBM ','K200L3 ')):
    memory.append(dict(name=name,kind=line.split()[0],**{k:int(v) for k,v in kv_values(line).items()}))
   if line.startswith('GUARD END'):
    v=kv_values(line);safety.append(dict(name=name,peak_c=float(v['peak']),rc=int(v['rc'])))
   if 'native totals PD' in line:
    v=kv_values(line);fallback.append(dict(name=name,pd=int(re.search(r'PD([01])',line).group(1)),**{k:int(v) for k,v in v.items() if k in ['native_sd','native_cl','cdnn','direct_q4']}))
 groups=collections.defaultdict(list)
 for row in prefix:groups[(row['name'],row['pp_tokens'])].append(row)
 prefix_summary=[]
 for (name,p),rows in groups.items():
  prefix_summary.append(dict(name=name,pp_tokens=p,pp_ts=summary(x['pp_ts'] for x in rows),pp_ms=summary(x['pp_ms'] for x in rows),tg_ts=summary(x['tg_ts'] for x in rows if x['tg_ts'] is not None),tg_ms=summary(x['tg_ms'] for x in rows),tg_steps=rows[0]['tg_steps']))
 timeline=[]
 for f in ROOT.glob('*.events.csv'):
  grouped=collections.defaultdict(lambda:[0,0,0])
  with f.open() as stream:
   for row in csv.DictReader(stream):
    category=row['category']
    if category not in ['xpu.activation','xpu.node','cpu.node','scheduler.split','backend.sync']:continue
    if category=='cpu.node' and row['worker']!='0':continue
    key=(row['phase'],category,row['name'],row['tensor'] if category=='cpu.node' else '');v=grouped[key];v[0]+=1;v[1]+=int(row['end_ns'])-int(row['start_ns']);v[2]+=int(row['cpu_ns'])
  for (phase,category,op,tensor),(calls,ns,cpu) in grouped.items():timeline.append(dict(tensor=tensor,name=f.name.replace('.events.csv',''),phase=phase,category=category,op=op,calls=calls,ms=ns/1e6,cpu_ms=cpu/1e6))
 data=dict(standard=standards,prefix=prefix_summary,quant=quant,transfers=transfers,memory=memory,activation=activation,safety=safety,fallback=fallback,timeline=timeline,paths=paths,weights=weights,library=library,cpu_gold=cpu_gold,api=api)
 (DEST/'results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2));print({k:len(v) for k,v in data.items()})
 # Flat CSV includes all individual samples in the JSON above.
 with (DEST/'standard.csv').open('w',newline='') as stream:
  writer=csv.writer(stream);writer.writerow(['series','round','model','split','ubatch','glu','share','phase','tokens','mean_ts','median_ts','stddev_ts','min_ts','max_ts','mean_ms','median_ms','stddev_ms','n'])
  for x in standards:writer.writerow([x[k] for k in ['series','round','model','split','ubatch','glu','share','phase','length']]+[x['ts'][k] for k in ['mean','median','stddev','min','max']]+[x['ms'][k] for k in ['mean','median','stddev']]+[x['ts']['n']])
 return data
if __name__=='__main__':scan(sys.argv[1] if len(sys.argv)>1 else None)
