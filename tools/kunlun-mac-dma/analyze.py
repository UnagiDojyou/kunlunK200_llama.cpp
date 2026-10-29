from pathlib import Path
import sys,json,re,csv,collections
from analysis_helpers import summary,parse_standard,bench_rows,values

def analyze(root):
 O=Path(root);A=O/'analysis';A.mkdir(exist_ok=True)
 standard=[];prefix=[];transfers=[];api=[];kernelwaits=[];quant=[];memory=[];paths=[];macpaths=[];workspace=[];weights=[];safety=[];frozen=[];timeline=[];fallback=[];activation=[]
 for f in sorted((O/'logs').glob('*.txt')):
  name=f.stem;meta=f.with_suffix('.json')
  if not meta.exists():continue
  meta=json.loads(meta.read_text())
  if meta['returncode']!=0:raise RuntimeError('failed run: '+name)
  s=f.read_text(errors='replace');standard.extend(parse_standard(name,s))
  match=re.fullmatch(r'prefix-(tiny|7b)-(pd0|pd1|layer|row)-u(\d+)-m([01])-i(\d+)',name)
  for trial,pp_ms,length,tg_ms,steps,cold in re.findall(r'K200PERF trial=(\d+) pp_ms=([\d.]+) pp_tokens=(\d+) tg_ms=([\d.]+) tg_steps=(\d+) first_process_ms=([\d.]+)',s):
   row=dict(name=name,trial=int(trial),pp_tokens=int(length),pp_ms=float(pp_ms),tg_ms=float(tg_ms),tg_steps=int(steps),cold_ms=float(cold),pp_ts=int(length)*1000/float(pp_ms),tg_ts=int(steps)*1000/float(tg_ms) if int(steps) else 0)
   if match:
    model,split,u,mac,order=match.groups();row.update(model=model,split=split,ubatch=int(u),mac=int(mac),order=int(order),share=1)
   prefix.append(row)
  match=re.fullmatch(r'frozen-control-(tiny|7b)-(pd0|pd1|layer|row)-s([01])-i(\d+)',name)
  if match:
   model,split,stable,order=match.groups()
   for b in bench_rows(s):
    length=b['n_prompt'] or b['n_gen'];times=[v/1e6 for v in b['samples_ns']]
    frozen.append(dict(name=name,model=model,split=split,stable=int(stable),order=int(order),phase='pp' if b['n_prompt'] else 'tg',ms=summary(times),ts=summary(length*1000/t for t in times)))
  for pd,body in re.findall(r'KUNLUN activation phase PD([01]): ([^\n]+)',s):
   v=values(body);quant.append(dict(name=name,pd=int(pd),phase=v.pop('phase'),**{k:float(x) if k.endswith('_ms') else int(x) for k,x in v.items()}))
  for pd,body in re.findall(r'KUNLUN common phase PD([01]): ([^\n]+)',s):
   v=values(body);paths.append(dict(name=name,pd=int(pd),phase=v.pop('phase'),**{k:int(x) for k,x in v.items()}))
  for pd,body in re.findall(r'KUNLUN common MAC phase PD([01]): ([^\n]+)',s):
   v=values(body);macpaths.append(dict(name=name,pd=int(pd),phase=v.pop('phase'),**{k:int(x) for k,x in v.items()}))
  for pd,body in re.findall(r'KUNLUN common MAC PD([01]): ([^\n]+)',s):
   workspace.append(dict(name=name,pd=int(pd),**{k:int(v) for k,v in values(body).items()}))
  for pd,body in re.findall(r'KUNLUN common totals PD([01]): ([^\n]+)',s):
   weights.append(dict(name=name,pd=int(pd),**{k:float(v) if k.endswith('_ms') else int(v) for k,v in values(body).items()}))
  for pd,body in re.findall(r'KUNLUN activation PD([01]): ([^\n]+)',s):
   activation.append(dict(name=name,pd=int(pd),**{k:float(v) if k.endswith('_ms') else int(v) for k,v in values(body).items()}))
  for line in s.splitlines():
   if line.startswith('K200PROFILE\t'):
    _,phase,kind,count,ms,bytes_=line.split('\t');entry=dict(name=name,phase=phase,kind=kind,calls=int(count),ms=float(ms),bytes=int(bytes_))
    if kind.startswith('runtime.'):transfers.append(entry)
    elif kind.startswith('q4.'):api.append(entry)
    elif kind.startswith(('kernel.existing_wait.','kernel.enqueue_complete.')):kernelwaits.append(entry)
   elif line.startswith(('K200HBM ','K200L3 ')):
    memory.append(dict(name=name,kind=line.split()[0],**{k:int(v) for k,v in values(line).items()}))
   elif line.startswith('GUARD END'):
    v=values(line);safety.append(dict(name=name,peak_c=float(v['peak']),rc=int(v['rc'])))
   elif 'native totals PD' in line:
    v=values(line);fallback.append(dict(name=name,pd=int(re.search(r'PD([01])',line)[1]),**{k:int(x) for k,x in v.items() if k in ['native_sd','native_cl','cdnn','direct_q4']}))
 # Load only finished diagnostic traces; individual worker spans overlap and are retained separately.
 for f in sorted(O.glob('*.events.csv')):
  name=f.name.removesuffix('.events.csv')
  meta=O/'logs'/(name+'.json')
  if not meta.exists() or json.loads(meta.read_text())['returncode']!=0:continue
  groups=collections.defaultdict(lambda:[0,0,0])
  with f.open() as stream:
   for row in csv.DictReader(stream):
    category=row['category']
    if category not in ['xpu.activation','xpu.node','cpu.node','scheduler.split','backend.sync']:continue
    if category=='cpu.node' and row['worker']!='0':continue
    key=(row['phase'],category,row['name'],row['tensor'] if category=='cpu.node' else '')
    v=groups[key];v[0]+=1;v[1]+=int(row['end_ns'])-int(row['start_ns']);v[2]+=int(row['cpu_ns'])
  for (phase,category,op,tensor),(calls,ns,cpu) in groups.items():timeline.append(dict(name=name,phase=phase,category=category,op=op,tensor=tensor,calls=calls,ms=ns/1e6,cpu_ms=cpu/1e6))
 def aggregate(rows,keys,metrics):
  groups=collections.defaultdict(list)
  for row in rows:
   if all(k in row for k in keys):groups[tuple(row[k] for k in keys)].append(row)
  result=[]
  for key,group in sorted(groups.items()):
   row=dict(zip(keys,key));row['processes']=len(set(x['name'] for x in group))
   for metric in metrics:
    samples=[]
    for x in group:samples.extend(x[metric]['samples'] if isinstance(x[metric],dict) else [x[metric]])
    row[metric]=summary(samples)
   result.append(row)
  return result
 data=dict(standard=standard,standard_summary=aggregate(standard,['model','split','ubatch','share','mac','phase'],['ms','ts']),prefix=prefix,prefix_summary=aggregate(prefix,['model','split','ubatch','mac','pp_tokens'],['pp_ms','tg_ms','pp_ts','tg_ts']),frozen=frozen,frozen_summary=aggregate(frozen,['model','split','stable','phase'],['ms','ts']),transfers=transfers,api=api,kernelwaits=kernelwaits,quant=quant,paths=paths,macpaths=macpaths,workspace=workspace,weights=weights,memory=memory,safety=safety,timeline=timeline,fallback=fallback,activation=activation)
 (A/'llm-results.json').write_text(json.dumps(data,ensure_ascii=False,indent=2))
 for label,rows in [('standard',data['standard_summary']),('prefix',data['prefix_summary']),('frozen',data['frozen_summary'])]:
  flat=[]
  for row in rows:
   x={}
   for k,v in row.items():
    if isinstance(v,dict):x.update({k+'_'+m:n for m,n in v.items() if m!='samples'})
    else:x[k]=v
   flat.append(x)
  if flat:
   with (A/(label+'.csv')).open('w',newline='') as stream:
    writer=csv.DictWriter(stream,fieldnames=list(flat[0]));writer.writeheader();writer.writerows(flat)
 print({k:len(v) for k,v in data.items()});return data
if __name__=='__main__':analyze(sys.argv[1] if len(sys.argv)>1 else Path(__file__).resolve().parent)
