from pathlib import Path
from collections import defaultdict
import csv,json,statistics,sys
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=R/'analysis'
def read(n,rep=False):
 with (A/n).open(newline='',encoding='utf-8') as f:return [r for r in csv.DictReader(f) if not rep or r.get('name','').startswith('diagrep-')]
def by(rs,keys):
 d=defaultdict(list)
 for r in rs:d[tuple(r[k] for k in keys)].append(r)
 return d
result={}
stages=read('q4-stages.csv',True)
result['q4_stages']=[dict(model=k[0],context=k[1],phase=k[2],stage=k[3],ms_token=sum(float(r['ms_per_token']) for r in v)/3) for k,v in by(stages,['model','context','phase','stage']).items()]
result['opportunities']=read('bounded-opportunities.csv')
cr=read('transfers.csv',True)
result['transfers']=[dict(model=k[0],context=k[1],phase=k[2],direction=k[3],kib_token=sum(float(r['bytes'])/int(r['tokens']) for r in v)/3/1024,ms_token=sum(float(r['ms'])/int(r['tokens']) for r in v)/3,calls_token=sum(1/int(r['tokens']) for r in v)/3) for k,v in by(cr,['model','context','phase','direction']).items()]
result['head_hidden_transfers']=[dict(model=k[0],context=k[1],phase=k[2],tensor=k[3],kib_token=sum(float(r['bytes'])/int(r['tokens']) for r in v)/3/1024,ms_token=sum(float(r['ms'])/int(r['tokens']) for r in v)/3) for k,v in by([r for r in cr if r['direction']=='D2H' and 'result' in r['tensor']],['model','context','phase','tensor']).items()]
kv=read('setrows-overhead.csv',True)
result['setrows']=[dict(model=k[0],context=k[1],phase=k[2],**{z:sum(float(r[z])/int(r['tokens']) for r in v)/3 for z in ['node_ms','library_union_ms','index_get_union_ms','layout_index_validation_host_residual_ms']}) for k,v in by(kv,['model','context','phase']).items()]
ov=read('overlap.csv',True)
result['host']=[dict(model=k[0],context=k[1],phase=k[2],**{z:statistics.mean(float(r[z])/int(r['tokens']) for r in v) for z in ['process_wall_ms','wait_union_ms','launch_union_ms','wait_calls','launch_calls','main_thread_cpu_ms','other_worker_cpu_ms']}) for k,v in by(ov,['model','context','phase']).items()]
tr=read('transfers.csv');result['glu_transfers']=[]
for m in ['tiny','7b']:
 for g in [0,1]:
  v=[r for r in tr if r['name']==f'diag-{m}-pd0-u128-g{g}' and r['phase']=='tg' and r['context']=='128']
  result['glu_transfers'].append(dict(model=m,glu=g,**{d+'_kib_token':sum(float(r['bytes'])/int(r['tokens']) for r in v if r['direction']==d)/1024 for d in ['H2D','D2H','P2P']}))
apis=read('inclusive-api.csv',True)
result['tg_kernel_counts']=[dict(model=k[0],context=k[1],kernel=k[2],calls_token=sum(float(r['calls'])/int(r['tokens']) for r in v)/3) for k,v in by([r for r in apis if r['phase']=='tg' and r['key'].startswith('runtime.launch.') and any(x in r['key'] for x in ['flash','SDK'])],['model','context','key']).items()]
(A/'findings.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
