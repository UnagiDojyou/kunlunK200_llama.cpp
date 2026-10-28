"""Measured optimistic bounds, not implementation claims."""
from pathlib import Path
from collections import defaultdict
import csv,re,statistics,sys
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=R/'analysis'
def read(n):
 with (A/n).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
wall=read('wall-partition.csv');q4=read('q4-stages.csv');copies=read('device-copies.csv');out=[]
for m in ['tiny','7b']:
 for c in [128,2048]:
  for p in ['pp','tg']:
   def select(rows):return [r for r in rows if r['name']==f'diagrep-{m}-pd0-u128-g1' and r['context']==str(c) and r['phase']==p]
   total=statistics.mean(float(r['ms_per_token']) for r in select(wall) if r['category']=='TOTAL')
   qs=[r for r in select(q4) if r['stage']=='Activation quantization']
   candidates={
    'Eliminate K/V/up repeated activation quantization (keep Q/gate)':sum(float(r['ms_per_token']) for r in qs if re.search(r'^(?:Kcur|Vcur|ffn_up)(?:-|$)',r['tensor']))/3,
    'Eliminate all activation quantization (impossible zero-cost bound)':sum(float(r['ms_per_token']) for r in qs)/3,
    'Eliminate alias-protection copies and their existing completion waits':sum(float(r['completion_ms'])/int(r['tokens']) for r in select(copies) if r['alias_candidate']=='True' and r['completion_ms'])/3,
   }
   for k,v in candidates.items():
    assert 0<=v<=total,(k,v,total)
    out.append(dict(model=m,context=c,phase=p,candidate=k,removable_ms_per_token_optimistic=v,process_ms_per_token=total,fraction=v/total,zero_cost_upper_speedup=1/(1-v/total)))
with (A/'bounded-opportunities.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,out[0].keys());w.writeheader();w.writerows(out)
print('BOUNDED_OPPORTUNITIES_COMPLETE')
