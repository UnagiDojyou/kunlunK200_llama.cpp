from pathlib import Path
from collections import defaultdict
import csv,json,sys
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=R/'analysis';out={m:defaultdict(float) for m in ['tiny','7b']}
def rs(n):
 with (A/n).open(newline='',encoding='utf-8') as f:
  for r in csv.DictReader(f):
   if r.get('name') in ['diag-tiny-row-u128-g1','diag-7b-row-u128-g1'] and r['context']=='128' and r['phase']=='tg':yield r
for r in rs('q4-stages.csv'):out[r['model']]['q4_library_apis_ms']+=float(r['ms_per_token'])
for r in rs('inclusive-api.csv'):
 k=r['key']
 for prefix in ['peer.ready_inclusive.','runtime.copy.P2P.','copy_rows.enqueue.']:
  if k.startswith(prefix):out[r['model']][prefix+'ms']+=float(r['ms'])/8;out[r['model']][prefix+'calls']+=int(r['calls'])/8
for r in rs('wall-partition.csv'):
 if r['category']=='Q4 GEMV/GEMM':out[r['model']]['q4_backend_node_ms']=float(r['ms_per_token'])
with (A/'row-api-summary.json').open('w') as f:json.dump(out,f,indent=2)
print(json.dumps(out,indent=2))
