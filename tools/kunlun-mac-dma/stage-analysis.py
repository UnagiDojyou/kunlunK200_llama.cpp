from pathlib import Path
import json,re,statistics as st,collections,hashlib,sys
from analysis_helpers import ticks_by_name
O=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=O/'analysis';A.mkdir(exist_ok=True)
def stat(v):
 v=list(v);return dict(n=len(v),mean=st.mean(v),median=st.median(v),stddev=st.stdev(v) if len(v)>1 else 0,min=min(v),max=max(v),samples=v)
stages=[]
for f in sorted((O/'logs').glob('special-profile-*.txt')):
 text=f.read_text();rows=[json.loads(s.split(' ',1)[1]) for s in text.splitlines() if s.startswith('MAC_RESULT ')];ticks=ticks_by_name(text)
 for i,row in enumerate(rows):
  if not row['name'].startswith('profile-'):continue
  samples=ticks[row['name']][1:];assert len(samples)==len(row['sd_ms']),(f,row['name'],len(samples))
  t=[st.median(v[j] for v in samples) for j in range(512)]
  normal_name={'profile-base':'base-gemv'}.get(row['name'],row['name'].removeprefix('profile-'))
  normal=next(x for x in rows if x['name']==normal_name)
  stages.append(dict(normal_name=normal_name,normal_sd_ms=stat(normal['sd_ms']),profile_overhead_percent=100*(st.median(row['sd_ms'])/st.median(normal['sd_ms'])-1),pd=row['pd'],k=row['k'],n=row['n'],m=row['m'],name=row['name'],sd_ms=stat(row['sd_ms']),join_ms=stat(row['join_ms']),core_ticks=[t[c*16:(c+1)*16] for c in range(32)],tick_note='Median over all ten measured invocations, grouped by exact TICKS name after discarding the initial warmup sample; issue/completion/barrier overlap across owners, not exclusive engine busy time.'))
(A/'selected-stages.json').write_text(json.dumps(stages,indent=2))
asm=[]
for name in ['baseline-source','special16','special128','profile-base','profile-special128','profile-rows128']:
 p=O/'probe-images'/(name+'.s')
 if not p.exists():continue
 text=p.read_text();instructions=collections.Counter(m[1] for line in text.splitlines() if (m:=re.match(r'\s+([a-zA-Z][a-zA-Z0-9_.]*)\s',line)))
 b=O/'probe-images'/(name+'.bin')
 asm.append(dict(name=name,assembly_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),binary_bytes=b.stat().st_size if b.exists() else None,binary_sha256=hashlib.sha256(b.read_bytes()).hexdigest() if b.exists() else None,mnemonics=dict(instructions),note='Static instructions for whole image including all eight specialized owner functions; not dynamic instruction count.'))
(A/'assembly.json').write_text(json.dumps(asm,indent=2))
print('STAGES',[(x['name'],x['pd'],x['k'],x['n']) for x in stages]);print('ASM',[(x['name'],x['binary_bytes']) for x in asm])
for row in stages:
 if row['pd']==0 and row['k']==4096 and row['n']==4096:
  print(row['name'],'SD',row['sd_ms']['median'],'JOIN',row['join_ms']['median'])
  for i in range(8):print(i,row['core_ticks'][i][:7])
