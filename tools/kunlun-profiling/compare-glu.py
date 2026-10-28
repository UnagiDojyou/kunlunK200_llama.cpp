"""Reuse identical teacher-forced calibration dumps; no additional inference."""
from pathlib import Path
import array,math,json,sys
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;out=[]
for model in ['tiny','7b']:
 for tag in ['a1','a2']:
  for trial in range(3):
   suffix='.'+str(trial) if trial else ''
   paths=[R/(f'cal-{model}-g{g}-{tag}.bin'+suffix) for g in [0,1]]
   if not all(p.exists() for p in paths):continue
   a,b=[array.array('f',p.read_bytes()) for p in paths];assert len(a)==len(b)==32*32000
   err=sum((x-y)**2 for x,y in zip(a,b));power=sum(x*x for x in a);kls=[];matches=0
   for i in range(32):
    x=a[i*32000:(i+1)*32000];y=b[i*32000:(i+1)*32000];xm=max(x);ym=max(y)
    zx=math.log(sum(math.exp(v-xm) for v in x))+xm;zy=math.log(sum(math.exp(v-ym) for v in y))+ym
    kls.append(max(0,sum(math.exp(u-zx)*((u-zx)-(v-zy)) for u,v in zip(x,y))))
    matches+=max(range(32000),key=x.__getitem__)==max(range(32000),key=y.__getitem__)
   out.append(dict(model=model,tag=tag,trial=trial,reference='SwiGLU CPU',candidate='SwiGLU XPU',positions=32,nmse=err/max(power,1e-300),max_abs=max(abs(x-y) for x,y in zip(a,b)),kl_mean=sum(kls)/32,kl_max=max(kls),greedy_matches=matches,bitwise=paths[0].read_bytes()==paths[1].read_bytes()))
(R/'analysis/swiglu-logits.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
assert len(out)==12 and all(math.isfinite(v['nmse']) for v in out)
