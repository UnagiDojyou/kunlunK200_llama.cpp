from pathlib import Path
import csv,sys
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=R/'analysis'
with (A/'ordinary-summary.csv').open(newline='',encoding='utf-8') as f:rs=list(csv.DictReader(f))
def find(kind,m,s,b,u,g,c,p):return next(r for r in rs if (r['kind'],r['model'],r['split'],r['batch'],r['ubatch'],r['swiglu'],r['context'],r['phase'])==(kind,m,s,str(b),str(u),str(g),str(c),p))
out=[]
for r in rs:
 if r['swiglu']!='1':continue
 z=find(r['kind'],r['model'],r['split'],r['batch'],r['ubatch'],0,r['context'],r['phase'])
 out.append(dict(comparison='SwiGLU 0 to 1',kind=r['kind'],model=r['model'],split=r['split'],batch=r['batch'],ubatch=r['ubatch'],context=r['context'],phase=r['phase'],before_tokens_s=float(z['tokens_s_mean']),after_tokens_s=float(r['tokens_s_mean']),rate_gain_percent=(float(r['tokens_s_mean'])/float(z['tokens_s_mean'])-1)*100,wall_reduction_percent=(1-float(r['ms_mean'])/float(z['ms_mean']))*100))
for m in ['tiny','7b']:
 for c in [128,512,1024,2048]:
  for p in ['pp','tg']:
   b=find('prefix',m,'pd0',512,512,1,c,p);a=find('prefix',m,'pd0',512,128,1,c,p)
   out.append(dict(comparison='ubatch512 to128, batch512',kind='prefix',model=m,split='pd0',batch=512,ubatch=128,context=c,phase=p,before_tokens_s=float(b['tokens_s_mean']),after_tokens_s=float(a['tokens_s_mean']),rate_gain_percent=(float(a['tokens_s_mean'])/float(b['tokens_s_mean'])-1)*100,wall_reduction_percent=(1-float(a['ms_mean'])/float(b['ms_mean']))*100))
with (A/'configuration-effects.csv').open('w',newline='',encoding='utf-8') as f:
 w=csv.DictWriter(f,out[0].keys());w.writeheader();w.writerows(out)
print('CONFIGURATION_EFFECTS_COMPLETE')
