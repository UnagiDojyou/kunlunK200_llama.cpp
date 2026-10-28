"""Generate Japanese data appendix; all inputs are completed measurements."""
from pathlib import Path
from collections import defaultdict
import csv,statistics,json,math,sys
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent;A=R/'analysis'
def read(n):
 with (A/n).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def avg(rs,k):return statistics.mean(float(r[k]) for r in rs) if rs else 0
def sd(rs,k):return statistics.stdev(float(r[k]) for r in rs) if len(rs)>1 else 0
def group(rs,keys):
 d=defaultdict(list)
 for r in rs:d[tuple(r[k] for k in keys)].append(r)
 return d
def md(headers,rows):return '\n'.join(['|'+'|'.join(map(str,headers))+'|','|'+'|'.join(['---']*len(headers))+'|']+['|'+'|'.join(map(str,row))+'|' for row in rows])+'\n'
def rep(rows,m,c,p):return [r for r in rows if r['name']==f'diagrep-{m}-pd0-u128-g1' and r['context']==str(c) and r['phase']==p]
def aggregate_per_trial(rs,key,metric,divisor=None):
 d=defaultdict(float)
 for r in rs:d[(r['trial'],r[key])]+=float(r[metric])/(float(r[divisor]) if divisor else 1)
 out=defaultdict(list)
 for (trial,k),v in d.items():out[k].append(v)
 return out
wall=read('wall-partition.csv');apis=read('inclusive-api.csv');copies=read('transfers.csv');q4=read('q4-stages.csv');ov=read('overlap.csv');cpu=read('cpu-workers.csv');dev=read('device-copies.csv');kv=read('setrows-overhead.csv')
lines=['# 詳細診断データ付録','', '全て追加同期なしの全runtime observer実行。Single PD0、batch/ubatch=128/128、SwiGLU=1、各3試行。PPは入力tokenあたり、TGは1 decodeあたり。通常のtokens/sとは区別する。TGのobserver費用がTiny約18%/7B約8%あるため、この付録の割合・Amdahlは補助値。主時間割合と判断には[軽量内訳](LIGHT_RESULTS.ja.md)を使用する。kernel段階・bytes・callsをここで詳しく扱う。', '']
am=[];summary_rows=[];transfer_summary=[];stage_summary=[]
for m in ['tiny','7b']:
 for c in [128,2048]:
  for p in ['pp','tg']:
   rows=rep(wall,m,c,p);total=avg([r for r in rows if r['category']=='TOTAL'],'ms_per_token');units=int(rows[0]['tokens'])
   lines+= [f'## {m} / prefix {c} / {p.upper()}','',f'1試行の単位数: {units}。以下の「区間計」は{units}単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。','']
   table=[]
   for (k,),rs in sorted(group(rows,['category']).items(),key=lambda kv:avg(kv[1],'ms_per_token'),reverse=True):
    table.append([k,f"{avg(rs,'ms'):.3f}",f"{avg(rs,'ms_per_token'):.4f} ± {sd(rs,'ms_per_token'):.4f}",f"{avg(rs,'percent'):.2f}",f"{avg(rs,'calls_inclusive') if rs[0].get('calls_inclusive') else avg(rs,'calls'):.0f}"])
    if k!='TOTAL':
     f=avg(rs,'ms_per_token')/total
     am.append(dict(model=m,context=c,phase=p,candidate=k,fraction=f,current_ms_per_token=total,target_ms_per_token=avg(rs,'ms_per_token'),hypothetical_2x_speedup=1/(1-f+f/2),zero_cost_upper_speedup=1/(1-f) if f<1 else math.inf))
    summary_rows.append(dict(model=m,context=c,phase=p,category=k,mean_ms_per_token=avg(rs,'ms_per_token'),std_ms_per_token=sd(rs,'ms_per_token'),mean_percent=avg(rs,'percent'),mean_ms=avg(rs,'ms')))
   lines+= [md(['排他的区間','区間計 ms','ms/単位','割合 %','区間内呼出数 (inclusive)'],table),'呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。','']
   stages=rep(q4,m,c,p);table=[]
   by=aggregate_per_trial(stages,'stage','ms_per_token')
   for k,v in sorted(by.items(),key=lambda kv:statistics.mean(kv[1]),reverse=True):
    value=statistics.mean(v);table.append([k,f'{value:.4f}',f'{value/total*100:.2f}'])
    stage_summary.append(dict(model=m,context=c,phase=p,stage=k,mean_ms_per_token=value,process_percent=value/total*100))
    if 'MAC' in k:
     f=value/total;am.append(dict(model=m,context=c,phase=p,candidate=k,fraction=f,current_ms_per_token=total,target_ms_per_token=value,hypothetical_2x_speedup=1/(1-f+f/2),zero_cost_upper_speedup=1/(1-f)))
   lines+=['### Q4の既存完了waitを境界とする段階時間','',md(['段階','ms/単位','process比 %'],table),'MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。','']
   cr=rep(copies,m,c,p);table=[]
   for (direction,),rs in sorted(group(cr,['direction']).items()):
    by_trial=group(rs,['trial']);bs=[sum(float(r['bytes']) for r in v)/units for v in by_trial.values()];ms=[sum(float(r['ms']) for r in v)/units for v in by_trial.values()];ns=[len(v)/units for v in by_trial.values()]
    b=statistics.mean(bs);t=statistics.mean(ms);n=statistics.mean(ns)
    table.append([direction,f'{b/1024:.3f}',f'{t:.5f}',f'{n:.2f}'])
    transfer_summary.append(dict(model=m,context=c,phase=p,direction=direction,bytes_per_token=b,api_ms_per_token=t,calls_per_token=n))
   lines+=['### runtime論理転送','',md(['方向','KiB/単位','API ms/単位','回/単位'],table),'API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。','']
   tr=[]
   for (direction,tensor,dst),rs in group(cr,['direction','tensor','tensor_dst']).items():
    by_trial=group(rs,['trial']);b=sum(float(r['bytes']) for r in rs)/3/units;t=sum(float(r['ms']) for r in rs)/3/units
    tr.append([direction,tensor,dst,f'{b/1024:.3f}',f'{t:.5f}'])
   lines+= [md(['方向','src/owner tensor','dst tensor','KiB/単位','API ms/単位'],sorted(tr,key=lambda r:float(r[3]),reverse=True)[:15]),'']
   overlaps=rep(ov,m,c,p)
   lines+=['### 待機・起動・CPU（包含値）','',md(['指標','平均/単位'],[[k,f'{avg(overlaps,k)/units:.5f}'] for k in ['process_wall_ms','api_union_ms','wait_union_ms','launch_union_ms','main_thread_cpu_ms','other_worker_cpu_ms','wait_calls','launch_calls']]),'main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。','']
   inner=rep(apis,m,c,p);inner_table=[]
   for prefix in ['runtime.arguments.','runtime.launch_config.','runtime.launch.','runtime.wait.','peer.ready_inclusive.']:
    rr=[r for r in inner if r['key'].startswith(prefix)]
    inner_table.append([prefix,f"{sum(float(r['calls']) for r in rr)/3/units:.2f}",f"{sum(float(r['ms']) for r in rr)/3/units:.5f}"])
   lines += [md(['host API分類','回/単位','包含 ms/単位'],inner_table),'引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。','']
   als=rep(dev,m,c,p);alias=[r for r in als if r['alias_candidate']=='True'];completion=sum(float(r['completion_ms']) for r in alias if r['completion_ms'])/3/units
   lines += ['### device内copy_rows','',md(['指標','1単位平均'],[['全copy回数',f'{len(als)/3/units:.2f}'],['alias候補回数',f'{len(alias)/3/units:.2f}'],['alias候補payload KiB',f"{sum(float(r['bytes']) for r in alias)/3/units/1024:.3f}"],['alias候補 enqueue→既存wait完了 ms',f'{completion:.5f}']]),'alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。','']
   kr=rep(kv,m,c,p);kt=[]
   for metric in ['node_ms','library_union_ms','index_get_union_ms','layout_index_validation_host_residual_ms']:
    kt.append([metric,f"{sum(float(r[metric]) for r in kr)/3/units:.5f}"])
   assert abs(sum(float(r['node_ms'])-float(r['library_union_ms'])-float(r['index_get_union_ms'])-float(r['layout_index_validation_host_residual_ms']) for r in kr))<1e-6
   lines += ['### SET_ROWSとindex取得・検証','',md(['区間','ms/単位'],kt),'libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。','']
lines+=['## Q6_K出力層の軽量observer','', 'XPU runtime auditを外し、CPU workerとharnessだけを計測。worker wallの和集合を使う。sum CPUは複数coreのCPU時間。logits取得はCPU buffer内のコピーであり、XPU D2Hに加えない。postはgreedy/有限値検査/detokenizeで、通常PP/TGの計測外。','']
head=read('head-low-overhead.csv');table=[]
for (m,c,p),rs in sorted(group(head,['model','context','phase']).items()):
 units=int(c) if p=='pp' else int(rs[0]['q6_calls'])
 cols=['q6_worker_union_ms','q6_graph_ms','q6_worker_cpu_ms','logits_get_ms','postprocessing_ms']
 table.append([m,c,p,f'{avg(rs,"q6_calls"):.0f}']+[f'{avg(rs,k)/units:.5f}' for k in cols[:-1]]+[f"{avg(rs,'postprocessing_ms')/avg(rs,'postprocessing_calls'):.5f}" if avg(rs,'postprocessing_calls') else '—'])
lines += [md(['モデル','prefix','phase','Q6呼出/試行','Q6 worker ms/単位','Q6 graph ms/単位','sum CPU ms/単位','logits get ms/単位','post ms/position'],table),'postのscopeは最初のPP logitsもTGラベルへ入り、1試行32位置。postはposition数で割り、31decodeでは割らない。','']
lines+=['## observer校正（ABBA）','', '通常A→診断B→診断B→通常A。同じteacher入力、各実行3試行、A/B各6試行。PP128とその後31decode。','']
cal=read('calibration.csv');lines+=[md(['モデル','GLU','phase','observer','n','平均 ms/単位','中央値','SD','平均増加 %'],[[r['model'],r['swiglu'],r['phase'],r['observer'],r['n_ms_per_token'],f"{float(r['mean_ms_per_token']):.5f}",f"{float(r['median_ms_per_token']):.5f}",f"{float(r['std_ms_per_token']):.5f}",f"{float(r['mean_overhead_percent']):.2f}" if r.get('mean_overhead_percent') else '—'] for r in cal]),'']
lines+=['## CPU fallback一覧（全経路診断）','','CPUノードを実測で検出したもの。metadata viewは算術fallbackと分ける。worker時間は同じ演算・同じtensor内の和集合。','']
fallback=[]
for keys,rs in sorted(group([r for r in cpu if r['name'].startswith('diag-')],['model','split','batch','ubatch','swiglu','phase','op','tensor']).items()):
 if keys[-2] in ['VIEW','RESHAPE','TRANSPOSE','PERMUTE','NONE']:continue
 fallback.append(list(keys)+[sum(int(r['node_calls']) for r in rs),f"{sum(float(r['worker_wall_union_ms']) for r in rs):.3f}"])
lines+=[md(['model','split','batch','ubatch','GLU','phase','op','tensor','calls（4長さ合計）','worker union計 ms'],fallback),'']
lines+=['## 全構成の上位ボトルネック','','全経路診断は各長さ1試行、TGは8decode。上記の代表3試行とは統計精度が異なる。転送は方向別のruntime論理bytes。','']
matrix=[];matrix_csv=[]
for keys,rs in sorted(group([r for r in wall if r['name'].startswith('diag-')],['name','model','split','batch','ubatch','swiglu','context','phase']).items()):
 name,m,split,b,u,g,c,p=keys;total=next(float(r['ms_per_token']) for r in rs if r['category']=='TOTAL');units=int(rs[0]['tokens']);largest=sorted([r for r in rs if r['category']!='TOTAL'],key=lambda r:float(r['percent']),reverse=True)[:3]
 trans=[r for r in copies if r['name']==name and r['context']==c and r['phase']==p];bs={k:sum(float(r['bytes']) for r in trans if r['direction']==k)/units for k in ['H2D','D2H','P2P']}
 matrix.append([m,split,f'{b}/{u}',g,c,p,f'{total:.4f}']+[f"{r['category']} ({float(r['percent']):.1f}%)" for r in largest]+[f'{bs[k]/1024:.3f}' for k in ['H2D','D2H','P2P']])
 matrix_csv.append(dict(zip(['name','model','split','batch','ubatch','swiglu','context','phase'],keys),ms_per_token=total,**{k+'_bytes_per_token':v for k,v in bs.items()},**{'top'+str(i+1):r['category'] for i,r in enumerate(largest)}))
lines+=[md(['model','split','b/ub','GLU','prefix','phase','ms/単位','第1','第2','第3','H2D KiB','D2H KiB','P2P KiB'],matrix),'']
lines+=['## 分離kernel診断','','起動後にstream待機を追加した参考診断。通常0.7.6の並列性を保った性能値ではない。以下を通常wall内訳へ加算しない。','']
serial=read('serialized-kernels.csv');lines+=[md(['model','GLU','prefix','phase','kernel分類','回/単位','enqueue→完了 ms/単位'],[[r['model'],r['swiglu'],r['context'],r['phase'],r['kind'],f"{int(r['kernels'])/int(r['tokens']):.2f}",f"{float(r['ms'])/int(r['tokens']):.5f}"] for r in serial]),'']
lines+=['## 排他的割合を使ったAmdahl仮定','','対象区間全体を2倍、または費用ゼロにした楽観的な全体倍率。実装後の速度ではない。別候補の値は足し合わせない。','']
lines+=[md(['model','prefix','phase','対象','割合 %','仮に対象2倍: 全体倍率','費用ゼロ上限倍率'],[[r['model'],r['context'],r['phase'],r['candidate'],f"{100*r['fraction']:.2f}",f"{r['hypothetical_2x_speedup']:.3f}",f"{r['zero_cost_upper_speedup']:.3f}"] for r in am if r['candidate'] in ['Q4 GEMV/GEMM','FlashAttention XPU','Q6_K output CPU graph','SwiGLU XPU','KV SET_ROWS XPU'] or 'MAC' in r['candidate']]),'']
def write(n,rs):
 with (A/n).open('w',newline='',encoding='utf-8') as f:
  w=csv.DictWriter(f,list(dict.fromkeys(k for r in rs for k in r)));w.writeheader();w.writerows(rs)
write('amdahl.csv',am);write('representative-wall.csv',summary_rows);write('representative-transfers.csv',transfer_summary);write('representative-q4-stages.csv',stage_summary);write('configuration-bottlenecks.csv',matrix_csv)
(R/'DIAGNOSTIC_RESULTS.ja.md').write_text('\n'.join(lines),encoding='utf-8')
print('DIAGNOSTIC_REPORT_COMPLETE')
