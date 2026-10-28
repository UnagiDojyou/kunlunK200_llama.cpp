from pathlib import Path
from collections import defaultdict
import csv,sys,statistics
R=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def read(n):
 with (R/'analysis'/n).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def table(h,rs):return '\n'.join(['|'+'|'.join(h)+'|','|'+'|'.join(['---']*len(h))+'|']+['|'+'|'.join(map(str,r))+'|' for r in rs])+'\n'
cal=read('light-calibration.csv');wall=read('light-wall-partition.csv');amd=read('light-amdahl.csv')
out=['# 軽量observerによる主時間内訳','', 'XPU runtimeのinterposeを外し、GGMLの演算・CPU worker・scheduler・tensor操作scopeだけを収集した。追加XPU同期なし、同じ0.7.6 library/shader、同じGGML演算ソース。全runtime traceは転送bytes・kernel段階の補助資料へ分離する。','', '## ABBA校正','', 'Single PD0、SwiGLU=1、prefix128。各実行3試行、A/B各6試行。全32位置のlogitsは4実行間でビット一致。','',table(['model','observer','phase','n','平均 ms/token','中央値','SD','通常比増加 %'],[[r['model'],r['observer'],r['phase'],r['n_ms_per_token'],*[f"{float(r[k+'_ms_per_token']):.5f}" for k in ['mean','median','std']],f"{float(r['mean_overhead_percent']):.2f}" if r.get('mean_overhead_percent') else '—'] for r in cal]),'', '## 排他的区間','', 'PD0、batch/ubatch128/128、SwiGLU=1、context容量4096、各3試行。PPの単位はprompt token、TGはdecode。外部tensor操作にはCPU bufferコピーも含むため、このscopeのbytesをH2D/D2Hへ読み替えない。','']
for m in ['tiny','7b']:
 for c in [128,2048]:
  for p in ['pp','tg']:
   rs=[r for r in wall if r['model']==m and r['context']==str(c) and r['phase']==p];by=defaultdict(list)
   for r in rs:by[r['category']].append(r)
   rows=[]
   for k,vs in sorted(by.items(),key=lambda kv:statistics.mean(float(r['ms_per_token']) for r in kv[1]),reverse=True):
    assert len(vs)==3
    rows.append([k,f"{statistics.mean(float(r['ms']) for r in vs):.3f}",f"{statistics.mean(float(r['ms_per_token']) for r in vs):.5f} ± {statistics.stdev(float(r['ms_per_token']) for r in vs):.5f}",f"{statistics.mean(float(r['percent']) for r in vs):.2f}",f"{statistics.mean(float(r['calls']) for r in vs):.0f}"])
   out +=[f'### {m} / prefix {c} / {p.upper()}','',table(['区間','区間計 ms','ms/単位 ± SD','割合 %','呼出/試行'],rows),'']
out += ['## Amdahl仮定','','軽量診断の排他的区間を使う。対象を2倍、または費用ゼロと置いた全体倍率。実装後の実測値ではない。observerの費用も含まれ、別候補の倍率を足したり掛けたりしない。','',table(['model','prefix','phase','対象','割合 %','対象2倍の全体倍率','費用ゼロ上限'],[[r['model'],r['context'],r['phase'],r['candidate'],f"{float(r['fraction'])*100:.2f}",f"{float(r['hypothetical_2x_speedup']):.3f}",f"{float(r['zero_cost_upper_speedup']):.3f}"] for r in amd if r['candidate'] in ['Q4 GEMV/GEMM','FlashAttention XPU','Q6_K output CPU graph','RMSNorm XPU','Elementwise/residual XPU','KV SET_ROWS XPU','SwiGLU XPU']]),'']
(R/'LIGHT_RESULTS.ja.md').write_text('\n'.join(out),encoding='utf-8');print('LIGHT_REPORT_COMPLETE')
