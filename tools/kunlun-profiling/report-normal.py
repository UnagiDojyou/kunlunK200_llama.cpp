from pathlib import Path
import csv,re,sys
root=Path(sys.argv[1]) if len(sys.argv)>1 else Path(__file__).resolve().parent
def load(name):
 with (root/'analysis'/name).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
rows=load('ordinary-summary.csv');thermal=load('thermal-memory.csv');logs=load('dispatch-logs.csv')
models={'tiny':'TinyLlama 1.1B Q4_0','7b':'Llama 2 7B Q4_0'};splits={'pd0':'Single PD0','pd1':'Single PD1','layer':'Layer','row':'Row'}
def key(r):return r['model'],r['split'],int(r['batch']),int(r['ubatch']),int(r['swiglu'])
keys=sorted({key(r) for r in rows},key=lambda k:(['tiny','7b'].index(k[0]),['pd0','pd1','layer','row'].index(k[1]),k[2],k[3],k[4]))
def get(k,kind,phase,context):return next(r for r in rows if key(r)==k and r['kind']==kind and r['phase']==phase and int(r['context'])==context)
def fmt(r):return f"{float(r['tokens_s_mean']):.3f} ± {float(r['tokens_s_std']):.3f} / {float(r['tokens_s_median']):.3f}"
text=['# 通常性能の全測定結果（0.7.6）','',
'追加同期・profiling preloadなしの結果。数値は **平均 ± 標本標準偏差 / 中央値**、速度の単位はtoken/s。全試行の値、min/max、msの統計は[ordinary-samples.csv](analysis/ordinary-samples.csv)と[ordinary-summary.csv](analysis/ordinary-summary.csv)に保存する。','',
'6 CPU threads、XPU900 MHz、FP16 K/V、common Q4/Flash/RoPE有効。通常実行と詳細診断は別ビルド。測定方式と限界は[METHODS.ja.md](METHODS.ja.md)を参照。','',
'## 標準llama-bench PP512 / 空KV TG128','',
'各5試行、warmupあり。PPとTGは独立した試験。標準benchが選ぶtoken列を使う。TG128は128tokenのprefixを与えた試験ではない。','',
'|モデル|Split|batch/ubatch|SwiGLU|PP512|TG128（空KV）|','|---|---|---:|---:|---:|---:|']
for k in keys:
 pp=get(k,'standard','pp',512);tg=get(k,'standard','tg',0)
 assert int(float(pp['tokens_s_n']))==5 and int(float(tg['tokens_s_n']))==5
 text.append(f'|{models[k[0]]}|{splits[k[1]]}|{k[2]}/{k[3]}|{k[4]}|{fmt(pp)}|{fmt(tg)}|')
text+=['','![通常benchの比較](figures/normal-standard-128.png)','',
'## 固定prefix後のPP/TG','',
'context容量4096、同じ2048 IDsの先頭を使用。各条件はwarmup後3試行。TGは固定teacher入力による31 decode/試行。32位置のうち最初のlogitsはPPから得る。greedy/finite確認・detokenize等はTG時間の外。標準benchとは入力・context容量・測定範囲が異なるため、直接混ぜて比較しない。','']
for model in models:
 text += ['### '+models[model],'',
 '|Split|batch/ubatch|GLU|prefix|PP全体 ms（平均）|PP ms/token|PP token/s|TG ms/decode|TG token/s|',
 '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for k in keys:
  if k[0]!=model:continue
  for ctx in [128,512,1024,2048]:
   pp=get(k,'prefix','pp',ctx);tg=get(k,'prefix','tg',ctx)
   assert int(float(pp['tokens_s_n']))==3 and int(float(tg['tokens_s_n']))==3
   assert int(tg['tokens_per_trial'])==31
   text.append(f"|{splits[k[1]]}|{k[2]}/{k[3]}|{k[4]}|{ctx}|{float(pp['ms_mean']):.3f}|{float(pp['ms_mean'])/ctx:.4f}|{fmt(pp)}|{float(tg['ms_mean'])/31:.4f}|{fmt(tg)}|")
 text.append('')
text+=['![prefix性能](figures/normal-prefix-configurations.png)','',
'## 通常実行のメモリ・温度','',
'prefix試験（context容量4096）でのprocess全体のサンプリングpeak。SwiGLUの有効・無効を並べる。HBM監視は0.5秒間隔で、短いpeakを逃し得る。PD別peakの時点は同一とは限らず、単純和を同時使用量と断定しない。CPU RSSはmodelロードも含む最大値で、CPU model buffer表示に含まれない元Q4_0 shadow等も含む。診断イベント保存RAMは含まない。','',
'|モデル|Split|batch/ubatch|GLU|PD0 HBM peak MiB|PD1 HBM peak MiB|CPU RSS peak MiB|最高温度℃（両PD）|XPU MHz min/max|',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for k in keys:
 pp=get(k,'prefix','pp',128)
 n=next(r['name'] for r in logs if key(r)==k and r['line'].startswith('K200RSS'))
 rssline=next(r['line'] for r in logs if key(r)==k and r['line'].startswith('K200RSS'))
 rss=int(re.search(r'peak_kib=(\d+)',rssline)[1])/1024
 a=[r for r in thermal if r['name']==n];assert len(a)==2
 peaks={r['pd']:float(r['sampled_peak_hbm_mib']) for r in a}
 text.append(f"|{models[k[0]]}|{splits[k[1]]}|{k[2]}/{k[3]}|{k[4]}|{peaks['0']:.0f}|{peaks['1']:.0f}|{rss:.1f}|{max(max(float(r['max_die_c']),float(r['max_hbm_c'])) for r in a):.1f}|{min(float(r['min_mhz']) for r in a):.0f}/{max(float(r['max_mhz']) for r in a):.0f}|")
text+=['','全flag・通常benchも含む監視値は[thermal-memory.csv](analysis/thermal-memory.csv)、ロード時間は[loads.csv](analysis/loads.csv)、既存dispatch/workspace/CPU使用量の出力は[dispatch-logs.csv](analysis/dispatch-logs.csv)。','',
'## 解釈上の注意','',
'- CPU clock/governor/affinityは既存設定を維持した。小さい条件間差は、process間の変動も含めて慎重に扱う。表の標準偏差は同一process内の試行から求めた値。','- batch変更は出力logitsの要求回数や最終層のgraph pruningも変える。512/128と512/512を対にしてubatchの影響を確認する。','- ubatch512では、既存のAttention query/workspace制限によりPPのAttentionがCPUへfallbackする。速度低下量が全てこの処理の時間であるとは限らず、詳細診断で切り分ける。','- 既定値の変更は行っていない。GLU0/1、batch/ubatch、各Splitは明示設定で比較した。','']
(root/'NORMAL_RESULTS.ja.md').write_text('\n'.join(text),encoding='utf-8')
print('NORMAL_REPORT_COMPLETE',len(keys),'configurations')
