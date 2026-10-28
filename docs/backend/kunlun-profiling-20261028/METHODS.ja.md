# Kunlun K200 0.7.6 — 全体プロファイルの測定方法

## 固定する実装

- ライブラリ: `cbe2be1ae5324edb714472538472f4089e6d9c51`。実行部分は0.7.6の`8a9db26823a79659b5d7b640558c9d98a378e353`と同じ。
- llama.cpp: `1e6191a47b24813396238caad9294d570f6f0446`。submoduleの参照は上記ライブラリcommit。
- 通常測定は、このソースから作った専用ビルドと、凍結した0.7.6 installを使用する。
- 診断は別worktree、別ビルド。カーネル、量子化、weight ABI、演算の実行順序は変更しない。
- CPUはi7-8700、6 threads。XPUは900 MHz。CPU governor等は既存設定を維持する。

## 通常性能と診断の分離

|用途|実装と設定|時間の扱い|
|---|---|---|
|通常のPP512/TG128|未改変の`llama-bench`、5試行、warmupあり|主性能値|
|prefix付きPP/TG|既存`kunlun-llm-smoke`、4 prefix長、各3試行、各warmupあり|主性能値。ただし標準benchと別workload|
|追加同期なしの詳細診断|専用GGML timing hooksとaudit preload|内訳、呼出回数、転送量。observer overheadは別に実測|
|軽量の主内訳|同じGGML hooksとnode-audit。XPU runtime interposeなし|主時間割合とAmdahl。TG校正負荷Tiny約4.16%/7B約1.76%|
|CPU出力層の軽量診断|CPU/harness hooksのみ。XPU runtimeのinterposeなし|Q6_K worker/graph、logits取得、greedy等のCPU処理|
|各kernelの分離診断|起動後に、そのkernelのstreamを待機|起動から完了までの参考値。通常性能へ加算しない|
|既存Flash stage/device-clock probe|既存のPROFILE shaderとTRACE機構|別条件の診断。通常のgroup8 DMAやpipelineの性能と混同しない|

通常実行では`LD_PRELOAD`、K200/FlashのTRACE・PROFILE・VERIFY等を指定しない。実プロセスのロード済みライブラリと、診断関連の初期環境変数も確認する。

全runtime auditのGLU1 TG校正負荷がTiny約18.04%、7B約8.18%と大きかったため、主判断には追加したnode-auditの軽量結果を使う。細かいkernel段階・runtime bytesは全auditから求める。軽量calibrationもABBA各6試行、代表内訳は各3試行で、同一入力のlogitsビット一致を確認した。calibrationはprefix128であり、全条件の費用を一律補正しない。

診断コードは`K200_DIAGNOSTIC_HOOKS`を定義したビルドだけで有効になる。定義のないビルドではtiming引数も評価しない。weak callbackが存在する場合にだけ時刻を取得する。追加のXPU待機を行うのは、明示的な分離診断だけである。

## 時間と呼出回数

`CLOCK_MONOTONIC`で同一の時刻系を使い、CPU workerには`CLOCK_THREAD_CPUTIME_ID`も記録する。

1. PP/TGの各`process`を測定区間とする。
2. XPU nodeとCPU backend graphの区間を取り出す。
3. その外にある実転送、backend同期、schedulerの分割・割当を切り分ける。
4. 未分類の区間をhost/scheduler/observer残余として残す。
5. 区間の和集合と差集合で排他的な内訳を作り、合計がprocess時間と一致することを検証する。

CPU workerの経過時間6本を合計すると、並列実行を二重計上する。workerの区間和集合をcritical-pathに近いwall値として使い、各threadのCPU時間の合計はcore使用量として別に示す。CPU graph wallにはthreadpool起動・barrier等も含まれる。

main threadのCPU時間にはworker 0としての処理も含まれるので、main CPU時間と全worker CPU時間も単純加算しない。runtime auditレコードの`cpu_ns=0`は未取得を表し、CPUを使用していないという意味ではない。

XPU node/API wallには、引数設定、enqueue、既存の待機、device演算が含まれる。`xpu_wait`はその内側の値であり、API時間に加算しない。hostのCPU時間が長くても、driver内でdeviceを待つ時間をCPU行列演算の時間とは扱わない。

wait区間が長いことだけを理由に、それを全て削除可能な「同期オーバーヘッド」とみなさない。waitにはその時点で未完了のXPU計算・DMAも含まれる。enqueue APIはhost発行時間であり、device完了までの時間ではない。純粋なdriver待機費用とdevice busyを完全に分離できない部分は、その限界を残す。

演算の外で生じるruntime待機は、backend同期の残余と分ける。P2Pのready区間には送信元・送信先の既存waitと、先行する出力gatherが含まれ得るため、純粋な転送API時間とは別のinclusive値とする。

Q4の量子化、MAC、scale/sumまたはslice joinは、元から存在する完了待機を境界として分ける。GEMM kernel内で融合されたFP16 block-scale補正とMACは分離しない。引数設定と先行kernel実行が重なる場合も、段階全体のwall区間として一度だけ数える。

Flashは複数streamで動く場合がある。hostの待機時間をKQ・Online・PVへ単純分配しない。stream識別子を記録し、全体API時間、分離診断、device-clock診断を区別する。異なるcluster/coreのticksは合計しない。counterの時刻系が確かめられなければ、coreを跨ぐ絶対時刻から重複率を求めない。

## 転送量とメモリ

H2D/D2H/D2D/P2PはXPU runtimeが受け取る要求bytesを数える。これは論理転送量であり、物理PCIe/内部busの全トラフィックを直接測った値ではない。CPU buffer内のlogits memcpyをD2Hに含めない。Attentionの重複した補助scopeも加算しない。

`copy_rows`によるdevice内コピーは別に記録する。要素演算のalias保護とRow Splitの出力gatherを分ける。enqueue時間だけをコピー完了時間とせず、要素演算内の既存waitまでを確認する。

HBM/L3はAPIで追跡できる確保量と、監視ツールでサンプリングしたHBM peakを分ける。共有L3を参照する各workspaceの統計値は合計しない。確保APIのinterposeで見えないSDK内部確保や、サンプリングの間に発生する短いpeakは限界として記載する。

APIのlive値はemit時点、すなわちモデル解放前の値である。解放後のリーク量とは解釈しない。runtime転送bytesにはdriverのcommand packetやkernel内部DMAの全量は含まれない。

CPUのRSSは通常測定から採用する。詳細診断はイベント保存用のRAMも消費するため、そのRSSを通常推論の必要メモリと解釈しない。backendの`tensor_parts::original`はCPU fallback用に元Q4_0を保持しており、CPU model bufferの表示値以外にもhost RAMを使う。

## workloadの違い

標準`llama-bench`のTG128は空KVからの128回のdecodeで、入力tokenは標準benchが選ぶ。prefix付き試験は同一の2048 IDsの先頭128/512/1024/2048を用い、固定teacher列を入力する。32位置の出力のうち最初はPPから得るため、TGの時間は31回のdecodeに対応する。context容量は4096で固定する。

標準benchのcontext要求はソースの`n_prompt+n_gen+n_depth`に従い、今回のPP512では512、TG128では128（内部alignment paddingは別）。quiet出力では全context初期化行を保存しない。標準benchをcontext容量4096のprefixハーネスと同一条件として扱わない。各標準構成の比較では同じbench設定を使用した。

既存ハーネスは各batchの最後にlogitsを要求する。batchを変更すると、PP中のQ6_K出力層の呼出回数も変わる。そのため128/128、512/512に加え、PD0では512/128も測定し、batchとubatchの影響を分ける。

`K200PERF`のTG時間はモデルprocessとlogits取得に対応し、greedy/finite検査、detokenize、ファイル書込は含まない。これらは別scopeで記録する。通常性能試験ではlogitsファイルを出力しない。

全試行を保存し、平均・中央値・標本標準偏差・min/maxを出す。主性能値は追加同期なし、preloadなしの結果だけを用いる。詳細診断の代表条件は3試行、全条件の経路診断は1試行で演算の多数の呼出しを記録する。calibrationは同一入力のABBA順で、observerあり・なしを各6試行比較する。

## 予測の限界

排他的な時間割合を`f`、対象区間の仮の高速化倍率を`r`とすると、全体倍率の目安は`1 / (1 - f + f/r)`。これは実装後の実測値ではない。operator API全体を半減する仮定と、MACだけを半減する仮定を区別する。

Q6_KのXPU化ではCPU matmulとhidden stateのD2Hを削減できても、新しいXPU演算とlogitsのD2Hが必要になる。CPU greedyを残す場合のlogitsは32000×F32 = 128000 bytesである。この費用をゼロと置いた値は楽観的な上限とする。

## 安全性とGit

各実行前の使用者確認、両PDの状態確認、既存thermal guard、80℃停止、persistent abort latch、失敗時の保存を継承する。XPU異常後に次の試験を無条件で続けない。clock、driver、firmware、OS設定は変更しない。

通常の0.7.6 worktreeと既存の実験worktreeを保護する。診断変更は専用ブランチに保存し、公開ブランチの既定動作は変更しない。生ログ、モデル、SDK、ビルド生成物はcommit対象から外す。
