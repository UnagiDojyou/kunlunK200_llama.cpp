# Q4_0 GEMV N-major 再評価（2026-10-29）

## 結果

7B向け3形状だけのopt-inとして採用した。最終TG中央値は約+2.0%、初期の同一0.7.8ホスト比較は約+0.8%であり、効果は小さい。TinyLlamaの高速化は達成していない。基準は0.7.8のstream GEMV（KUNLUN_Q4_MAC_DMA=1）で、0.7.7とは比較していない。最終commitはdelivery-summary.jsonに記録する。

## 改善点と実装

ActivationをL1Dへ常駐させたまま、同じN128群の隣接するN64を処理し、その群についてresident内のK窓を先に巡回するN-major順序を追加した。weight/scale DMAの局所性を狙った変更で、転送byte数、MAC回数、partialの配置、FP32加算順序、全既存fence/barrierは維持する。新たな非同期化や二重bufferは追加していない。物理HBM帯域の増加を計測したものではない。

適用はKp x Np = 4096 x 4096 / 4096 x 11008 / 11008 x 4096のみ。従来streamのshape/ABI検査が先に適用される。TinyLlama等の未検証shapeは0.7.8順序へ戻す。Row Splitの代表shapeもこの制限で既存経路になる。

変更はcommon-q4/gemv-stream.hの走査順序分岐、新しいkunlun-common-gemv-nmajor.xpu、src/q4-common.incのopt-in選択、CMake/versionだけ。公開API、weight ABI v1、量子化方式、Lease、GEMM、Attention、SET_ROWS、SwiGLUは変更しない。

## 試した案

| 案 | 判定 |
| --- | --- |
| CL最終和の32/64ブロック読出し | 一貫した改善がなく不採用 |
| CL最終和の8ブロック固定展開 | K11008の局所差だけで、全体効果が小さいため不採用 |
| SDのcount=16専用化 | 単体の小改善がN-majorとの組合せで消えたため不採用 |
| MAC scale設定の4ブロック共有 | 代表形状で改善せず不採用。出力はbitwise一致 |
| N-major + 連続N cluster割当 | K11008で効果がなく、採用せず |
| N-major単独 | 7B向け3形状のみopt-in採用 |

全shapeへN-majorを使った初期試験では、7B TG 8.026 -> 8.090 token/s（+0.80%）に対し、TinyLlama TGは-4.88%だった。TinyLlamaでの回帰を隠さず、最終候補から該当shapeを除外した。

## GEMV単体

公開APIのprepared Lease入力。量子化を除くSD + 従来CL和 + APIのwall時間。PD0/F32、warmup後3反復、順序反転。旧新は同じ0.7.9 DSO内で切り替える。以下の旧は0.7.8とbyte一致するstream kernelである。

| K x N | 0.7.8順序 ms | N-major ms | 時間短縮率 | 旧/新SD ms |
| --- | --- | --- | --- | --- |
| 4096 x 4096 | 0.228929 | 0.228047 | +0.39% | 0.001181/0.002437 |
| 4096 x 11008 | 0.519604 | 0.512659 | +1.34% | 0.002820/0.004166 |
| 11008 x 4096 | 0.599632 | 0.595809 | +0.64% | 0.000733/0.003573 |

探索段階はPD0の代表3形状、各5～7反復だけ。生値は各*-results.jsonとlogs/に保存した。大規模profile、全context、perplexity、全バイナリの再ハッシュは実施していない。再ビルドした旧streamが0.7.8とbyte一致、新kernelが試作のN-majorとbyte一致する2件だけを確認した。

## 通常LLM

Single PD0、b/ub=128、CPU threads=6、OMP=6/PASSIVE、F16 KV、Attention 0.7.6、Activation共有=1、SwiGLU=0。標準llama-bench PP512/TG128。各設定2プロセス x 2試行=4標本のABBA（TinyはBAAB）。token/s。診断フック・LD_PRELOAD・追加同期は使わない。同一の旧llama実行ファイルと同一の新DSOでNMAJORだけを切り替えた。旧buildと新buildの絶対速度を混ぜていない。

| モデル | 処理 | 旧 平均/中央値/標本SD | 新 平均/中央値/標本SD | 中央値差 |
| --- | --- | --- | --- | --- |
| Llama 2 7B | TG | 7.949 / 7.947 / 0.095 | 8.103 / 8.107 / 0.017 | +2.01% |
| Llama 2 7B | PP | 75.521 / 75.511 / 0.415 | 75.561 / 75.545 / 0.273 | +0.04% |
| TinyLlama | TG | 26.112 / 26.139 / 0.105 | 26.053 / 26.071 / 0.321 | -0.26% |
| TinyLlama | PP | 306.165 / 306.048 / 0.527 | 307.421 / 306.431 / 2.685 | +0.13% |

平均・中央値・標本SDは少数標本の記述統計であり、有意差検定ではない。同一プロセス内の2試行は独立とは限らない。PPの小差を高速化として主張しない。大きなTG改善を得たとは言わず、特定shapeの小改善として扱う。

## 正確性・メモリ・安全性

公開APIは7ケース。主要3形状、F16/外れ値、ゼロ、K/N端数fallback、M8 GEMMを含み、旧新出力・guard・量子化値/scale・weightがbitwise一致し、upload=1を確認した。統合途中で旧streamの初回ログ条件を誤り、N-majorでも毎GEMVログが出るミスがあった。条件へ!nmajorを追加し、旧/新各workspaceで初回だけ出力することをAPI試験で確認した。混入系列final-llm-results.json/confirm-results.jsonは採否から除外し、final-clean-llm-results.jsonへ通常性能を取り直した。GPU算術や同期は変わらず、正確性の大規模再試験は行っていない。benchmarkはPP/TGで2 contextを作るため、初回メッセージ数の検査もそれを反映した。

短い実LLM logitsは7B PD0共有ON/OFF、Tiny PD0共有OFF、7B PD1/Layer/Rowの6組で一致。7Bの共有ON/OFF間も一致した。N-major使用/shape fallbackをログで確認した。今回は長contextやcorpus評価は省略した。

重みの再pack・二重常駐はなく、A/scale/partial/workspaceの容量も不変。L1D最大512 KiB/cluster、L3は既存poolと寿命を維持。programは旧7960 byte/新8084 byteで、loader確保はいずれも8448 byte。設定固定の実LLMでは置換される。API試験のように同一workspaceで両方を起動した場合は2本を保持するため8448 byte増える。HBM総量の全profiler測定は行っていない。

既存guard下の実行は67件、全終了コード0、最高60.0℃。両PD RUNNING/no errorで、errtaskは前回終了時の履歴と一致。80℃停止とabort latchを維持し、reset、driver/firmware/clock/OS設定変更は行っていない。旧0.7.8の両worktreeはHEAD/clean状態を維持した。

## 使い方・再現

追加設定は KUNLUN_Q4_NMAJOR=1。既定は0。KUNLUN_Q4_MAC_DMA=1とGGML_KUNLUN_COMMON_Q4=1が前提。未対応shapeと無効設定は既存経路へ戻る。実際のargv/envはlogs/*.json、最小測定手順はllm-trial.py/final-tests.pyに保存してある。

~~~sh
source /home/unagidojyou/kunlun-k200/env.sh
export LD_LIBRARY_PATH=/home/unagidojyou/kunlun-k200/results/gemv-next-20261029/install/lib:$LD_LIBRARY_PATH
export GGML_KUNLUN_COMMON_Q4=1 GGML_KUNLUN_Q4_ACT_SHARE=1
export KUNLUN_Q4_MAC_DMA=1 KUNLUN_Q4_NMAJOR=1
export GGML_KUNLUN_SWIGLU=0
# その他の固定Attention環境はllm-env.jsonを使用。
# 最終比較の新経路を同じguardと引数で再実行する例:
cd /home/unagidojyou/kunlun-k200/results/gemv-next-20261029
python3 - <<'PY'
import json,runner
m=json.load(open("logs/clean-llm-7b-a1-i0.json"))
runner.run("repro-7b",m["command"],m["environment"],600)
PY
~~~

## 残る改善余地

N-majorの効果は小さい。DMA転送量やMAC数を減らしていないため、大幅な改善にはSDの発行・供給効率を変える別方式が必要。CL最終和は転送粒度変更だけで改善せず、起動・完了待ちの削減には安全なSD→CL依存制御が必要になる。今回は未確認のfence削除や融合は行わなかった。TinyLlamaの小N形状は個別の走査順序・負荷配分が次の候補である。

## Git

Library: a0e02f78d3b93aa3434fca8a651b041de14669df。vendor/KunlunK200はこのcommitへ更新。llama.cppの演算コードは変更せず、最小要求versionとpin・文書のみを更新した。両開発branchはcodex/k200-gemv-next-20261029。最終parent commit/clean状態はdelivery-summary.jsonを参照。push・main/master反映はしていない。
