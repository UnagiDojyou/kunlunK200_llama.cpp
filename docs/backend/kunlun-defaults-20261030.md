# Kunlun K200 0.7.12：SwiGLU再評価と安全な最適化の既定化

2026-10-30 JST。0.7.11を保護し、両repoの独立branch codex/k200-defaults-20261030で実装・実機検証。push・PR・main/master反映なし。

## 1. 採用判断

Activation共有、GEMV stream、7B向けN-major、M128 GEMM Ownersを**既存の共通Q4プロファイル内・検証済みshapeに限定して既定有効化**した。従来の明示1利用者は同じ演算経路を維持する。SwiGLUは現在の構成でPP/TGとも高速化したが、7Bの文章生成が分岐するため既定0。Large PPも品質判断の範囲が限られ、既定0を維持。

| 設定 | 0.7.11未設定 | 0.7.12未設定 | 範囲／依存関係 |
| --- | --- | --- | --- |
| GGML_KUNLUN_Q4_ACT_SHARE | 0 | 共通Q4内で1 | 同一tensor・graph/group世代・PD、F32連続matrix、複数consumer。Lease/書込み検出/16MiB準備上限を維持 |
| KUNLUN_Q4_MAC_DMA | 0 | 対応shapeで1 | M=1・K未padding、既存14shape、L3条件 |
| KUNLUN_Q4_NMAJOR | 0 | 対応shapeで1 | streamが必要。Kp/Np=4096/4096,4096/11008,11008/4096 |
| KUNLUN_Q4_GEMM_OWNERS | 0 | 対応shapeで1 | physical M=tileM=128、上記3つのKp/Np。Tiny shapeは対象外 |
| KUNLUN_LARGE_PP | 0 | 0 | 明示1で0.7.11のquery分割・関連placement・KV writer拡張 |
| GGML_KUNLUN_SWIGLU | 0 | 0 | 1=対応PP/TG、pp=複数行のみ、0=CPU |

4つの新既定は明示0で旧経路、1で従来の検証済み高速経路、不正値で無効。MAC_DMA=0ならN-majorも使わない。共有とOwnersはstreamに依存しない。shape、prepared容量・契約、L3制限のfallbackは既存のまま。実HBM確保失敗／kernel errorは従来fail-stopであり、異常後の無条件fallbackや継続はしない。

**共通Q4やAttention自体は今回既定化していない。** COMMON_Q4 > NATIVE_Q4 > REQUANT_I8 > direct Q4の優先順位を維持。COMMON_Q4/NATIVE_Q4/REQUANT_I8、ATTENTION/FLASH、BROADCAST_VECTOR、FLASH_KVは引き続き明示有効化。RoPEはAttention有効時の既存既定。RMSNorm、ROW_COPY/ROW_SPLIT、SET_ROWS、Flash内部のauto/Online/pipeline/KV_MAX等は変更なし。詳細なshape一覧・他設定は両repoのDEFAULTS/optimization-defaults文書、source由来のoption-inventory.jsonを参照。

GGML_KUNLUN_DIAGNOSTICS=1で起動policyを1行表示。実際の選択は既存のstream/N-major/Owners/共有counterでも確認可能。設定だけのログを実行回数と混同しない。

## 2. SwiGLU CPU/XPUの通常性能

Single PD0、threads6、F16 KV、batch512、ubatch128/512。標準llama-bench PP512/TG128（TGは空KVの独立試験）。Attention/Flash/RoPE、共有、stream、N-major、Owners、Large PP=1、broadcast/FlashKVを固定し、SwiGLUだけ0/1。同じ0.7.11実行ファイル・libraryでABBA各2sample。token/s中央値±sample SD（n=2では平均=中央値）。warmup/loadは計測外。callback・LD_PRELOAD・capture・詳細profileなし。

| モデル | ubatch | 処理 | CPU SwiGLU | XPU SwiGLU | 変化 |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 128 | PP | 78.069 ±0.014 | 80.274 ±0.030 | +2.82% |
| Llama 2 7B | 128 | TG | 8.101 ±0.015 | 8.471 ±0.054 | +4.57% |
| Llama 2 7B | 512 | PP | 78.553 ±0.177 | 80.538 ±0.007 | +2.53% |
| Llama 2 7B | 512 | TG | 8.032 ±0.118 | 8.498 ±0.000 | +5.81% |
| TinyLlama 1.1B | 128 | PP | 308.664 ±2.230 | 319.154 ±0.885 | +3.40% |
| TinyLlama 1.1B | 128 | TG | 26.132 ±0.315 | 28.789 ±0.124 | +10.17% |
| TinyLlama 1.1B | 512 | PP | 315.541 ±5.225 | 326.862 ±2.759 | +3.59% |
| TinyLlama 1.1B | 512 | TG | 25.153 ±1.348 | 28.782 ±0.068 | +14.43% |

以前のTiny TG回帰はこの構成では再現しなかった。Attention・graph配置が変わった結果であり、全モデル/contextへの保証ではない。

別runの転送診断：7B、ubatch512、固定512token PP後に3TG step。既存observerを使い、強制node同期を追加していない。表はruntime copy APIで、device内部DMAとは区別。

| phase | 転送 | MiB CPU版→XPU版 | 呼出回数 | API inclusive ms |
| --- | --- | --- | --- | --- |
| PP | H2D | 675.210 → 8.671 | 194 → 163 | 110.954 → 2.638 |
| PP | D2H | 1349.476 → 16.392 | 323 → 259 | 221.242 → 6.010 |
| TG | H2D | 4.530 → 0.498 | 380 → 284 | 3.733 → 2.065 |
| TG | D2H | 8.207 → 0.144 | 777 → 585 | 7.022 → 4.395 |

PPのCPU GLU32→0、XPU GLU0→32。TG3stepはCPU96→0、XPU0→96。PPのCPU/XPU graph35/34→3/2（合計69→5）、TG3stepは105/102→9/6（207→15）。P2P=0。XPU graph自身はGLUを引き受けて長くなる一方、scheduler境界と外側の転送が減る。inclusive graph/copy/wait時間を単純加算していない。診断時間を通常token/sに混入させていない。

HBM API peak4,350,267,392→4,372,814,848byte（+21.5MiB）、L3 peak12,713,984byteで不変。XPU compute buffer68.01→89.51MiB、CPU105.52→103.52MiB。GLU独自workspaceは0。共有HBM2,134,016byte、prepare256/reuse384/quant_chunks890も両版同じ（診断run全体累積）。Leaseの共有条件を緩めていない。

## 3. SwiGLU単体誤差とA8への伝播

既存GGML試験10case（F32/F16、separate/packed/swapped/padded/3D）、未対応stride/BF16/GEGLU/0/pp配置検査にpass。合成入力は±200、微小値、±0、大負値を含む。F32 CPU SIMD対XPU：NMSE1.55e-18、最大絶対差1.91e-6、最大相対差3.46e-7（参照絶対値>1e-6）。大負値の最大差5.29e-23、正側outlierとゼロ近傍はこのfixtureでは差0。既存の特殊値処理とstack guardを維持。

実LLM layer0のPP128 gate/upを同一入力として実GGML CPU/XPU GLUへ再投入。**CPU再生は取得済みCPU GLU出力とbitwise一致**。後段も実libraryのprepare/Leaseで比較した。captureは同期診断専用で通常測定外。

| 実入力 | GLU NMSE | 最大絶対差 | 最大相対差（参照絶対値>1e-6） | A8値不一致byte | scale float不一致 |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B layer0・128行 F32 | 2.9408718769e-15 | 1.19209289551e-07 | 4.15733258889e-07 | 0/1409024 | 608/2048 |
| TinyLlama 1.1B layer0・128行 F32 | 3.19219589106e-15 | 2.98023223877e-08 | 3.75638667906e-07 | 0/720896 | 544/2048 |

scaleは各row16個の同値複製なので、7B38/128row、Tiny34/128rowにscale差。このfixtureの整数bin自体は一致し、直ちにbin差が出たとは扱わない。scaleのulp差がMACに伝わり、後段Row A8で丸め境界を跨ぎ得る。以前の評価では次layer norm後のbin差も確認されている。

CPUはAVX2/FMAの多項式exp近似、XPUはCL expf・除算・乗算で、実FFNはF32入出力・F32中間。暗黙のFP16化やgate/up取り違えは確認されない。実入力のXPU対scalar libm NMSEは7B4.71e-16、Tiny4.50e-16。F64からF32へ丸めた参照ではXPUのNMSEがCPU SIMDより小さい例もある。CPU SIMDへのbitwise一致と数学的精度向上は同義ではない。

実Tiny入力をF16へ丸めた試験は720,896要素中5要素差、NMSE1.86e-12、最大絶対差3.81e-6（F16丸め境界）。F16の一般的bitwise同等性も主張しない。

誤差自体はFP32丸め程度で、全体差にはA8感度が関与する。特定CPU近似の固定移植や低速な倍精度expは追加せず、元のSwiGLU計算を維持した。追加実装は再利用可能な数値再生ツール。

## 4. logits・生成品質・KV

SwiGLU以外を固定。teacher replayはCPU版の生成tokenを両版に入力し、全32位置で同じprefixを保証。KLはCPU-GLU分布基準D_KL(p||q)。自由生成の位置一致とteacher argmax一致は別指標。

| モデル | 条件 | teacher NMSE | 平均KL | 最大絶対差 | teacher argmax一致 | 自由生成一致 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 自然文短prompt・Large PP=0 | 0.000142865 | 0.000302494 | 0.538461 | 31/32 | 25/32 |
| TinyLlama 1.1B | 自然文短prompt・Large PP=0 | 5.09464e-06 | 8.64014e-06 | 0.070816 | 32/32 | 32/32 |
| Llama 2 7B | 固定512token・Large PP=1 | 0.00133282 | 0.00218623 | 1.22793 | 31/32 | 31/32 |
| TinyLlama 1.1B | 固定512token・Large PP=1 | 0.00067723 | 0.000914446 | 0.755702 | 32/32 | 32/32 |

自然文promptは「The capital of France is」。実例の共通生成部分は「a city of contrasts ... a mix of old and new」。その後CPU版は「of tradition and modernity, of history and」、XPU版は「of rich and poor, of beautiful and ugly」へ分岐した。Tinyは「Paris.」「The capital of Spain is Madrid.」などを含む同じ32token列。7Bは24tokenまで一致してから分岐（位置一致25/32）。自然文は短いのでubatch512でも大query分割を使わず、別途固定512token・ubatch512で上表の組合せ確認を実施した。

各mode内のPP→TG、F16 KV保存→進行→復元→再実行でnext logitsはbitwise一致。単体誤差が小さくても生成が変わるため、品質同等としてSwiGLUを既定化しない。

過去のAttention更新前の1020位置評価は7B PPL8.35053→8.37023（+0.24%）、Tiny11.16770→11.10330。今回のFlash/Large PP構成の測定値ではなく、量子化感度の既存証拠としてのみ参照。今回はopt-in維持に十分な差を確認したためcorpus/PPLを追加しなかった。Large PPも0.7.11の限定的CPU/XPU比較だけでは品質同等と判断しない。

## 5. 新しい既定値の性能と互換性

同じ0.7.12実行ファイル・libraryを使用。common Q4/Attention等のプロファイルは明示有効。SwiGLU/Large PPは未設定=0。4つの出力一致型設定を全0、新既定（未設定）、全1で比較。全0/新既定はABBA各2sample、全1は補助確認1process内2反復。表は中央値±SD、n=2。

| モデル | ubatch | 処理 | 4機能全0 | 新既定 | 4機能明示1 | 全0比 |
| --- | --- | --- | --- | --- | --- | --- |
| Llama 2 7B | 128 | PP | 70.742 ±0.607 | 78.004 ±0.001 | 78.446 ±0.455 | +10.26% |
| Llama 2 7B | 128 | TG | 7.139 ±0.020 | 8.108 ±0.019 | 8.142 ±0.046 | +13.58% |
| Llama 2 7B | 512 | PP | 68.557 ±0.088 | 74.154 ±0.019 | 74.220 ±0.004 | +8.16% |
| Llama 2 7B | 512 | TG | 7.274 ±0.001 | 8.239 ±0.019 | 8.173 ±0.006 | +13.27% |
| TinyLlama 1.1B | 128 | PP | 278.859 ±0.159 | 314.462 ±10.365 | 306.551 ±0.420 | +12.77% |
| TinyLlama 1.1B | 128 | TG | 21.958 ±0.897 | 25.988 ±0.533 | 26.343 ±0.113 | +18.35% |
| TinyLlama 1.1B | 512 | PP | 234.492 ±0.009 | 247.884 ±0.313 | 248.274 ±0.053 | +5.71% |
| TinyLlama 1.1B | 512 | TG | 22.497 ±0.221 | 25.340 ±1.196 | 25.414 ±0.514 | +12.64% |

全0比は既定化の効果で、既に全1を使っていた環境への新規kernel高速化率ではない。第2節はSwiGLUだけの独立ABBA。追加で同じ0.7.12実行ファイル（SHA一致確認）に全6機能を明示1として各2反復した補助比較も以下に示す。未設定側は上表のABBA値、全6機能側は1process内2反復で、順序反転した新しい対比較ではない。独立機能の改善率は第2節を使い、細かな差を有意な改善とは判断しない。新既定ではLarge PP=0なのでubatch512のCPU Attention fallbackが残る。速度優先のその経路は引き続き明示1。

| モデル | ubatch | 処理 | 新既定（数値modeは0） | 全6機能明示1 |
| --- | --- | --- | --- | --- |
| Llama 2 7B | 128 | PP | 78.004 ±0.001 | 80.307 ±0.009 |
| Llama 2 7B | 128 | TG | 8.108 ±0.019 | 8.540 ±0.005 |
| Llama 2 7B | 512 | PP | 74.154 ±0.019 | 81.054 ±0.043 |
| Llama 2 7B | 512 | TG | 8.239 ±0.019 | 8.483 ±0.071 |
| TinyLlama 1.1B | 128 | PP | 314.462 ±10.365 | 320.916 ±0.090 |
| TinyLlama 1.1B | 128 | TG | 25.988 ±0.533 | 28.799 ±0.026 |
| TinyLlama 1.1B | 512 | PP | 247.884 ±0.313 | 332.388 ±0.016 |
| TinyLlama 1.1B | 512 | TG | 25.340 ±1.196 | 28.847 ±0.015 |

精度：両モデル×ubatch128/512のPP512→4logitsで未設定/全1/全0がbitwise一致。7Bの各機能個別0も一致。TinyのPD1/Layer/Row PP128→TG4では未設定対全1が一致し、state復元もpass。全Split性能測定は追加していない。

libraryの5shapeを0/1/未設定/不正値で比較し、N-major/stream/M128 Ownersのbitwise一致、K512のstream fallback、M32既存GEMMを確認。支持範囲を拡張していない。実HBM枯渇を故意に発生させる試験は省略し、既存容量・L3 guardを維持。runtime確保失敗は従来の安全停止。

system-library要求を0.7.12に統一し、0.7.11 packageをconfigure時に拒否することを確認。bundled submoduleも同commit。公開API、kernel source、weight ABI/pack、量子化方式、FP32累積順序、同期・所有権は変更なし。

## 6. 変更と再現

library：src/q4-common.incの未設定時選択、version/CMake/README、docs/optimization-defaults.md、bench/q4-defaults-test.cpp。backend：q4-activation-share.incの既定、起動時policy、最小library version、examples/kunlun/swiglu-numerics.cpp、設定文書、gitlink。通常推論は数値診断を呼ばない。

保存済みbuild.pyを使用：libraryはRelease、KUNLUN_BUILD_BENCHMARKS=OFF、KUNLUN_PREBUILT_KERNELSにソース不変の0.7.11 images。backendはRelease、GGML_KUNLUN=ON、KUNLUN_USE_SYSTEM_LIBRARY=ON、CMAKE_PREFIX_PATH=/home/unagidojyou/kunlun-k200/results/defaults-20261030/install。独立buildでSDK・モデル複製なし。

~~~sh
cmake --build /home/unagidojyou/kunlun-k200/build/defaults-20261030-lib -j4 --target kunlunk200
cmake --build /home/unagidojyou/kunlun-k200/build/defaults-20261030-llama -j4 --target llama-bench kunlun-llm-smoke kunlun-swiglu-numerics
cd /home/unagidojyou/kunlun-k200/results/defaults-20261030
python3 - <<'PY'
import json,runner
m=json.load(open('logs/default-perf-7b-u128-default-i1.json'))
e=m['environment']
# 旧経路へ：4変数を各0。既存高速設定の明示：各1。
# e.update({'GGML_KUNLUN_Q4_ACT_SHARE':'0','KUNLUN_Q4_MAC_DMA':'0',
#            'KUNLUN_Q4_NMAJOR':'0','KUNLUN_Q4_GEMM_OWNERS':'0'})
# 品質差のある追加opt-in：
# e.update({'KUNLUN_LARGE_PP':'1','GGML_KUNLUN_SWIGLU':'1'})
runner.run('manual-default-pp512',m['command'],e,300)
PY
~~~

標準benchmark引数：-p 512 -n 128 -r 1 -b 512 -ub 128/512 -t 6 -ngl 99 -sm none -dev KUNLUN0 -mg 0 -ctk f16 -ctv f16 -fa on -o json。モデルは既存llama2-7b-q4_0.gguf/tinyllama-q4_0.gguf。環境はnew-environment.jsonと各metadataに保存。SwiGLU比較はswiglu-evaluate.py、品質はquality.py/prefix-quality.py、既定化比較はregression.py。再実行は必要caseだけ既存runnerを使う。

## 7. 安全性・残課題・Git

80℃thermal guard、persistent abort、両PD監視、他利用者検査を継承。今回device異常/resetなし。driver/firmware/clock/OS変更なし。最初の数値ツールCMake configureでimported targetのscope不足を修正（hostのみ、device起動前）。全kernel再hash、大量CSV、全context/Split測定はしていない。

残課題はSwiGLUのA8感度・F16境界・異なる文章での品質、Large PPの品質と既定化。GLU単体はPPでCPUより遅く、全体改善は転送とgraph境界削減による。Embedding/Q6_K出力層等のCPU fallbackは変更していない。

基準library 9d399d41bc9a0072aab90c64a074e0883165bbc2、backend382db101dcb039d3782f6fcc08d4de577b1d7582とbuildを保護。新library 8e901e1f2e438091d9edfa761cfbb8a59bd98ca6、backend機能commit 76600486cc2bf432341c8f5f583285191fd95bf9。submoduleは新libraryと一致。レポートcommitを含む最終HEAD/clean状態はdelivery-summary.jsonに記録。SDK/model/build/log/raw logitsはGit対象外。

最終機能commitから再buildし、library DSOは通常測定時とSHA一致。最終buildの7B・PP512→TG4で新既定／全6機能明示1それぞれが対応する測定済みlogitsとbitwise一致。両PD RUNNING/no error・errtask不変、デバイス異常/resetなし。
