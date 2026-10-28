# 再現手順

対象ホストは `unagidojyou@192.168.50.112`。以下はそのLinuxホスト上で実行する。SDK、モデル、凍結installはGitに含めない。

## ソースとビルド

`build-manifest.json`とモデルSHA256を照合する。通常版のソースはllama.cpp `1e6191a47b24813396238caad9294d570f6f0446`、ライブラリ`cbe2be1ae5324edb714472538472f4089e6d9c51`。

```bash
cd /home/unagidojyou/kunlun-k200
source env.sh
O="$PWD/results/profiling-20261028"
cmake -S worktrees/publish-attention-20261028-llama \
  -B build/profiling-076-normal -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=ON \
  -DKunlunK200_DIR="$O/frozen-install/lib/cmake/KunlunK200" \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_SERVER=OFF -DLLAMA_BUILD_APP=OFF
cmake --build build/profiling-076-normal -j4 \
  --target llama-bench kunlun-llm-smoke kunlun-quality-eval
```

診断版は`a4f96e61b6906f71416bec1d8daebd46447f9fe3`のworktreeを用い、C/CXX両方へ`-DK200_DIAGNOSTIC_HOOKS`を渡す。保存したbuild-update-diag.shは通常測定後の移行処理も含むため、ビルドのみ再現する際はcmake/c++行を選ぶ。head-audit.soは別途`c++ -shared -fPIC -O2 -std=c++17 head-audit.cpp -o head-audit.so -pthread`で作る。

## 安全ガード付き通常測定

measure.pyは古いK200/KUNLUN/GGML_KUNLUN設定とLD_PRELOADを消去してから、明示したbase環境を使う。runner.pyが既存thermal guard・両PD確認・80℃停止・persistent latch・失敗保存を行う。

```bash
cd /home/unagidojyou/kunlun-k200
source env.sh
python3 - <<'PY'
import sys
from pathlib import Path
R=Path('/home/unagidojyou/kunlun-k200')
O=R/'results/profiling-20261028'
sys.path.insert(0,str(O))
from measure import base,once,smoke
args=[R/'build/profiling-076-normal/bin/llama-bench',
 '-m',R/'models/tinyllama-q4_0.gguf','-p',512,'-n',128,'-r',5,
 '-b',128,'-ub',128,'-t',6,'-ngl',99,'-sm','none','-dev','KUNLUN0',
 '-mg',0,'-ctk','f16','-ctv','f16','-fa','on','-o','json']
once('repro-standard-tiny-pd0-u128-g1',args,{**base,'GGML_KUNLUN_SWIGLU':'1'})
smoke('repro-prefix-tiny-pd0-u128-g1','tiny','pd0',128,1,
 predict=32,repeat=3,lengths='128,512,1024,2048',warmup=1)
PY
```

既に成功した同名ログがあればonceは再実行せず結果を再利用する。再計測には新しいnameを使う。停止ラッチを削除して再計測を強行しない。

変更箇所：モデルはllama2-7b-q4_0.gguf / smokeの`7b`、PD1は`-dev KUNLUN1` / smokeの`pd1`。Layer/Rowは`-sm layer/row`、`-dev KUNLUN0/KUNLUN1`、`-ts 1/1`、smokeの`layer/row`。SwiGLU無効は`GGML_KUNLUN_SWIGLU=0` / glu=0。batch512/ubatch128のsmokeは`extra={'K200_BATCH':'512'}`、512/512はu=512。

共通環境はmeasure.pyのbaseに保存した。COMMON_Q4=1、ATTENTION=1、FLASH=1、ROPE=1、FLASH_KV=1、Online/KQ/Decoding/Pipeline/KV_DMA/KV_MAXはauto。REQUANT_I8や旧native GEMVへの切替えは今回行っていない。

## 詳細診断

```python
smoke('repro-diag-tiny-pd0-u128-g1','tiny','pd0',128,1,
 diag=True,predict=32,repeat=3,lengths='128,2048',warmup=1)
```

解析器は今回の`diag-*` / `diagrep-*`のname規則を使うため、新nameを解析する場合はregexへ追加する。全計画はdiagnostic-plan.pyに保存した。全経路は各長さ1試行9位置、代表内訳は3試行32位置。通常tokens/sへ診断結果を混ぜない。

K200_SERIAL_KERNELS=1はstream待機を追加する分離診断。FLASH_DECODING_PROFILE / FLASH_PIPELINE_PROFILEはprofile shaderとlegacy DMA等へ切り替わる比較条件で、通常0.7.6のgroup8/pipeline性能ではない。

## オフライン解析

計測ホスト上でベンチマークと同時に解析・圧縮・大きな転送をしない。今回の集計と作図はWindows上で行った。

```bash
python analyze.py
python analyze-extra.py
python compare-glu.py
python analyze-flash.py
python report-normal.py
python report-diagnostic.py
python plots.py
python analyze-light.py
python report-light.py
python plots-light.py
python analyze-opportunities.py
python analyze-configurations.py
python summarize-findings.py
python inspect-row-apis.py
python report-main.py
```

解析はPython標準ライブラリのみ。作図はMatplotlib。plot-depsの導入はWindowsのworkspace内だけに行った。plot-deps、SDK、bin、so、モデル、生traceはcommitしない。raw-data-manifest.jsonに元ログ・trace・logits dumpのSHA256を保存する。

軽量node-audit.cppはSDK/runtime interpositionを使わず、head-auditと同じc++オプションでコンパイルする。light-plan.pyにABBA校正と代表3試行を保存した。各解析器には必要に応じて末尾へ結果ディレクトリを渡せる。Gitに収録する解析ツールと、実機固有のガード付き制御スクリプト・生データは分けて保存している。
