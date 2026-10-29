# Q4_0 GEMM M128 controller専用化（0.7.10）

## 結果と基準

0.7.9のGEMMを基準に、M128 Rows128/Combo4のcontroller専用化を実装した。採用は7Bの代表3形状に限定し、TinyLlamaはTG維持を優先して対象外とした。新設定はKUNLUN_Q4_GEMM_OWNERS=1、既定は0。GEMV stream/N-major・Activation共有は全比較で有効、SwiGLUは無効。0.7.9の正常なsource/buildは保護し、push・main/master反映は行わない。

## 改善箇所と選定

既存診断のM128/K4096/N11008ではSD約6.47ms、MAC owner約3.74ms、EW owner約5.43msだった。これらは重複区間であり加算しない。今回はdevice-clock profileを取り直さず、EWを含むcontrollerの動的分岐・除算・アドレス計算とM128固定パラメータに着目した。

| 候補 | 単体結果と判断 |
| --- | --- |
| EWのfirst分岐を外へ出し、b/ns loopをflat化 | 約0.3～1.1%の小改善。最終実装には採用せず |
| M128専用controller + 固定タイル定数 | Rows/Comboで約6～7%短縮、7BのPP全体で確認し、3形状限定で採用 |
| 上記2案の組合せ | 専用controllerへの追加差が約0.0～0.2%に留まり、不採用 |

SDKのew_cfg_coeffは係数をまとめた命令なので、未確認の部分レジスタ更新は使わなかった。専用化は6つのengine ownerと2つのidle controllerを分離し、M128/nt32/kt128、work_m=1の定数でコードを生成する。EWのK32補正・FP32累積・finishの順序、weight/scale DMA、DS、MAC、RS、全既存fence/barrierは維持する。EW演算量を減らしたものではなく、EW単独の改善寄与は分離計測していない。

## 単体性能

PD0、共有Leaseのprepared API、warmup後3反復・順序反転。量子化を除くSD + 必要なCL join + API wall、中央値ms。旧新は同一0.7.10 DSO内で切替。Wideの小差は同一カーネルのばらつきである。

| M | K x N | 旧ms | 新ms | 時間短縮 | 適用 |
| --- | --- | --- | --- | --- | --- |
| 128 | 4096 x 11008 | 6.4619 | 6.0397 | +6.53% | 対象 |
| 128 | 11008 x 4096 | 6.8085 | 6.3635 | +6.54% | 対象 |
| 32 | 4096 x 4096 | 1.0775 | 1.0779 | -0.04% | 旧Wideのまま |

探索は指定3形状から開始し、有望なM128経路だけN4096とTinyLlama代表2形状を追加した。探索のSDのみの値はowner-results.json/combined-results.jsonに保存。大規模profile、全context/batch、perplexity、全kernel再検証は行っていない。

## 通常LLM性能

Single PD0、b/ub128、CPU threads6、OMP6/PASSIVE、KV F16、FlashAttention 0.7.6、Lease=1、MAC_DMA=1、NMAJOR=1、SwiGLU=0。標準llama-bench PP512/TG128。7Bは2プロセス x 2試行=各4標本のABBA。TinyLlamaはfallback後の機能確認として各1プロセス x 2試行であり、性能改善の根拠にはしない。同じ実行ファイル・同じ0.7.10 DSOでGEMM_OWNERSだけを変えた。診断hook・LD_PRELOAD・追加同期はない。

| モデル | 処理 | 各n | 旧 平均/中央値/標本SD | 新 平均/中央値/標本SD | 中央値差 |
| --- | --- | --- | --- | --- | --- |
| Llama 2 7B | PP | 4 | 75.719 / 75.695 / 0.454 | 78.108 / 78.114 / 0.048 | +3.20% |
| Llama 2 7B | TG | 4 | 7.948 / 7.935 / 0.118 | 8.086 / 8.084 / 0.017 | +1.89% |
| TinyLlama | PP | 2 | 306.334 / 306.334 / 0.222 | 306.286 / 306.286 / 0.957 | -0.02% |
| TinyLlama | TG | 2 | 26.304 / 26.304 / 0.231 | 26.265 / 26.265 / 0.135 | -0.15% |

初期の同一0.7.9 DSOでkernelだけを切り替えた試作でも、7B PPは75.34から78.04 token/s（+3.58%）だった。少数標本の記述統計であり有意差検定ではない。同じプロセスの2試行は独立とは限らず、TGの小差を高速化とは主張しない。

TinyLlamaへ広く適用した段階ではPP中央値が+6.88%だった一方、TGは-2.63%だった。逆順の追加確認でもTG -2.13%が観測された。ばらつきは大きく原因分離は未完だが、同方向の差が続いたため、慎重にTiny形状を除外した。元の系列はfinal-results.jsonとconfirm-results.jsonに保存し、最終の制限後結果と混ぜていない。TinyLlamaのPP高速化は今回採用しない。

## 精度・適用条件・メモリ

公開APIは9ケース。指定3形状、F16/外れ値、ゼロ、端数M65/K544/N513、M129のchunk/crop、Tiny代表2形状、M1 GEMVを確認した。出力/guard・量子化値/scale・weightはbitwise一致、upload=1。通常APIとLeaseを両方使い、無効設定で既存GEMMへ戻ることも確認した。

実LLMはPP128 + 短いgreedy生成で、両モデルPD0を0.7.9基準とbitwise比較。7B共有ON/OFFの一致、PD1・Layer/Row Splitの旧新一致を確認した。長context/corpus評価は今回省略した。

適用は物理M128、padded Kp x Np = 4096 x 4096 / 4096 x 11008 / 11008 x 4096、nt32/kt128、4 cluster x 8 controller、既存capacity検査でtileM128となるRows128/Combo4のみ。TinyLlamaやRow Split代表shapeは既存GEMMへ戻す。M8/32/64や他条件は既存経路。端数Mは既存padding/crop、M>128は既存chunkを維持する。GGML側のshape/type/stride判定は変更していない。

weight ABI v1、FP16 K32 scale、Row A8量子化、Lease寿命は不変。重みの再pack/二重常駐、追加のActivation量子化・CPU readbackはない。L1D上限512KiB、L1Eの元の配置（Rows128最大214528 byte）、Combo slice scratch、L3 poolは同じ。増減はprogram codeだけである。

| family | 旧bin byte | 新bin byte | 旧loader byte | 新loader byte | 差byte |
| --- | --- | --- | --- | --- | --- |
| rows128 | 5704 | 6112 | 6144 | 6400 | 256 |
| combo4 | 5228 | 5796 | 5632 | 6144 | 512 |

設定固定の実LLMは対応programを置換する。同一workspaceで旧新を切り替えるAPI試験は両programを保持する。総HBMの全profiler測定はしていない。

## 安全性と作業中の問題

有効なguard実行は60件、全rc0、最高60.0℃。80℃停止・abort latch・他利用者の確認を維持し、両PD RUNNING/no error、errtaskは開始時と一致。driver/firmware/clock/OS変更なし。

最初のsmokeは2048token fixtureをctx256へ渡したホスト入力条件ミスでrc2となり、GEMV/GEMM実行前に停止した。128token fixtureへ修正した。未初期化の新parent indexを巨大な削除差分として自動保存したため空き容量もなくなったが、差分はSHA一致の完全コピーを手元へ退避してから実機の重複だけを除いた。新parentはsparse indexを初期化してcleanへ戻した。ラッチ移動・再開は自動承認レビューで一度拒否されたため、ユーザーの明示許可後にreviewed履歴として保持し、既知正常小テスト後に再開した。デバイスresetは不要だった。過去の実験や0.7.9 source/buildは削除していない。

## 設定と再現

追加設定はKUNLUN_Q4_GEMM_OWNERS=1。0/未設定で0.7.9 GEMMへ戻る。GGML_KUNLUN_COMMON_Q4=1が前提。GEMVはKUNLUN_Q4_MAC_DMA=1/KUNLUN_Q4_NMAJOR=1を維持する。実際のargv/envはlogs/restricted-llm-*.jsonに保存した。

~~~sh
cd /home/unagidojyou/kunlun-k200/results/gemm-next-20261029
python3 - <<'PY'
import json,runner
m=json.load(open("logs/restricted-llm-7b-a1-i1.json"))
runner.run("reproduce-pp",m["command"],m["environment"],600)
PY
~~~

ビルドはenv.shを読み、独立buildへcmake -DCMAKE_BUILD_TYPE=Release -DKUNLUN_BUILD_BENCHMARKS=OFFで設定する。今回のビルドでは旧生成物を再利用し、新しい2カーネルだけをソースからコンパイルした。採用元ソースはkernels/common-q4/gemm-m128-owners.hと2つのwrapper、host選択はsrc/q4-common.inc。公開APIとGGML演算コードは変更していない。

## 次の候補

EWの演算量・K32 scale/累積順序は残る。M64やWideへ専用化を広げる前に実PPでの利用比率を確認する。係数取得とEW実行の重畳やtile間の再充填は候補だが、所有権と完了条件が確認できる範囲で扱う。今回の小さなEW-flat効果だけを理由に方式を増やさなかった。

## 変更ファイルとGit

- Library: CMakeLists.txt/src/k200.cppを0.7.10へ更新、src/q4-common.incへopt-in選択を追加。
- Library: kernels/common-q4/gemm-m128-owners.hとRows128/Combo4 wrapperの2本、docs/q4-gemm-owners.mdを追加。
- llama.cpp: 最小library版を0.7.10へ更新し、vendor/KunlunK200のgitlinkを下記library commitへ固定。GGML runtime演算は変更していない。

Library commit: 74b0073b803153e6a41e0e86668cd52295a86cd9。両方の作業branchはcodex/k200-gemm-next-20261029。Parentの最終commitはdelivery-summary.jsonに記録する。空き容量のため、新parentのsubmodule checkoutは未初期化だが、index/commit内のgitlinkはlibrary HEADと一致させた。

保護した比較元はlibrary a0e02f78d3b93aa3434fca8a651b041de14669df、parent c65817afcdb232ca4b7f0808a20590d3ce0fbf08。実測exeはruntime commit 0f64f3aa83f8ec8a787c7f6858e2436f3cc38a08の既存build（同じruntime、以後は文書/工具/build設定変更）を再利用し、llamaの再build差を入れていない。llama-benchおよび選択DSO・新2kernelの対応はrestricted-manifest.jsonに保存する。旧kernel87本は既知正常0.7.9から再利用した。

~~~sh
R=/home/unagidojyou/kunlun-k200
export XRE=$R/external/xre XDNN=$R/external/xdnn XTDK_DIR=$R/external/xtdk
# ソースからの通常build例（今回の試験では旧87本を再利用して新2本のみ生成）
cmake -S $R/worktrees/gemm-next-20261029-lib -B $R/build/gemm-owners-rebuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DKUNLUN_BUILD_BENCHMARKS=OFF -DCMAKE_INSTALL_PREFIX=$R/results/gemm-owners-rebuild/install
cmake --build $R/build/gemm-owners-rebuild -j4
cmake --install $R/build/gemm-owners-rebuild
# 後日parentをbuildする場合は、対応libraryをimportしてsubmoduleを初期化するか、system libraryで設定する
cmake -S $R/worktrees/gemm-next-20261029-llama -B $R/build/gemm-owners-llama -G Ninja -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=ON -DCMAKE_PREFIX_PATH=$R/results/gemm-next-20261029/install
cmake --build $R/build/gemm-owners-llama -j4 --target llama-bench
~~~

生logは最小の比較/精度記録と試作sourceのみを成果物へ含め、SDK、model、exe、DSO、kernel bin、raw logits、巨大なホストsetup差分は含めない。GitHub push・PR・main/master反映は行っていない。
