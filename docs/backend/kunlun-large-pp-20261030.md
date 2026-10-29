# PP CPU fallbackの削減（0.7.11 opt-in、2026-10-30 JST）

## 結果

0.7.10を基準に、KUNLUN_LARGE_PP=1を追加。7B PP512/ubatch512は約5.9%、TinyLlamaは約25.8%改善した。既定OFF。Q4のweight ABI v1、Row A8、Lease generation、GEMV/GEMM、SwiGLU、既存Flash kernelは変更していない。新しいdevice kernelは追加していない。

## 原因と実装

- Flashのlibrary支持判定にM>256拒否がある。また、query全体のscore/部分和を計上する16MiB workspace制約により、7BではM256でも拒否され得る。kernelを大きくするのではなく、Qのaxis1・出力のaxis2・maskのaxis1を128query単位で同時にずらす。K/Vとmaskのhead/batch strideは維持する。maskは元の絶対位置を表すため、chunk内の位置に作り直さない。
- 全tensor/pointer/alias検証後、既存の完了待ち付きFlash APIを再帰mutex内で順に使用する。workspace/config/packは再利用。FP16 Q変換、Online Softmax、GQA、scale、ALiBi、出力順は旧演算のまま。flag有効時のquery上限は1024、D64/128・mask/type/stride/broadcast/SRAM・各chunkのworkspace制約は引き続き検査する。
- GGMLのRoPE/CONTは、kernel未対応ではなく「query256超のAttention一式をCPUへ置く」配置規則が主因だった。新flagとFlash有効時だけ、既存のRoPE/CONT個別検査を使う。RoPEは連続要素行の明示検査も追加した。in-placeは同じlayoutの完全aliasだけ。位置IDの範囲確認の小さなD2Hは従来通り残す。
- kv_max_set_rowsのwriterはlocal ids[512]にI64で256行しか置けない。257..1024行を256以下へ分ける。1行tailを作らずTG専用pending最大値処理に入らないようにする。host targetsは順序ではなく集合なので、全chunk終了までmutexで暫定metadataを隠す。index検証、F16丸め後のfinite検出、poison保持は省略しない。

## CPU配置・転送・graph

7B PP512/ubatch512を各1回、既存observerを集計出力だけへ絞って診断。eval callbackや強制kernel同期は追加していない。ただしobserverのCPU費用はあるため、以下の診断時間を通常token/sへ混ぜず、各時間を加算しない。

| PPの実行対象 | 旧CPU回数 | 新CPU回数 | 新K200回数 |
| --- | --- | --- | --- |
| FLASH_ATTN_EXT | 32 | 0 | 32（内側128query x 4） |
| ROPE | 64 | 0 | 64 |
| CONT | 0 | 0 | 0（このFlash graphでは独立CONTなし） |
| layer K/V SET_ROWS | 0 | 0 | 64 |

CPU graph区間131 -> 35、K200 graph区間130 -> 34、合計261 -> 69。個別op数とgraph区間数は同じではない。CONT支持拡張は単体でbyte一致を確認したが、このモデルgraphでCPU CONTを減らしたとは主張しない。

| 転送 | 旧MB | 新MB | API回数 | 診断API inclusive ms |
| --- | --- | --- | --- | --- |
| H2D | 1244.303 | 708.010 | 180 -> 194 | 198.12 -> 111.61 |
| D2H | 2220.202 | 1415.028 | 323 -> 323 | 346.13 -> 221.43 |

MBは10^6 byte。Single PD0のP2Pは0。runtime APIの転送量であり、device内部のHBM/DMA走査をD2Hへ換算した値ではない。

同じPP512+短いTG/state診断全体で、Activation reuseは544 -> 576、量子化chunk数1274 -> 1146、量子化launch数2558 -> 2430、所有workspaceは両方2134016 byte。group数384は同じ。graph分断が減って共有可能範囲が広がった結果であり、Leaseの同一性・世代・寿命条件は緩めていない。この累積値はPP単独の件数ではない。

## 通常性能

Single PD0、PP512/TG128、batch512、ubatch128/512、threads6、KV F16。COMMON_Q4=1、ACT_SHARE=1、MAC_DMA=1、NMAJOR=1、GEMM_OWNERS=1、SwiGLU=0、Attention/Flash/RoPE有効、その他auto。新しい同一exe/DSOでLARGE_PPのみ0/1切替。各方式2標本のABBAまたはBAAB、llama-bench r1。中央値±標本SD、token/s。診断preload・追加同期は使っていない。

| モデル | ubatch | 処理 | 旧 | 新 | 差 |
| --- | --- | --- | --- | --- | --- |
| 7b | 512 | PP | 74.100 ± 0.149 | 78.446 ± 0.041 | +5.87% |
| 7b | 512 | TG | 8.105 ± 0.027 | 8.118 ± 0.026 | +0.16% |
| 7b | 128 | PP | 78.132 ± 0.029 | 78.492 ± 0.668 | +0.46% |
| 7b | 128 | TG | 8.031 ± 0.077 | 7.991 ± 0.089 | -0.50% |
| tiny | 512 | PP | 247.706 ± 0.481 | 311.608 ± 0.456 | +25.80% |
| tiny | 512 | TG | 25.259 ± 0.995 | 25.403 ± 1.211 | +0.57% |
| tiny | 128 | PP | 307.654 ± 0.048 | 314.179 ± 11.068 | +2.12% |
| tiny | 128 | TG | 26.337 ± 0.006 | 26.013 ± 0.087 | -1.23% |

n2の小差を改善とは判定しない。ubatch128は出力bitwise一致で、PP/TGに明確な構造的回帰は見られない。ubatch512が常に128より速いという結果ではなく、7Bは同程度、Tinyはなお128が速い。batch既定値は変更しない。平均・全標本はsummary.jsonへ保存した。

### Tiny TGだけの追加確認

混合PP/TG系列でubatch128のTiny TGが-1.23%だったため、この条件のみPP0/TG128、batch512/ubatch128、各方式4標本のABBA（各process r2）で再確認した。旧平均25.890 / 中央値26.141 / SD0.689、新平均25.882 / 中央値26.029 / SD0.583 token/s。平均差-0.03%、中央値差-0.43%。差はこの系列のばらつきより小さく、一定のTG低下は再現しなかった。元の-1.23%の結果も上表に残す。サブ%の無回帰を精密に証明したという主張はしない。

commit確定後にhostを再buildし、7B PP512から4位置生成の最終logitsが実測時とbitwise一致することも確認した。

## 数値精度・KV・Lease

query513のtail、prefix付きS768、GQA Q32/KV4、通常Q32/KV32、batch broadcast、F16/F32 mask、padding、ALiBi、全mask行を確認。手動で旧APIを128queryずつ呼んだ結果と全出力bitwise一致。sampleした独立F64参照の最大NMSEは約1.63e-8。RoPEはF32/F16、標準/NEOX、strided row、in-placeを旧小query呼出しとbitwise比較。CONTはCPUのview gather期待値とbyte一致。guardと入力不変も確認した。

KV513行はI64/F32とI32/F16を、順序が異なるhost target集合で確認。cache最大値はSDK参照と不一致0。途中chunkにNaNを入れた後、有限値で上書きしてもpoisonによる安全なfallbackが残ることを確認した。

実LLMでは両modeの7B state保存/復元後next logitsがbitwise一致。新ubatch512の7BでPP512 -> TG7を同contextで2回繰返し、全8位置logitsがbitwise一致。通常prefix試験ではhits384、seeds64、unverified0、mismatch0だった。stateをCPUから復元した場合の保守的metadata無効化はそのまま維持する。

ubatch128の両モデルはflag0/1で全4位置logits bitwise一致。新しい7B ubatch512もACT_SHARE=0/1でbitwise一致した。

一方、CPU Flashとのlogitsはbitwise一致しない。既存F16/I16 FlashとCPUの演算差、KVの範囲/最大値、後続Row A8の閾値を含む。全差を一つの原因へ帰属させる切分けはしていない。以下はPP512直後の同一入力1位置のNMSE/KLと、同じgreedy履歴の4位置のargmax一致。品質同等の主張やPPL評価ではない。

| 比較 | 最初のlogits NMSE | KL(reference,new) | greedy一致 |
| --- | --- | --- | --- |
| 7b-cpu512-vs-new512 | 0.00117035 | 0.00725023 | 4/4 |
| 7b-xpu128-vs-new512 | 0.000825213 | 0.00692889 | 4/4 |
| tiny-cpu512-vs-new512 | 0.00141797 | 0.00495111 | 4/4 |
| tiny-xpu128-vs-new512 | 0.00135905 | 0.00243959 | 4/4 |

## 残るCPU fallback・次の候補

EmbeddingのGET_ROWS（Q4_0）、入力のDUP/F32 SET_ROWS、無効化しているFFN GLU32回、Q6_Kのresult_outputがCPUに残る。入力のCPU SET_ROWS1回と、K/VのGPU SET_ROWS64回は区別した。現状の大きい転送はGLUのgate/up往復。次は既存PP限定SwiGLU設定の再評価、またはQ6_K出力層。大query Flash内で繰返すK/V findmax/packの共有も候補だが、今回の変更へ混ぜていない。

## 安全性・範囲

成功guard実行34件、最高59.0℃。最初の単体で「初回seedをhitと要求」したテスト誤りによりhost assertion停止が1回あった。初回seed/次回hitへ修正し、両PDとerrtask不変を確認、ラッチをreviewed履歴として保持して既知正常小試験から再開した。device fault・resetなし。診断observerのcompile用define/includeも調整した。

80℃guard、abort latch、利用者確認を維持。driver/firmware/clock/OS変更なし。旧0.7.10 source/buildを保持。全context/全Split/perplexityは追加していない。新mask/sinks/soft cap等の未対応条件は元のCPU経路へ戻る。量子化方式・演算kernel・weight再pack・二重常駐の追加はない。

## Git・ビルド・実行

Library: 9d399d41bc9a0072aab90c64a074e0883165bbc2（0.7.11）。両branchはcodex/k200-pp-fallback-20261030。parentの確定commitとpinはdelivery-summary.jsonへ記録する。主な変更はlibraryのsrc/flash.inc、src/kv-max.inc、RoPE stride検査、private flag、bench/large-pp-test.cppとdocs、GGMLのattention.inc支持規則とFlash CPU理由log。公開API signatureとQ4/kernel sourceは不変。

~~~sh
R=/home/unagidojyou/kunlun-k200
export XRE=$R/external/xre XDNN=$R/external/xdnn XTDK_DIR=$R/external/xtdk
cmake -S $R/worktrees/pp-fallback-20261030-lib -B $R/build/pp-lib-rebuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DKUNLUN_BUILD_BENCHMARKS=OFF -DCMAKE_INSTALL_PREFIX=$R/results/pp-lib-rebuild/install
cmake --build $R/build/pp-lib-rebuild -j4
cmake --install $R/build/pp-lib-rebuild
cmake -S $R/worktrees/pp-fallback-20261030-llama -B $R/build/pp-llama-rebuild -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=ON -DCMAKE_PREFIX_PATH=$R/results/pp-lib-rebuild/install
cmake --build $R/build/pp-llama-rebuild -j4 --target llama-bench kunlun-llm-smoke
~~~

仕上げで追加した説明コメント1行により、SDKエラー表示の行番号定数のみが変わった。測定時libraryとの逆アセンブル照合で確認し、binary-comment-proof.jsonへ保存した。通常の演算命令・命令数・位置は不変。

今回の開発buildは変更のない89kernelを0.7.10から再利用し、host部を再buildした。通常source buildではそれらも生成される。最小の再測定は既存thermal guardを使う以下の方法。0/1だけ切り替えれば同一条件になる。

~~~sh
cd /home/unagidojyou/kunlun-k200/results/pp-fallback-20261030
python3 - <<'PY'
import json,runner
m=json.load(open('logs/pilot-7b-u512-m1-i1.json'))
m['environment']['KUNLUN_LARGE_PP']='1' # '0' is the old path
runner.run('manual-pp512',m['command'],m['environment'],300)
PY
~~~

KUNLUN_LARGE_PP_STATS=1でFlashの論理batch/chunk件数、GGML_KUNLUN_DIAGNOSTICS=1で非対応理由を確認できる。通常性能測定ではLD_PRELOAD/TRACE/verify等を外す。SDK/model/bin/raw logitsや巨大traceはGitに含めず、最小のJSON/logと差分bundleのみ配布する。GitHub push・PR・main/master更新は行っていない。
