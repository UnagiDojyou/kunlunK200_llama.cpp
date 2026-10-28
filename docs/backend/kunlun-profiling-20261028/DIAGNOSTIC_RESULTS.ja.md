# 詳細診断データ付録

全て追加同期なしの全runtime observer実行。Single PD0、batch/ubatch=128/128、SwiGLU=1、各3試行。PPは入力tokenあたり、TGは1 decodeあたり。通常のtokens/sとは区別する。TGのobserver費用がTiny約18%/7B約8%あるため、この付録の割合・Amdahlは補助値。主時間割合と判断には[軽量内訳](LIGHT_RESULTS.ja.md)を使用する。kernel段階・bytes・callsをここで詳しく扱う。

## tiny / prefix 128 / PP

1試行の単位数: 128。以下の「区間計」は128単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|446.128|3.4854 ± 0.0123|100.00|1|
|Q4 GEMV/GEMM|307.057|2.3989 ± 0.0062|68.83|154|
|Elementwise/residual XPU|39.388|0.3077 ± 0.0033|8.83|89|
|FlashAttention XPU|29.045|0.2269 ± 0.0004|6.51|22|
|SwiGLU XPU|22.279|0.1741 ± 0.0007|4.99|22|
|RMSNorm XPU|17.619|0.1376 ± 0.0009|3.95|45|
|RoPE XPU|14.118|0.1103 ± 0.0017|3.16|44|
|KV SET_ROWS XPU|11.135|0.0870 ± 0.0013|2.50|44|
|Q6_K output CPU graph|2.010|0.0157 ± 0.0006|0.45|1|
|Remaining host/scheduler/measurement|1.268|0.0099 ± 0.0009|0.28|1|
|Scheduler split/allocation|0.611|0.0048 ± 0.0000|0.14|2|
|External D2H|0.470|0.0037 ± 0.0002|0.11|135|
|XPU metadata|0.443|0.0035 ± 0.0001|0.10|264|
|External H2D|0.318|0.0025 ± 0.0002|0.07|141|
|Embedding CPU graph|0.306|0.0024 ± 0.0005|0.07|1|
|External backend synchronization|0.042|0.0003 ± 0.0001|0.01|25|
|Output selection GET_ROWS CPU graph|0.010|0.0001 ± 0.0000|0.00|1|
|External runtime wait|0.007|0.0001 ± 0.0000|0.00|1274|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMM MAC, block-scale correction and device DMA|1.4069|40.37|
|Activation quantization|0.9401|26.97|
|GEMM FP32 slice join|0.0316|0.91|
|Remaining Q4 API host/observer|0.0060|0.17|
|GEMV MAC and device DMA|0.0043|0.12|
|GEMV FP32 scale/sum|0.0008|0.02|
|Completed-queue wait|0.0008|0.02|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|16.579|0.01294|1.05|
|H2D|8.926|0.00965|1.10|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||8.000|0.00206|
|D2H|attn_out-21||8.000|0.00177|
|D2H|l_out-20||8.000|0.00185|
|H2D|KUNLUN0#attn_inp_kq_mask#0||0.500|0.00016|
|D2H|KUNLUN0#attn_inp_k_idxs#0||0.172|0.00299|
|D2H|KUNLUN0#attn_inp_v_idxs#0||0.172|0.00117|
|H2D|KUNLUN0#node_678#0||0.062|0.00006|
|H2D|KUNLUN0#node_679#0||0.062|0.00005|
|D2H|result_norm||0.062|0.00006|
|H2D|KUNLUN0#attn_inp_k_idxs#0||0.008|0.00005|
|H2D|KUNLUN0#attn_inp_v_idxs#0||0.008|0.00005|
|H2D|KUNLUN0#inp_pos#0||0.004|0.00005|
|D2H|(internal API) Qcur-0||0.004|0.00006|
|D2H|(internal API) Kcur-0||0.004|0.00006|
|H2D|(internal API) cache_k_l0 (view)||0.004|0.00005|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|3.48538|
|api_union_ms|2.61655|
|wait_union_ms|3.31753|
|launch_union_ms|0.01616|
|main_thread_cpu_ms|3.41648|
|other_worker_cpu_ms|0.07158|
|wait_calls|9.95312|
|launch_calls|7.15625|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|6.98|0.02732|
|runtime.launch_config.|7.16|0.00021|
|runtime.launch.|7.16|0.01616|
|runtime.wait.|9.95|3.31753|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|0.70|
|alias候補回数|0.70|
|alias候補payload KiB|680.312|
|alias候補 enqueue→既存wait完了 ms|0.05710|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|0.08700|
|library_union_ms|0.07904|
|index_get_union_ms|0.00585|
|layout_index_validation_host_residual_ms|0.00211|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## tiny / prefix 128 / TG

1試行の単位数: 31。以下の「区間計」は31単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|1537.803|49.6065 ± 0.3392|100.00|31|
|Q4 GEMV/GEMM|996.515|32.1457 ± 0.3404|64.81|4774|
|Elementwise/residual XPU|99.297|3.2031 ± 0.0078|6.46|2759|
|KV SET_ROWS XPU|94.095|3.0353 ± 0.0053|6.12|1364|
|RoPE XPU|88.658|2.8599 ± 0.0011|5.77|1364|
|FlashAttention XPU|68.421|2.2071 ± 0.0030|4.45|682|
|Q6_K output CPU graph|58.518|1.8877 ± 0.0064|3.81|31|
|RMSNorm XPU|47.834|1.5430 ± 0.0507|3.11|1395|
|Remaining host/scheduler/measurement|36.877|1.1896 ± 0.7161|2.39|31|
|SwiGLU XPU|26.210|0.8455 ± 0.0231|1.70|682|
|XPU metadata|10.608|0.3422 ± 0.0015|0.69|8184|
|Embedding CPU graph|6.234|0.2011 ± 0.0043|0.41|31|
|External H2D|1.768|0.0570 ± 0.0017|0.11|4311|
|External backend synchronization|0.978|0.0316 ± 0.0011|0.06|775|
|External D2H|0.676|0.0218 ± 0.0000|0.04|4185|
|Scheduler split/allocation|0.632|0.0204 ± 0.0006|0.04|2|
|Output selection GET_ROWS CPU graph|0.248|0.0080 ± 0.0002|0.02|31|
|External runtime wait|0.233|0.0075 ± 0.0002|0.02|44797|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMV MAC and device DMA|17.5971|35.47|
|Activation quantization|9.6545|19.46|
|GEMV FP32 scale/sum|3.1985|6.45|
|Remaining Q4 API host/observer|0.7983|1.61|
|Completed-queue wait|0.0986|0.20|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|24.688|0.92489|135.00|
|H2D|60.279|0.95272|139.06|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||8.000|0.01072|
|D2H|attn_out-21||8.000|0.00737|
|D2H|l_out-20||8.000|0.00711|
|H2D|KUNLUN0#node_678#0||8.000|0.00733|
|H2D|KUNLUN0#node_679#0||8.000|0.00717|
|D2H|result_norm||8.000|0.00734|
|H2D|KUNLUN0#attn_inp_kq_mask#0||0.500|0.00684|
|H2D|(internal API) cache_k_l0 (view)||0.500|0.00675|
|H2D|(internal API) cache_v_l0 (view)||0.500|0.00709|
|H2D|(internal API) cache_k_l1 (view)||0.500|0.00690|
|H2D|(internal API) cache_v_l1 (view)||0.500|0.00670|
|H2D|(internal API) cache_k_l2 (view)||0.500|0.00672|
|H2D|(internal API) cache_v_l2 (view)||0.500|0.00673|
|H2D|(internal API) cache_k_l3 (view)||0.500|0.00676|
|H2D|(internal API) cache_v_l3 (view)||0.500|0.00670|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|49.60654|
|api_union_ms|33.46263|
|wait_union_ms|31.29091|
|launch_union_ms|2.34013|
|main_thread_cpu_ms|49.27586|
|other_worker_cpu_ms|8.54470|
|wait_calls|1445.06452|
|launch_calls|1369.00000|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|1347.00|1.61523|
|runtime.launch_config.|1369.00|0.03142|
|runtime.launch.|1369.00|2.34013|
|runtime.wait.|1445.06|31.29091|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|90.00|
|alias候補回数|90.00|
|alias候補payload KiB|720.000|
|alias候補 enqueue→既存wait完了 ms|0.95568|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|3.03531|
|library_union_ms|2.35767|
|index_get_union_ms|0.50119|
|layout_index_validation_host_residual_ms|0.17646|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## tiny / prefix 2048 / PP

1試行の単位数: 2048。以下の「区間計」は2048単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|8127.388|3.9685 ± 0.0177|100.00|16|
|Q4 GEMV/GEMM|4915.342|2.4001 ± 0.0032|60.48|2464|
|FlashAttention XPU|1430.305|0.6984 ± 0.0003|17.60|352|
|Elementwise/residual XPU|630.417|0.3078 ± 0.0018|7.76|1424|
|SwiGLU XPU|356.280|0.1740 ± 0.0003|4.38|352|
|RMSNorm XPU|282.429|0.1379 ± 0.0004|3.48|720|
|RoPE XPU|225.784|0.1102 ± 0.0007|2.78|704|
|KV SET_ROWS XPU|178.072|0.0869 ± 0.0005|2.19|704|
|Remaining host/scheduler/measurement|45.497|0.0222 ± 0.0220|0.56|16|
|Q6_K output CPU graph|31.744|0.0155 ± 0.0001|0.39|16|
|XPU metadata|7.627|0.0037 ± 0.0001|0.09|4224|
|External D2H|7.387|0.0036 ± 0.0000|0.09|2160|
|External H2D|5.782|0.0028 ± 0.0000|0.07|2233|
|Scheduler split/allocation|5.707|0.0028 ± 0.0005|0.07|16|
|Embedding CPU graph|4.031|0.0020 ± 0.0001|0.05|16|
|External backend synchronization|0.714|0.0003 ± 0.0000|0.01|400|
|Output selection GET_ROWS CPU graph|0.154|0.0001 ± 0.0000|0.00|16|
|External runtime wait|0.116|0.0001 ± 0.0000|0.00|20361|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMM MAC, block-scale correction and device DMA|1.4070|35.46|
|Activation quantization|0.9405|23.70|
|GEMM FP32 slice join|0.0316|0.80|
|Remaining Q4 API host/observer|0.0061|0.15|
|GEMV MAC and device DMA|0.0043|0.11|
|GEMV FP32 scale/sum|0.0009|0.02|
|Completed-queue wait|0.0008|0.02|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|16.579|0.01280|1.05|
|H2D|10.674|0.00986|1.09|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||8.000|0.00196|
|D2H|attn_out-21||8.000|0.00177|
|D2H|l_out-20||8.000|0.00178|
|H2D|KUNLUN0#attn_inp_kq_mask#0||2.250|0.00059|
|D2H|KUNLUN0#attn_inp_k_idxs#0||0.172|0.00295|
|D2H|KUNLUN0#attn_inp_v_idxs#0||0.172|0.00117|
|H2D|KUNLUN0#node_678#0||0.062|0.00006|
|H2D|KUNLUN0#node_679#0||0.062|0.00006|
|D2H|result_norm||0.062|0.00006|
|H2D|KUNLUN0#attn_inp_k_idxs#0||0.008|0.00006|
|H2D|KUNLUN0#attn_inp_v_idxs#0||0.008|0.00006|
|H2D|KUNLUN0#inp_pos#0||0.004|0.00005|
|D2H|(internal API) Qcur-0||0.004|0.00006|
|D2H|(internal API) Kcur-0||0.004|0.00006|
|H2D|(internal API) cache_k_l0 (view)||0.004|0.00005|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|3.96845|
|api_union_ms|3.08884|
|wait_union_ms|3.78202|
|launch_union_ms|0.01848|
|main_thread_cpu_ms|3.88441|
|other_worker_cpu_ms|0.07078|
|wait_calls|9.94189|
|launch_calls|7.92969|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|7.76|0.02732|
|runtime.launch_config.|7.93|0.00023|
|runtime.launch.|7.93|0.01848|
|runtime.wait.|9.94|3.78202|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|0.70|
|alias候補回数|0.70|
|alias候補payload KiB|680.312|
|alias候補 enqueue→既存wait完了 ms|0.05703|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|0.08695|
|library_union_ms|0.07896|
|index_get_union_ms|0.00585|
|layout_index_validation_host_residual_ms|0.00215|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## tiny / prefix 2048 / TG

1試行の単位数: 31。以下の「区間計」は31単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|1649.149|53.1984 ± 0.0233|100.00|31|
|Q4 GEMV/GEMM|984.746|31.7660 ± 0.0126|59.71|4774|
|FlashAttention XPU|204.971|6.6120 ± 0.0020|12.43|682|
|Elementwise/residual XPU|98.895|3.1902 ± 0.0042|6.00|2759|
|KV SET_ROWS XPU|94.455|3.0469 ± 0.0051|5.73|1364|
|RoPE XPU|89.137|2.8754 ± 0.0068|5.41|1364|
|Q6_K output CPU graph|58.413|1.8843 ± 0.0123|3.54|31|
|RMSNorm XPU|47.031|1.5171 ± 0.0031|2.85|1395|
|SwiGLU XPU|25.778|0.8316 ± 0.0019|1.56|682|
|Remaining host/scheduler/measurement|24.460|0.7890 ± 0.0099|1.48|31|
|XPU metadata|10.612|0.3423 ± 0.0008|0.64|8184|
|Embedding CPU graph|6.101|0.1968 ± 0.0097|0.37|31|
|External H2D|1.734|0.0559 ± 0.0015|0.11|4311|
|External backend synchronization|1.007|0.0325 ± 0.0005|0.06|775|
|External D2H|0.681|0.0220 ± 0.0003|0.04|4185|
|Scheduler split/allocation|0.650|0.0210 ± 0.0009|0.04|2|
|Output selection GET_ROWS CPU graph|0.238|0.0077 ± 0.0002|0.01|31|
|External runtime wait|0.238|0.0077 ± 0.0003|0.01|44797|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMV MAC and device DMA|17.3891|32.69|
|Activation quantization|9.4883|17.84|
|GEMV FP32 scale/sum|3.1849|5.99|
|Remaining Q4 API host/observer|0.8009|1.51|
|Completed-queue wait|0.0989|0.19|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|24.688|0.92414|135.00|
|H2D|64.281|0.95268|139.06|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||8.000|0.01143|
|D2H|attn_out-21||8.000|0.00759|
|D2H|l_out-20||8.000|0.00709|
|H2D|KUNLUN0#node_678#0||8.000|0.00720|
|H2D|KUNLUN0#node_679#0||8.000|0.00708|
|D2H|result_norm||8.000|0.00729|
|H2D|KUNLUN0#attn_inp_kq_mask#0||4.500|0.00710|
|H2D|(internal API) cache_k_l0 (view)||0.500|0.00677|
|H2D|(internal API) cache_v_l0 (view)||0.500|0.00687|
|H2D|(internal API) cache_k_l1 (view)||0.500|0.00695|
|H2D|(internal API) cache_v_l1 (view)||0.500|0.00685|
|H2D|(internal API) cache_k_l2 (view)||0.500|0.00686|
|H2D|(internal API) cache_v_l2 (view)||0.500|0.00689|
|H2D|(internal API) cache_k_l3 (view)||0.500|0.00672|
|H2D|(internal API) cache_v_l3 (view)||0.500|0.00671|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|53.19835|
|api_union_ms|37.48110|
|wait_union_ms|35.63438|
|launch_union_ms|2.32057|
|main_thread_cpu_ms|52.87536|
|other_worker_cpu_ms|8.52815|
|wait_calls|1445.06452|
|launch_calls|1369.00000|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|1347.00|1.61682|
|runtime.launch_config.|1369.00|0.03089|
|runtime.launch.|1369.00|2.32057|
|runtime.wait.|1445.06|35.63438|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|90.00|
|alias候補回数|90.00|
|alias候補payload KiB|720.000|
|alias候補 enqueue→既存wait完了 ms|0.94659|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|3.04694|
|library_union_ms|2.36353|
|index_get_union_ms|0.50497|
|layout_index_validation_host_residual_ms|0.17845|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## 7b / prefix 128 / PP

1試行の単位数: 128。以下の「区間計」は128単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|1768.967|13.8201 ± 0.0091|100.00|1|
|Q4 GEMV/GEMM|1312.193|10.2515 ± 0.0031|74.18|224|
|KV SET_ROWS XPU|141.424|1.1049 ± 0.0010|7.99|64|
|Elementwise/residual XPU|102.945|0.8043 ± 0.0021|5.82|129|
|SwiGLU XPU|61.051|0.4770 ± 0.0001|3.45|32|
|FlashAttention XPU|52.150|0.4074 ± 0.0001|2.95|32|
|RMSNorm XPU|48.113|0.3759 ± 0.0010|2.72|65|
|RoPE XPU|42.480|0.3319 ± 0.0002|2.40|64|
|Q6_K output CPU graph|3.715|0.0290 ± 0.0004|0.21|1|
|Remaining host/scheduler/measurement|1.626|0.0127 ± 0.0008|0.09|1|
|Scheduler split/allocation|0.871|0.0068 ± 0.0001|0.05|2|
|External D2H|0.771|0.0060 ± 0.0000|0.04|195|
|XPU metadata|0.645|0.0050 ± 0.0010|0.04|384|
|External H2D|0.478|0.0037 ± 0.0001|0.03|75|
|Embedding CPU graph|0.445|0.0035 ± 0.0008|0.03|1|
|External backend synchronization|0.038|0.0003 ± 0.0001|0.00|25|
|Output selection GET_ROWS CPU graph|0.014|0.0001 ± 0.0000|0.00|1|
|External runtime wait|0.008|0.0001 ± 0.0000|0.00|1718|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMM MAC, block-scale correction and device DMA|7.4818|54.14|
|Activation quantization|2.6520|19.19|
|GEMM FP32 slice join|0.0798|0.58|
|GEMV MAC and device DMA|0.0134|0.10|
|Remaining Q4 API host/observer|0.0087|0.06|
|GEMV FP32 scale/sum|0.0021|0.01|
|Completed-queue wait|0.0013|0.01|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|32.879|0.01973|1.52|
|H2D|17.024|0.00746|0.59|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||16.000|0.00319|
|D2H|attn_out-31||16.000|0.00299|
|D2H|l_out-30||16.000|0.00296|
|H2D|KUNLUN0#attn_inp_kq_mask#0||0.500|0.00022|
|D2H|KUNLUN0#attn_inp_k_idxs#0||0.250|0.00427|
|D2H|KUNLUN0#attn_inp_v_idxs#0||0.250|0.00172|
|H2D|KUNLUN0#node_988#0||0.125|0.00008|
|H2D|KUNLUN0#node_989#0||0.125|0.00008|
|D2H|result_norm||0.125|0.00008|
|H2D|KUNLUN0#attn_inp_k_idxs#0||0.008|0.00005|
|H2D|KUNLUN0#attn_inp_v_idxs#0||0.008|0.00005|
|H2D|KUNLUN0#inp_pos#0||0.004|0.00005|
|D2H|(internal API) Qcur-0||0.004|0.00006|
|D2H|(internal API) Kcur-0||0.004|0.00006|
|H2D|(internal API) cache_k_l0 (view)||0.004|0.00005|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|13.82006|
|api_union_ms|10.64522|
|wait_union_ms|13.57246|
|launch_union_ms|0.02526|
|main_thread_cpu_ms|13.42966|
|other_worker_cpu_ms|0.13896|
|wait_calls|13.42188|
|launch_calls|10.35938|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|10.11|0.00648|
|runtime.launch_config.|10.36|0.00032|
|runtime.launch.|10.36|0.02526|
|runtime.wait.|13.42|13.57246|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|1.02|
|alias候補回数|1.02|
|alias候補payload KiB|2000.625|
|alias候補 enqueue→既存wait完了 ms|0.12159|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|1.10488|
|library_union_ms|1.09303|
|index_get_union_ms|0.00885|
|layout_index_validation_host_residual_ms|0.00300|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## 7b / prefix 128 / TG

1試行の単位数: 31。以下の「区間計」は31単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|4553.561|146.8891 ± 0.6953|100.00|31|
|Q4 GEMV/GEMM|3542.850|114.2855 ± 0.6446|77.80|6944|
|FlashAttention XPU|314.378|10.1412 ± 0.0017|6.90|992|
|Elementwise/residual XPU|148.593|4.7933 ± 0.0035|3.26|3999|
|KV SET_ROWS XPU|145.684|4.6995 ± 0.3589|3.20|1984|
|RoPE XPU|119.082|3.8414 ± 0.0057|2.62|1984|
|Q6_K output CPU graph|113.139|3.6496 ± 0.0229|2.48|31|
|RMSNorm XPU|70.568|2.2764 ± 0.0044|1.55|2015|
|SwiGLU XPU|37.691|1.2158 ± 0.0007|0.83|992|
|Remaining host/scheduler/measurement|34.071|1.0991 ± 0.0040|0.75|31|
|XPU metadata|15.458|0.4986 ± 0.0026|0.34|11904|
|Embedding CPU graph|6.673|0.2152 ± 0.0080|0.15|31|
|External H2D|2.026|0.0654 ± 0.0027|0.04|2205|
|External backend synchronization|1.051|0.0339 ± 0.0018|0.02|775|
|External D2H|0.917|0.0296 ± 0.0001|0.02|6045|
|Scheduler split/allocation|0.856|0.0276 ± 0.0001|0.02|2|
|Output selection GET_ROWS CPU graph|0.271|0.0088 ± 0.0002|0.01|31|
|External runtime wait|0.253|0.0082 ± 0.0005|0.01|60981|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMV MAC and device DMA|83.5021|56.85|
|Activation quantization|16.4835|11.22|
|GEMV FP32 scale/sum|11.6349|7.92|
|Remaining Q4 API host/observer|1.3413|0.91|
|Completed-queue wait|0.1440|0.10|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|49.250|1.35505|195.00|
|H2D|80.540|0.50429|71.13|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||16.000|0.01553|
|D2H|attn_out-31||16.000|0.01003|
|D2H|l_out-30||16.000|0.00950|
|H2D|KUNLUN0#node_988#0||16.000|0.01017|
|H2D|KUNLUN0#node_989#0||16.000|0.01000|
|D2H|result_norm||16.000|0.01004|
|H2D|KUNLUN0#attn_inp_kq_mask#0||0.500|0.00687|
|H2D|(internal API) cache_k_l0 (view)||0.500|0.00694|
|H2D|(internal API) cache_v_l0 (view)||0.500|0.00689|
|H2D|(internal API) cache_k_l1 (view)||0.500|0.00684|
|H2D|(internal API) cache_v_l1 (view)||0.500|0.00685|
|H2D|(internal API) cache_k_l2 (view)||0.500|0.00687|
|H2D|(internal API) cache_v_l2 (view)||0.500|0.00671|
|H2D|(internal API) cache_k_l3 (view)||0.500|0.00684|
|H2D|(internal API) cache_v_l3 (view)||0.500|0.00672|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|146.88907|
|api_union_ms|123.11638|
|wait_union_ms|120.96359|
|launch_union_ms|3.38867|
|main_thread_cpu_ms|145.68376|
|other_worker_cpu_ms|17.29318|
|wait_calls|1967.12903|
|launch_calls|1989.00000|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|1957.00|0.74597|
|runtime.launch_config.|1989.00|0.04570|
|runtime.launch.|1989.00|3.38867|
|runtime.wait.|1967.13|120.96359|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|130.00|
|alias候補回数|130.00|
|alias候補payload KiB|2080.000|
|alias候補 enqueue→既存wait完了 ms|1.45280|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|4.69949|
|library_union_ms|3.49552|
|index_get_union_ms|0.95048|
|layout_index_validation_host_residual_ms|0.25349|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## 7b / prefix 2048 / PP

1試行の単位数: 2048。以下の「区間計」は2048単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|30137.063|14.7154 ± 0.0281|100.00|16|
|Q4 GEMV/GEMM|21037.690|10.2723 ± 0.0247|69.81|3584|
|FlashAttention XPU|2618.078|1.2784 ± 0.0003|8.69|512|
|KV SET_ROWS XPU|2264.695|1.1058 ± 0.0007|7.51|1024|
|Elementwise/residual XPU|1655.199|0.8082 ± 0.0030|5.49|2064|
|SwiGLU XPU|977.943|0.4775 ± 0.0004|3.24|512|
|RMSNorm XPU|771.952|0.3769 ± 0.0008|2.56|1040|
|RoPE XPU|680.962|0.3325 ± 0.0008|2.26|1024|
|Q6_K output CPU graph|59.232|0.0289 ± 0.0001|0.20|16|
|Remaining host/scheduler/measurement|25.972|0.0127 ± 0.0002|0.09|16|
|External D2H|12.374|0.0060 ± 0.0000|0.04|3120|
|XPU metadata|11.704|0.0057 ± 0.0000|0.04|6144|
|External H2D|8.486|0.0041 ± 0.0000|0.03|1147|
|Scheduler split/allocation|7.117|0.0035 ± 0.0000|0.02|16|
|Embedding CPU graph|4.698|0.0023 ± 0.0000|0.02|16|
|External backend synchronization|0.610|0.0003 ± 0.0000|0.00|400|
|Output selection GET_ROWS CPU graph|0.212|0.0001 ± 0.0000|0.00|16|
|External runtime wait|0.139|0.0001 ± 0.0000|0.00|27435|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMM MAC, block-scale correction and device DMA|7.4859|50.87|
|Activation quantization|2.6678|18.13|
|GEMM FP32 slice join|0.0803|0.55|
|GEMV MAC and device DMA|0.0133|0.09|
|Remaining Q4 API host/observer|0.0090|0.06|
|GEMV FP32 scale/sum|0.0022|0.01|
|Completed-queue wait|0.0012|0.01|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|32.879|0.01964|1.52|
|H2D|18.770|0.00762|0.56|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||16.000|0.00327|
|D2H|attn_out-31||16.000|0.00299|
|D2H|l_out-30||16.000|0.00298|
|H2D|KUNLUN0#attn_inp_kq_mask#0||2.250|0.00055|
|D2H|KUNLUN0#attn_inp_k_idxs#0||0.250|0.00423|
|D2H|KUNLUN0#attn_inp_v_idxs#0||0.250|0.00171|
|H2D|KUNLUN0#node_988#0||0.125|0.00008|
|H2D|KUNLUN0#node_989#0||0.125|0.00008|
|D2H|result_norm||0.125|0.00008|
|H2D|KUNLUN0#attn_inp_k_idxs#0||0.008|0.00005|
|H2D|KUNLUN0#attn_inp_v_idxs#0||0.008|0.00005|
|H2D|KUNLUN0#inp_pos#0||0.004|0.00007|
|D2H|(internal API) Qcur-0||0.004|0.00006|
|D2H|(internal API) Kcur-0||0.004|0.00006|
|H2D|(internal API) cache_k_l0 (view)||0.004|0.00005|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|14.71536|
|api_union_ms|11.53677|
|wait_union_ms|14.44698|
|launch_union_ms|0.02822|
|main_thread_cpu_ms|14.29877|
|other_worker_cpu_ms|0.13925|
|wait_calls|13.39600|
|launch_calls|11.48438|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|11.23|0.00532|
|runtime.launch_config.|11.48|0.00036|
|runtime.launch.|11.48|0.02822|
|runtime.wait.|13.40|14.44698|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|1.02|
|alias候補回数|1.02|
|alias候補payload KiB|2000.625|
|alias候補 enqueue→既存wait完了 ms|0.12303|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|1.10581|
|library_union_ms|1.09391|
|index_get_union_ms|0.00887|
|layout_index_validation_host_residual_ms|0.00303|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## 7b / prefix 2048 / TG

1試行の単位数: 31。以下の「区間計」は31単位分の3試行平均。「ms/単位」の±は試行間の標本標準偏差。

|排他的区間|区間計 ms|ms/単位|割合 %|区間内呼出数 (inclusive)|
|---|---|---|---|---|
|TOTAL|5202.566|167.8247 ± 2.9573|100.00|31|
|Q4 GEMV/GEMM|3525.683|113.7317 ± 0.0340|67.78|6944|
|FlashAttention XPU|909.101|29.3259 ± 0.0289|17.48|992|
|Elementwise/residual XPU|202.452|6.5307 ± 3.0210|3.87|3999|
|KV SET_ROWS XPU|162.412|5.2391 ± 0.0199|3.12|1984|
|RoPE XPU|119.596|3.8579 ± 0.0121|2.30|1984|
|Q6_K output CPU graph|112.829|3.6397 ± 0.0113|2.17|31|
|RMSNorm XPU|70.724|2.2814 ± 0.0076|1.36|2015|
|SwiGLU XPU|37.865|1.2215 ± 0.0027|0.73|992|
|Remaining host/scheduler/measurement|34.512|1.1133 ± 0.0344|0.66|31|
|XPU metadata|15.395|0.4966 ± 0.0021|0.30|11904|
|Embedding CPU graph|6.489|0.2093 ± 0.0125|0.12|31|
|External H2D|2.054|0.0663 ± 0.0023|0.04|2205|
|External backend synchronization|1.115|0.0360 ± 0.0019|0.02|775|
|External D2H|0.911|0.0294 ± 0.0004|0.02|6045|
|Scheduler split/allocation|0.883|0.0285 ± 0.0007|0.02|2|
|External runtime wait|0.273|0.0088 ± 0.0017|0.01|65941|
|Output selection GET_ROWS CPU graph|0.272|0.0088 ± 0.0001|0.01|31|

呼出数は元scopeの回数であり、包含される転送・waitの回数を別表と合計しない。

### Q4の既存完了waitを境界とする段階時間

|段階|ms/単位|process比 %|
|---|---|---|
|GEMV MAC and device DMA|83.5264|49.77|
|Activation quantization|16.4559|9.81|
|GEMV FP32 scale/sum|11.2083|6.68|
|Remaining Q4 API host/observer|1.2039|0.72|
|Completed-queue wait|0.1434|0.09|

MACの区間は起動設定・既存wait・device DMAを含む。GEMM内部のblock-scale補正を独立に測った値ではない。

### runtime論理転送

|方向|KiB/単位|API ms/単位|回/単位|
|---|---|---|---|
|D2H|49.250|1.35546|195.00|
|H2D|84.540|0.50509|71.13|

API時間には転送の内部waitを含み得る。物理PCIe帯域使用量やkernel内部DMA全量ではない。

|方向|src/owner tensor|dst tensor|KiB/単位|API ms/単位|
|---|---|---|---|---|
|H2D|KUNLUN0#embd#0||16.000|0.01690|
|D2H|attn_out-31||16.000|0.00979|
|D2H|l_out-30||16.000|0.00957|
|H2D|KUNLUN0#node_988#0||16.000|0.01015|
|H2D|KUNLUN0#node_989#0||16.000|0.00997|
|D2H|result_norm||16.000|0.01003|
|H2D|KUNLUN0#attn_inp_kq_mask#0||4.500|0.00715|
|H2D|(internal API) cache_k_l0 (view)||0.500|0.00712|
|H2D|(internal API) cache_v_l0 (view)||0.500|0.00673|
|H2D|(internal API) cache_k_l1 (view)||0.500|0.00689|
|H2D|(internal API) cache_v_l1 (view)||0.500|0.00673|
|H2D|(internal API) cache_k_l2 (view)||0.500|0.00689|
|H2D|(internal API) cache_v_l2 (view)||0.500|0.00672|
|H2D|(internal API) cache_k_l3 (view)||0.500|0.00687|
|H2D|(internal API) cache_v_l3 (view)||0.500|0.00672|


### 待機・起動・CPU（包含値）

|指標|平均/単位|
|---|---|
|process_wall_ms|167.82470|
|api_union_ms|141.73083|
|wait_union_ms|139.54995|
|launch_union_ms|3.71115|
|main_thread_cpu_ms|166.60502|
|other_worker_cpu_ms|17.32778|
|wait_calls|2127.12903|
|launch_calls|2151.06452|

main CPU時間にはdriver内のbusy waitも含む。CPU行列積の時間とは解釈しない。other workerはmain以外。

|host API分類|回/単位|包含 ms/単位|
|---|---|---|
|runtime.arguments.|2119.06|0.56043|
|runtime.launch_config.|2151.06|0.04878|
|runtime.launch.|2151.06|3.71115|
|runtime.wait.|2127.13|139.54995|
|peer.ready_inclusive.|0.00|0.00000|

引数設定・config・enqueueはhost発行時間。既に動くdevice処理と重なる場合がある。これらを演算wallへ加えない。

### device内copy_rows

|指標|1単位平均|
|---|---|
|全copy回数|130.00|
|alias候補回数|130.00|
|alias候補payload KiB|2080.000|
|alias候補 enqueue→既存wait完了 ms|1.44728|

alias候補はRMSNorm/要素演算に包含される。安全に削除できると証明した値ではなく、削除時の見積り上限候補。

### SET_ROWSとindex取得・検証

|区間|ms/単位|
|---|---|
|node_ms|5.23910|
|library_union_ms|4.24092|
|index_get_union_ms|0.74206|
|layout_index_validation_host_residual_ms|0.25612|

libraryはkv_max_set_rows/set_rows_f16の区間和集合。index_getはGGMLのindex取得（CPU内コピーの場合もある）。残余はlayout/indexのhost検証・targets処理・observer等であり、純粋な検証関数だけの時間とは断定しない。

## Q6_K出力層の軽量observer

XPU runtime auditを外し、CPU workerとharnessだけを計測。worker wallの和集合を使う。sum CPUは複数coreのCPU時間。logits取得はCPU buffer内のコピーであり、XPU D2Hに加えない。postはgreedy/有限値検査/detokenizeで、通常PP/TGの計測外。

|モデル|prefix|phase|Q6呼出/試行|Q6 worker ms/単位|Q6 graph ms/単位|sum CPU ms/単位|logits get ms/単位|post ms/position|
|---|---|---|---|---|---|---|---|---|
|7b|128|pp|1|0.02801|0.02912|0.16384|0.00005|—|
|7b|128|tg|31|3.39323|3.51736|19.98204|0.00588|0.09796|
|7b|2048|pp|16|0.02789|0.02894|0.16448|0.00008|—|
|7b|2048|tg|31|3.42218|3.54174|20.11498|0.00589|0.09961|
|tiny|128|pp|1|0.01434|0.01499|0.08333|0.00005|—|
|tiny|128|tg|31|1.68436|1.79144|9.75266|0.00670|0.09475|
|tiny|2048|pp|16|0.01436|0.01532|0.08368|0.00007|—|
|tiny|2048|tg|31|1.67281|1.78567|9.73404|0.00658|0.09943|

postのscopeは最初のPP logitsもTGラベルへ入り、1試行32位置。postはposition数で割り、31decodeでは割らない。

## observer校正（ABBA）

通常A→診断B→診断B→通常A。同じteacher入力、各実行3試行、A/B各6試行。PP128とその後31decode。

|モデル|GLU|phase|observer|n|平均 ms/単位|中央値|SD|平均増加 %|
|---|---|---|---|---|---|---|---|---|
|7b|0|pp|none|6|14.03402|14.00119|0.08087|—|
|7b|0|tg|none|6|143.22818|143.17680|0.50702|—|
|7b|0|pp|full|6|14.10461|14.10676|0.10345|0.50|
|7b|0|tg|full|6|157.52402|158.02501|2.05159|9.98|
|7b|1|pp|none|6|13.73919|13.74756|0.04658|—|
|7b|1|tg|none|6|136.81589|136.36772|0.75075|—|
|7b|1|pp|full|6|13.71668|13.73331|0.09932|-0.16|
|7b|1|tg|full|6|148.01174|147.97699|0.93733|8.18|
|tiny|0|pp|none|6|3.54715|3.55029|0.01082|—|
|tiny|0|tg|none|6|44.77755|44.89322|0.33555|—|
|tiny|0|pp|full|6|3.62324|3.62276|0.00803|2.14|
|tiny|0|tg|full|6|54.92187|54.67797|0.83972|22.65|
|tiny|1|pp|none|6|3.37694|3.41027|0.06562|—|
|tiny|1|tg|none|6|41.99070|41.39588|1.28986|—|
|tiny|1|pp|full|6|3.49039|3.49470|0.00981|3.36|
|tiny|1|tg|full|6|49.56520|49.36728|0.42318|18.04|


## CPU fallback一覧（全経路診断）

CPUノードを実測で検出したもの。metadata viewは算術fallbackと分ける。worker時間は同じ演算・同じtensor内の和集合。

|model|split|batch|ubatch|GLU|phase|op|tensor|calls（4長さ合計）|worker union計 ms|
|---|---|---|---|---|---|---|---|---|---|
|7b|layer|128|128|0|pp|GET_ROWS|embd|29|2.564|
|7b|layer|128|128|0|pp|GET_ROWS|node_988|29|0.091|
|7b|layer|128|128|0|pp|GET_ROWS|node_989|29|0.057|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-0|29|15.395|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-1|29|14.498|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-10|29|14.417|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-11|29|14.630|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-12|29|14.624|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-13|29|14.482|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-14|29|14.523|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-15|29|14.631|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-16|29|14.598|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-17|29|14.685|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-18|29|15.141|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-19|29|15.304|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-2|29|14.591|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-20|29|15.197|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-21|29|15.165|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-22|29|15.258|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-23|29|15.295|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-24|29|15.292|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-25|29|15.154|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-26|29|15.233|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-27|29|15.127|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-28|29|15.245|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-29|29|15.117|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-3|29|14.790|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-30|29|15.098|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-31|29|0.519|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-4|29|14.550|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-5|29|14.323|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-6|29|14.701|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-7|29|14.494|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-8|29|14.600|
|7b|layer|128|128|0|pp|GLU|ffn_swiglu-9|29|14.927|
|7b|layer|128|128|0|pp|MUL_MAT|result_output|29|104.842|
|7b|layer|128|128|0|tg|GET_ROWS|embd|32|0.428|
|7b|layer|128|128|0|tg|GET_ROWS|node_988|32|0.080|
|7b|layer|128|128|0|tg|GET_ROWS|node_989|32|0.052|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-0|32|0.547|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-1|32|0.523|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-10|32|0.498|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-11|32|0.577|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-12|32|0.554|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-13|32|0.518|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-14|32|0.467|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-15|32|0.572|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-16|32|0.498|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-17|32|0.548|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-18|32|0.509|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-19|32|0.636|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-2|32|0.479|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-20|32|0.518|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-21|32|0.594|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-22|32|0.519|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-23|32|0.580|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-24|32|0.547|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-25|32|0.510|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-26|32|0.579|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-27|32|0.592|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-28|32|0.545|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-29|32|0.469|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-3|32|0.580|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-30|32|0.498|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-31|32|0.555|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-4|32|0.508|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-5|32|0.473|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-6|32|0.602|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-7|32|0.533|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-8|32|0.475|
|7b|layer|128|128|0|tg|GLU|ffn_swiglu-9|32|0.528|
|7b|layer|128|128|0|tg|MUL_MAT|result_output|32|112.624|
|7b|layer|128|128|1|pp|GET_ROWS|embd|29|2.437|
|7b|layer|128|128|1|pp|GET_ROWS|node_988|29|0.093|
|7b|layer|128|128|1|pp|GET_ROWS|node_989|29|0.058|
|7b|layer|128|128|1|pp|MUL_MAT|result_output|29|103.339|
|7b|layer|128|128|1|tg|GET_ROWS|embd|32|0.335|
|7b|layer|128|128|1|tg|GET_ROWS|node_988|32|0.080|
|7b|layer|128|128|1|tg|GET_ROWS|node_989|32|0.053|
|7b|layer|128|128|1|tg|MUL_MAT|result_output|32|112.504|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|216.563|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|209.192|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|207.830|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|212.068|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|210.887|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|217.846|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|207.434|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|207.570|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|213.444|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|207.624|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|206.839|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|207.466|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|220.566|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|208.167|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|214.072|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|213.251|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|220.320|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|207.602|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|206.186|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|207.206|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|207.106|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_706|7|216.714|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_737|7|204.348|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_768|7|205.447|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_799|7|205.853|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_830|7|205.999|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|213.986|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_861|7|215.347|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_892|7|206.516|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_923|7|205.026|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_954|7|205.020|
|7b|layer|512|512|0|pp|FLASH_ATTN_EXT|node_985|7|206.832|
|7b|layer|512|512|0|pp|GET_ROWS|embd|8|3.084|
|7b|layer|512|512|0|pp|GET_ROWS|node_988|8|0.029|
|7b|layer|512|512|0|pp|GET_ROWS|node_989|8|0.019|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-0|8|20.317|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-1|8|19.977|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-10|8|20.075|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-11|8|19.990|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-12|8|20.075|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-13|8|20.056|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-14|8|20.142|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-15|8|20.112|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-16|8|20.140|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-17|8|20.112|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-18|8|20.038|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-19|8|20.134|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-2|8|20.011|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-20|8|19.927|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-21|8|20.046|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-22|8|19.924|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-23|8|20.020|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-24|8|19.981|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-25|8|19.972|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-26|8|20.161|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-27|8|19.912|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-28|8|19.915|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-29|8|19.995|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-3|8|20.063|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-30|8|20.080|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-31|8|0.121|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-4|8|20.029|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-5|8|20.012|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-6|8|20.074|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-7|8|20.095|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-8|8|20.008|
|7b|layer|512|512|0|pp|GLU|ffn_swiglu-9|8|20.245|
|7b|layer|512|512|0|pp|MUL_MAT|result_output|8|29.209|
|7b|layer|512|512|0|pp|ROPE|Kcur-0|7|1.607|
|7b|layer|512|512|0|pp|ROPE|Kcur-1|7|1.499|
|7b|layer|512|512|0|pp|ROPE|Kcur-10|7|1.625|
|7b|layer|512|512|0|pp|ROPE|Kcur-11|7|1.608|
|7b|layer|512|512|0|pp|ROPE|Kcur-12|7|1.541|
|7b|layer|512|512|0|pp|ROPE|Kcur-13|7|1.553|
|7b|layer|512|512|0|pp|ROPE|Kcur-14|7|1.688|
|7b|layer|512|512|0|pp|ROPE|Kcur-15|7|1.551|
|7b|layer|512|512|0|pp|ROPE|Kcur-16|7|1.552|
|7b|layer|512|512|0|pp|ROPE|Kcur-17|7|1.709|
|7b|layer|512|512|0|pp|ROPE|Kcur-18|7|1.422|
|7b|layer|512|512|0|pp|ROPE|Kcur-19|7|1.570|
|7b|layer|512|512|0|pp|ROPE|Kcur-2|7|1.619|
|7b|layer|512|512|0|pp|ROPE|Kcur-20|7|1.511|
|7b|layer|512|512|0|pp|ROPE|Kcur-21|7|1.471|
|7b|layer|512|512|0|pp|ROPE|Kcur-22|7|1.632|
|7b|layer|512|512|0|pp|ROPE|Kcur-23|7|1.495|
|7b|layer|512|512|0|pp|ROPE|Kcur-24|7|1.485|
|7b|layer|512|512|0|pp|ROPE|Kcur-25|7|1.581|
|7b|layer|512|512|0|pp|ROPE|Kcur-26|7|1.506|
|7b|layer|512|512|0|pp|ROPE|Kcur-27|7|1.630|
|7b|layer|512|512|0|pp|ROPE|Kcur-28|7|1.515|
|7b|layer|512|512|0|pp|ROPE|Kcur-29|7|1.625|
|7b|layer|512|512|0|pp|ROPE|Kcur-3|7|1.658|
|7b|layer|512|512|0|pp|ROPE|Kcur-30|7|1.583|
|7b|layer|512|512|0|pp|ROPE|Kcur-31|7|1.431|
|7b|layer|512|512|0|pp|ROPE|Kcur-4|7|1.575|
|7b|layer|512|512|0|pp|ROPE|Kcur-5|7|1.601|
|7b|layer|512|512|0|pp|ROPE|Kcur-6|7|1.513|
|7b|layer|512|512|0|pp|ROPE|Kcur-7|7|1.642|
|7b|layer|512|512|0|pp|ROPE|Kcur-8|7|1.588|
|7b|layer|512|512|0|pp|ROPE|Kcur-9|7|1.481|
|7b|layer|512|512|0|pp|ROPE|Qcur-0|7|1.663|
|7b|layer|512|512|0|pp|ROPE|Qcur-1|7|1.567|
|7b|layer|512|512|0|pp|ROPE|Qcur-10|7|1.531|
|7b|layer|512|512|0|pp|ROPE|Qcur-11|7|1.581|
|7b|layer|512|512|0|pp|ROPE|Qcur-12|7|1.451|
|7b|layer|512|512|0|pp|ROPE|Qcur-13|7|1.472|
|7b|layer|512|512|0|pp|ROPE|Qcur-14|7|1.583|
|7b|layer|512|512|0|pp|ROPE|Qcur-15|7|1.508|
|7b|layer|512|512|0|pp|ROPE|Qcur-16|7|1.576|
|7b|layer|512|512|0|pp|ROPE|Qcur-17|7|1.482|
|7b|layer|512|512|0|pp|ROPE|Qcur-18|7|1.489|
|7b|layer|512|512|0|pp|ROPE|Qcur-19|7|1.500|
|7b|layer|512|512|0|pp|ROPE|Qcur-2|7|1.432|
|7b|layer|512|512|0|pp|ROPE|Qcur-20|7|1.436|
|7b|layer|512|512|0|pp|ROPE|Qcur-21|7|1.517|
|7b|layer|512|512|0|pp|ROPE|Qcur-22|7|1.490|
|7b|layer|512|512|0|pp|ROPE|Qcur-23|7|1.458|
|7b|layer|512|512|0|pp|ROPE|Qcur-24|7|1.430|
|7b|layer|512|512|0|pp|ROPE|Qcur-25|7|1.475|
|7b|layer|512|512|0|pp|ROPE|Qcur-26|7|1.581|
|7b|layer|512|512|0|pp|ROPE|Qcur-27|7|1.557|
|7b|layer|512|512|0|pp|ROPE|Qcur-28|7|1.539|
|7b|layer|512|512|0|pp|ROPE|Qcur-29|7|1.510|
|7b|layer|512|512|0|pp|ROPE|Qcur-3|7|1.590|
|7b|layer|512|512|0|pp|ROPE|Qcur-30|7|1.514|
|7b|layer|512|512|0|pp|ROPE|Qcur-31|7|1.436|
|7b|layer|512|512|0|pp|ROPE|Qcur-4|7|1.451|
|7b|layer|512|512|0|pp|ROPE|Qcur-5|7|1.529|
|7b|layer|512|512|0|pp|ROPE|Qcur-6|7|1.615|
|7b|layer|512|512|0|pp|ROPE|Qcur-7|7|1.618|
|7b|layer|512|512|0|pp|ROPE|Qcur-8|7|1.518|
|7b|layer|512|512|0|pp|ROPE|Qcur-9|7|1.502|
|7b|layer|512|512|0|tg|GET_ROWS|embd|32|0.496|
|7b|layer|512|512|0|tg|GET_ROWS|node_988|32|0.079|
|7b|layer|512|512|0|tg|GET_ROWS|node_989|32|0.056|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-0|32|0.634|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-1|32|0.516|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-10|32|0.472|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-11|32|0.536|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-12|32|0.508|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-13|32|0.506|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-14|32|0.572|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-15|32|0.582|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-16|32|0.508|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-17|32|0.600|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-18|32|0.491|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-19|32|0.634|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-2|32|0.601|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-20|32|0.493|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-21|32|0.486|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-22|32|0.594|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-23|32|0.589|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-24|32|0.618|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-25|32|0.507|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-26|32|0.510|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-27|32|0.548|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-28|32|0.569|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-29|32|0.715|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-3|32|0.628|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-30|32|0.491|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-31|32|0.594|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-4|32|0.610|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-5|32|0.531|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-6|32|0.626|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-7|32|0.541|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-8|32|0.646|
|7b|layer|512|512|0|tg|GLU|ffn_swiglu-9|32|0.589|
|7b|layer|512|512|0|tg|MUL_MAT|result_output|32|113.713|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|219.610|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|219.630|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|217.506|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|216.872|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|231.969|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|217.325|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|217.417|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|217.923|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|217.771|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|217.437|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|219.259|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|219.946|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|219.691|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|217.651|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|217.488|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|217.740|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|216.599|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|217.346|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|217.737|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|221.660|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|221.653|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_706|7|221.719|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_737|7|219.530|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_768|7|219.818|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_799|7|218.261|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_830|7|217.665|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|216.521|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_861|7|217.345|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_892|7|217.382|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_923|7|227.234|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_954|7|219.586|
|7b|layer|512|512|1|pp|FLASH_ATTN_EXT|node_985|7|221.035|
|7b|layer|512|512|1|pp|GET_ROWS|embd|8|3.119|
|7b|layer|512|512|1|pp|GET_ROWS|node_988|8|0.027|
|7b|layer|512|512|1|pp|GET_ROWS|node_989|8|0.018|
|7b|layer|512|512|1|pp|MUL_MAT|result_output|8|29.089|
|7b|layer|512|512|1|pp|ROPE|Kcur-0|7|1.632|
|7b|layer|512|512|1|pp|ROPE|Kcur-1|7|1.719|
|7b|layer|512|512|1|pp|ROPE|Kcur-10|7|1.422|
|7b|layer|512|512|1|pp|ROPE|Kcur-11|7|1.706|
|7b|layer|512|512|1|pp|ROPE|Kcur-12|7|1.613|
|7b|layer|512|512|1|pp|ROPE|Kcur-13|7|1.755|
|7b|layer|512|512|1|pp|ROPE|Kcur-14|7|1.512|
|7b|layer|512|512|1|pp|ROPE|Kcur-15|7|1.652|
|7b|layer|512|512|1|pp|ROPE|Kcur-16|7|1.598|
|7b|layer|512|512|1|pp|ROPE|Kcur-17|7|1.596|
|7b|layer|512|512|1|pp|ROPE|Kcur-18|7|1.638|
|7b|layer|512|512|1|pp|ROPE|Kcur-19|7|1.638|
|7b|layer|512|512|1|pp|ROPE|Kcur-2|7|1.581|
|7b|layer|512|512|1|pp|ROPE|Kcur-20|7|1.739|
|7b|layer|512|512|1|pp|ROPE|Kcur-21|7|1.645|
|7b|layer|512|512|1|pp|ROPE|Kcur-22|7|1.692|
|7b|layer|512|512|1|pp|ROPE|Kcur-23|7|1.699|
|7b|layer|512|512|1|pp|ROPE|Kcur-24|7|1.651|
|7b|layer|512|512|1|pp|ROPE|Kcur-25|7|1.622|
|7b|layer|512|512|1|pp|ROPE|Kcur-26|7|1.616|
|7b|layer|512|512|1|pp|ROPE|Kcur-27|7|1.623|
|7b|layer|512|512|1|pp|ROPE|Kcur-28|7|1.600|
|7b|layer|512|512|1|pp|ROPE|Kcur-29|7|1.715|
|7b|layer|512|512|1|pp|ROPE|Kcur-3|7|1.766|
|7b|layer|512|512|1|pp|ROPE|Kcur-30|7|1.713|
|7b|layer|512|512|1|pp|ROPE|Kcur-31|7|1.694|
|7b|layer|512|512|1|pp|ROPE|Kcur-4|7|1.628|
|7b|layer|512|512|1|pp|ROPE|Kcur-5|7|1.619|
|7b|layer|512|512|1|pp|ROPE|Kcur-6|7|1.557|
|7b|layer|512|512|1|pp|ROPE|Kcur-7|7|1.611|
|7b|layer|512|512|1|pp|ROPE|Kcur-8|7|1.644|
|7b|layer|512|512|1|pp|ROPE|Kcur-9|7|1.589|
|7b|layer|512|512|1|pp|ROPE|Qcur-0|7|1.582|
|7b|layer|512|512|1|pp|ROPE|Qcur-1|7|1.598|
|7b|layer|512|512|1|pp|ROPE|Qcur-10|7|1.562|
|7b|layer|512|512|1|pp|ROPE|Qcur-11|7|1.500|
|7b|layer|512|512|1|pp|ROPE|Qcur-12|7|1.549|
|7b|layer|512|512|1|pp|ROPE|Qcur-13|7|1.497|
|7b|layer|512|512|1|pp|ROPE|Qcur-14|7|1.546|
|7b|layer|512|512|1|pp|ROPE|Qcur-15|7|1.601|
|7b|layer|512|512|1|pp|ROPE|Qcur-16|7|1.473|
|7b|layer|512|512|1|pp|ROPE|Qcur-17|7|1.458|
|7b|layer|512|512|1|pp|ROPE|Qcur-18|7|1.425|
|7b|layer|512|512|1|pp|ROPE|Qcur-19|7|1.487|
|7b|layer|512|512|1|pp|ROPE|Qcur-2|7|1.535|
|7b|layer|512|512|1|pp|ROPE|Qcur-20|7|1.473|
|7b|layer|512|512|1|pp|ROPE|Qcur-21|7|1.505|
|7b|layer|512|512|1|pp|ROPE|Qcur-22|7|1.486|
|7b|layer|512|512|1|pp|ROPE|Qcur-23|7|1.579|
|7b|layer|512|512|1|pp|ROPE|Qcur-24|7|1.569|
|7b|layer|512|512|1|pp|ROPE|Qcur-25|7|1.603|
|7b|layer|512|512|1|pp|ROPE|Qcur-26|7|1.520|
|7b|layer|512|512|1|pp|ROPE|Qcur-27|7|1.540|
|7b|layer|512|512|1|pp|ROPE|Qcur-28|7|1.502|
|7b|layer|512|512|1|pp|ROPE|Qcur-29|7|1.549|
|7b|layer|512|512|1|pp|ROPE|Qcur-3|7|1.526|
|7b|layer|512|512|1|pp|ROPE|Qcur-30|7|1.568|
|7b|layer|512|512|1|pp|ROPE|Qcur-31|7|1.517|
|7b|layer|512|512|1|pp|ROPE|Qcur-4|7|1.550|
|7b|layer|512|512|1|pp|ROPE|Qcur-5|7|1.516|
|7b|layer|512|512|1|pp|ROPE|Qcur-6|7|1.464|
|7b|layer|512|512|1|pp|ROPE|Qcur-7|7|1.558|
|7b|layer|512|512|1|pp|ROPE|Qcur-8|7|1.544|
|7b|layer|512|512|1|pp|ROPE|Qcur-9|7|1.547|
|7b|layer|512|512|1|tg|GET_ROWS|embd|32|0.371|
|7b|layer|512|512|1|tg|GET_ROWS|node_988|32|0.078|
|7b|layer|512|512|1|tg|GET_ROWS|node_989|32|0.055|
|7b|layer|512|512|1|tg|MUL_MAT|result_output|32|112.574|
|7b|pd0|128|128|0|pp|GET_ROWS|embd|29|2.500|
|7b|pd0|128|128|0|pp|GET_ROWS|node_988|29|0.099|
|7b|pd0|128|128|0|pp|GET_ROWS|node_989|29|0.057|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-0|29|15.304|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-1|29|15.398|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-10|29|15.220|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-11|29|14.992|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-12|29|15.230|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-13|29|15.051|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-14|29|15.122|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-15|29|14.953|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-16|29|15.233|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-17|29|15.101|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-18|29|15.104|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-19|29|15.008|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-2|29|15.189|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-20|29|15.171|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-21|29|15.194|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-22|29|15.132|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-23|29|15.145|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-24|29|15.208|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-25|29|15.368|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-26|29|15.256|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-27|29|15.232|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-28|29|15.261|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-29|29|15.036|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-3|29|15.147|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-30|29|15.061|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-31|29|0.454|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-4|29|15.090|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-5|29|15.081|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-6|29|15.135|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-7|29|15.162|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-8|29|15.208|
|7b|pd0|128|128|0|pp|GLU|ffn_swiglu-9|29|15.046|
|7b|pd0|128|128|0|pp|MUL_MAT|result_output|29|106.081|
|7b|pd0|128|128|0|tg|GET_ROWS|embd|32|0.338|
|7b|pd0|128|128|0|tg|GET_ROWS|node_988|32|0.081|
|7b|pd0|128|128|0|tg|GET_ROWS|node_989|32|0.053|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-0|32|0.402|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-1|32|0.432|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-10|32|0.487|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-11|32|0.391|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-12|32|0.388|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-13|32|0.442|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-14|32|0.409|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-15|32|0.407|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-16|32|0.383|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-17|32|0.369|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-18|32|0.415|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-19|32|0.402|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-2|32|0.434|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-20|32|0.388|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-21|32|0.433|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-22|32|0.460|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-23|32|0.402|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-24|32|0.401|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-25|32|0.440|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-26|32|0.391|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-27|32|0.429|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-28|32|0.430|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-29|32|0.441|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-3|32|0.439|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-30|32|0.423|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-31|32|0.498|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-4|32|0.442|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-5|32|0.401|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-6|32|0.454|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-7|32|0.382|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-8|32|0.398|
|7b|pd0|128|128|0|tg|GLU|ffn_swiglu-9|32|0.451|
|7b|pd0|128|128|0|tg|MUL_MAT|result_output|32|113.645|
|7b|pd0|128|128|1|pp|GET_ROWS|embd|29|2.389|
|7b|pd0|128|128|1|pp|GET_ROWS|node_988|29|0.094|
|7b|pd0|128|128|1|pp|GET_ROWS|node_989|29|0.056|
|7b|pd0|128|128|1|pp|MUL_MAT|result_output|29|103.082|
|7b|pd0|128|128|1|tg|GET_ROWS|embd|32|0.355|
|7b|pd0|128|128|1|tg|GET_ROWS|node_988|32|0.082|
|7b|pd0|128|128|1|tg|GET_ROWS|node_989|32|0.053|
|7b|pd0|128|128|1|tg|MUL_MAT|result_output|32|112.835|
|7b|pd0|512|128|0|pp|GET_ROWS|embd|29|2.213|
|7b|pd0|512|128|0|pp|GET_ROWS|node_988|8|0.027|
|7b|pd0|512|128|0|pp|GET_ROWS|node_989|8|0.018|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-0|29|15.597|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-1|29|14.775|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-10|29|14.630|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-11|29|14.689|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-12|29|14.728|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-13|29|14.678|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-14|29|14.449|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-15|29|14.803|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-16|29|14.644|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-17|29|14.670|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-18|29|14.616|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-19|29|14.895|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-2|29|14.562|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-20|29|14.818|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-21|29|14.645|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-22|29|14.667|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-23|29|14.745|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-24|29|14.747|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-25|29|14.765|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-26|29|14.506|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-27|29|14.660|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-28|29|14.877|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-29|29|14.777|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-3|29|14.658|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-30|29|14.545|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-31|8|0.118|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-4|29|14.864|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-5|29|14.921|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-6|29|14.504|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-7|29|14.819|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-8|29|14.747|
|7b|pd0|512|128|0|pp|GLU|ffn_swiglu-9|29|14.619|
|7b|pd0|512|128|0|pp|MUL_MAT|result_output|8|28.977|
|7b|pd0|512|128|0|tg|GET_ROWS|embd|32|0.348|
|7b|pd0|512|128|0|tg|GET_ROWS|node_988|32|0.079|
|7b|pd0|512|128|0|tg|GET_ROWS|node_989|32|0.063|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-0|32|0.432|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-1|32|0.512|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-10|32|0.420|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-11|32|0.478|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-12|32|0.456|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-13|32|0.434|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-14|32|0.395|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-15|32|0.428|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-16|32|0.439|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-17|32|0.394|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-18|32|0.444|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-19|32|0.422|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-2|32|0.406|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-20|32|0.471|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-21|32|0.541|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-22|32|0.416|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-23|32|0.449|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-24|32|0.400|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-25|32|0.445|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-26|32|0.461|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-27|32|0.499|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-28|32|0.425|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-29|32|0.506|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-3|32|0.438|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-30|32|0.458|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-31|32|0.466|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-4|32|0.423|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-5|32|0.455|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-6|32|0.474|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-7|32|0.449|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-8|32|0.464|
|7b|pd0|512|128|0|tg|GLU|ffn_swiglu-9|32|0.430|
|7b|pd0|512|128|0|tg|MUL_MAT|result_output|32|113.120|
|7b|pd0|512|128|1|pp|GET_ROWS|embd|29|2.146|
|7b|pd0|512|128|1|pp|GET_ROWS|node_988|8|0.026|
|7b|pd0|512|128|1|pp|GET_ROWS|node_989|8|0.018|
|7b|pd0|512|128|1|pp|MUL_MAT|result_output|8|29.031|
|7b|pd0|512|128|1|tg|GET_ROWS|embd|32|0.358|
|7b|pd0|512|128|1|tg|GET_ROWS|node_988|32|0.078|
|7b|pd0|512|128|1|tg|GET_ROWS|node_989|32|0.061|
|7b|pd0|512|128|1|tg|MUL_MAT|result_output|32|113.808|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|207.913|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|205.555|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|206.489|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|205.875|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|217.382|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|206.342|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|208.984|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|215.843|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|205.846|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|205.632|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|208.317|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|209.200|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|208.539|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|214.099|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|208.049|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|209.071|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|212.058|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|206.082|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|206.215|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|205.801|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|206.470|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_706|7|205.679|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_737|7|207.905|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_768|7|213.736|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_799|7|211.347|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_830|7|205.719|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|209.401|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_861|7|207.972|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_892|7|208.358|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_923|7|216.959|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_954|7|207.243|
|7b|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_985|7|208.192|
|7b|pd0|512|512|0|pp|GET_ROWS|embd|8|3.179|
|7b|pd0|512|512|0|pp|GET_ROWS|node_988|8|0.027|
|7b|pd0|512|512|0|pp|GET_ROWS|node_989|8|0.017|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-0|8|20.073|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-1|8|20.126|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-10|8|20.044|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-11|8|20.077|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-12|8|20.122|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-13|8|20.125|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-14|8|20.131|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-15|8|19.975|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-16|8|20.264|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-17|8|20.153|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-18|8|20.161|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-19|8|20.164|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-2|8|20.126|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-20|8|20.166|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-21|8|20.145|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-22|8|20.208|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-23|8|20.271|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-24|8|20.190|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-25|8|20.145|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-26|8|20.110|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-27|8|20.077|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-28|8|20.086|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-29|8|20.176|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-3|8|20.146|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-30|8|20.128|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-31|8|0.155|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-4|8|20.079|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-5|8|20.237|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-6|8|20.062|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-7|8|20.094|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-8|8|20.128|
|7b|pd0|512|512|0|pp|GLU|ffn_swiglu-9|8|20.160|
|7b|pd0|512|512|0|pp|MUL_MAT|result_output|8|29.086|
|7b|pd0|512|512|0|pp|ROPE|Kcur-0|7|1.737|
|7b|pd0|512|512|0|pp|ROPE|Kcur-1|7|1.704|
|7b|pd0|512|512|0|pp|ROPE|Kcur-10|7|1.705|
|7b|pd0|512|512|0|pp|ROPE|Kcur-11|7|1.699|
|7b|pd0|512|512|0|pp|ROPE|Kcur-12|7|1.707|
|7b|pd0|512|512|0|pp|ROPE|Kcur-13|7|1.663|
|7b|pd0|512|512|0|pp|ROPE|Kcur-14|7|1.712|
|7b|pd0|512|512|0|pp|ROPE|Kcur-15|7|1.714|
|7b|pd0|512|512|0|pp|ROPE|Kcur-16|7|1.669|
|7b|pd0|512|512|0|pp|ROPE|Kcur-17|7|1.736|
|7b|pd0|512|512|0|pp|ROPE|Kcur-18|7|1.703|
|7b|pd0|512|512|0|pp|ROPE|Kcur-19|7|1.591|
|7b|pd0|512|512|0|pp|ROPE|Kcur-2|7|1.758|
|7b|pd0|512|512|0|pp|ROPE|Kcur-20|7|1.741|
|7b|pd0|512|512|0|pp|ROPE|Kcur-21|7|1.651|
|7b|pd0|512|512|0|pp|ROPE|Kcur-22|7|1.655|
|7b|pd0|512|512|0|pp|ROPE|Kcur-23|7|1.737|
|7b|pd0|512|512|0|pp|ROPE|Kcur-24|7|1.766|
|7b|pd0|512|512|0|pp|ROPE|Kcur-25|7|1.799|
|7b|pd0|512|512|0|pp|ROPE|Kcur-26|7|1.668|
|7b|pd0|512|512|0|pp|ROPE|Kcur-27|7|1.704|
|7b|pd0|512|512|0|pp|ROPE|Kcur-28|7|1.747|
|7b|pd0|512|512|0|pp|ROPE|Kcur-29|7|1.775|
|7b|pd0|512|512|0|pp|ROPE|Kcur-3|7|1.766|
|7b|pd0|512|512|0|pp|ROPE|Kcur-30|7|1.696|
|7b|pd0|512|512|0|pp|ROPE|Kcur-31|7|1.849|
|7b|pd0|512|512|0|pp|ROPE|Kcur-4|7|1.652|
|7b|pd0|512|512|0|pp|ROPE|Kcur-5|7|1.709|
|7b|pd0|512|512|0|pp|ROPE|Kcur-6|7|1.790|
|7b|pd0|512|512|0|pp|ROPE|Kcur-7|7|1.597|
|7b|pd0|512|512|0|pp|ROPE|Kcur-8|7|1.695|
|7b|pd0|512|512|0|pp|ROPE|Kcur-9|7|1.547|
|7b|pd0|512|512|0|pp|ROPE|Qcur-0|7|1.689|
|7b|pd0|512|512|0|pp|ROPE|Qcur-1|7|1.485|
|7b|pd0|512|512|0|pp|ROPE|Qcur-10|7|1.473|
|7b|pd0|512|512|0|pp|ROPE|Qcur-11|7|1.449|
|7b|pd0|512|512|0|pp|ROPE|Qcur-12|7|1.459|
|7b|pd0|512|512|0|pp|ROPE|Qcur-13|7|1.522|
|7b|pd0|512|512|0|pp|ROPE|Qcur-14|7|1.458|
|7b|pd0|512|512|0|pp|ROPE|Qcur-15|7|1.568|
|7b|pd0|512|512|0|pp|ROPE|Qcur-16|7|1.486|
|7b|pd0|512|512|0|pp|ROPE|Qcur-17|7|1.508|
|7b|pd0|512|512|0|pp|ROPE|Qcur-18|7|1.463|
|7b|pd0|512|512|0|pp|ROPE|Qcur-19|7|1.407|
|7b|pd0|512|512|0|pp|ROPE|Qcur-2|7|1.544|
|7b|pd0|512|512|0|pp|ROPE|Qcur-20|7|1.423|
|7b|pd0|512|512|0|pp|ROPE|Qcur-21|7|1.509|
|7b|pd0|512|512|0|pp|ROPE|Qcur-22|7|1.427|
|7b|pd0|512|512|0|pp|ROPE|Qcur-23|7|1.466|
|7b|pd0|512|512|0|pp|ROPE|Qcur-24|7|1.480|
|7b|pd0|512|512|0|pp|ROPE|Qcur-25|7|1.594|
|7b|pd0|512|512|0|pp|ROPE|Qcur-26|7|1.541|
|7b|pd0|512|512|0|pp|ROPE|Qcur-27|7|1.477|
|7b|pd0|512|512|0|pp|ROPE|Qcur-28|7|1.472|
|7b|pd0|512|512|0|pp|ROPE|Qcur-29|7|1.459|
|7b|pd0|512|512|0|pp|ROPE|Qcur-3|7|1.538|
|7b|pd0|512|512|0|pp|ROPE|Qcur-30|7|1.489|
|7b|pd0|512|512|0|pp|ROPE|Qcur-31|7|1.552|
|7b|pd0|512|512|0|pp|ROPE|Qcur-4|7|1.445|
|7b|pd0|512|512|0|pp|ROPE|Qcur-5|7|1.416|
|7b|pd0|512|512|0|pp|ROPE|Qcur-6|7|1.537|
|7b|pd0|512|512|0|pp|ROPE|Qcur-7|7|1.508|
|7b|pd0|512|512|0|pp|ROPE|Qcur-8|7|1.487|
|7b|pd0|512|512|0|pp|ROPE|Qcur-9|7|1.480|
|7b|pd0|512|512|0|tg|GET_ROWS|embd|32|0.448|
|7b|pd0|512|512|0|tg|GET_ROWS|node_988|32|0.075|
|7b|pd0|512|512|0|tg|GET_ROWS|node_989|32|0.056|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-0|32|0.403|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-1|32|0.432|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-10|32|0.380|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-11|32|0.372|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-12|32|0.376|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-13|32|0.441|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-14|32|0.393|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-15|32|0.389|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-16|32|0.396|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-17|32|0.383|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-18|32|0.431|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-19|32|0.446|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-2|32|0.402|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-20|32|0.412|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-21|32|0.420|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-22|32|0.447|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-23|32|0.413|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-24|32|0.421|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-25|32|0.421|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-26|32|0.396|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-27|32|0.411|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-28|32|0.371|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-29|32|0.457|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-3|32|0.387|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-30|32|0.426|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-31|32|0.419|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-4|32|0.399|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-5|32|0.403|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-6|32|0.368|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-7|32|0.422|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-8|32|0.384|
|7b|pd0|512|512|0|tg|GLU|ffn_swiglu-9|32|0.362|
|7b|pd0|512|512|0|tg|MUL_MAT|result_output|32|112.586|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|213.064|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|214.312|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|225.472|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|213.085|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|218.608|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|211.417|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|211.700|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|221.353|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|220.298|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|212.928|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|216.357|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|219.306|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|219.779|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|213.790|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|213.057|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|213.171|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|213.737|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|214.569|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|214.366|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|213.872|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|215.212|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_706|7|215.355|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_737|7|217.702|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_768|7|212.802|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_799|7|212.669|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_830|7|222.179|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|211.985|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_861|7|210.884|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_892|7|210.993|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_923|7|216.925|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_954|7|217.604|
|7b|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_985|7|212.909|
|7b|pd0|512|512|1|pp|GET_ROWS|embd|8|3.025|
|7b|pd0|512|512|1|pp|GET_ROWS|node_988|8|0.027|
|7b|pd0|512|512|1|pp|GET_ROWS|node_989|8|0.018|
|7b|pd0|512|512|1|pp|MUL_MAT|result_output|8|29.389|
|7b|pd0|512|512|1|pp|ROPE|Kcur-0|7|1.653|
|7b|pd0|512|512|1|pp|ROPE|Kcur-1|7|1.546|
|7b|pd0|512|512|1|pp|ROPE|Kcur-10|7|1.648|
|7b|pd0|512|512|1|pp|ROPE|Kcur-11|7|1.615|
|7b|pd0|512|512|1|pp|ROPE|Kcur-12|7|1.631|
|7b|pd0|512|512|1|pp|ROPE|Kcur-13|7|1.562|
|7b|pd0|512|512|1|pp|ROPE|Kcur-14|7|1.572|
|7b|pd0|512|512|1|pp|ROPE|Kcur-15|7|1.515|
|7b|pd0|512|512|1|pp|ROPE|Kcur-16|7|1.620|
|7b|pd0|512|512|1|pp|ROPE|Kcur-17|7|1.590|
|7b|pd0|512|512|1|pp|ROPE|Kcur-18|7|1.517|
|7b|pd0|512|512|1|pp|ROPE|Kcur-19|7|1.595|
|7b|pd0|512|512|1|pp|ROPE|Kcur-2|7|1.499|
|7b|pd0|512|512|1|pp|ROPE|Kcur-20|7|1.565|
|7b|pd0|512|512|1|pp|ROPE|Kcur-21|7|1.531|
|7b|pd0|512|512|1|pp|ROPE|Kcur-22|7|1.593|
|7b|pd0|512|512|1|pp|ROPE|Kcur-23|7|1.524|
|7b|pd0|512|512|1|pp|ROPE|Kcur-24|7|1.654|
|7b|pd0|512|512|1|pp|ROPE|Kcur-25|7|1.392|
|7b|pd0|512|512|1|pp|ROPE|Kcur-26|7|1.623|
|7b|pd0|512|512|1|pp|ROPE|Kcur-27|7|1.574|
|7b|pd0|512|512|1|pp|ROPE|Kcur-28|7|1.564|
|7b|pd0|512|512|1|pp|ROPE|Kcur-29|7|1.596|
|7b|pd0|512|512|1|pp|ROPE|Kcur-3|7|1.634|
|7b|pd0|512|512|1|pp|ROPE|Kcur-30|7|1.532|
|7b|pd0|512|512|1|pp|ROPE|Kcur-31|7|1.519|
|7b|pd0|512|512|1|pp|ROPE|Kcur-4|7|1.560|
|7b|pd0|512|512|1|pp|ROPE|Kcur-5|7|1.628|
|7b|pd0|512|512|1|pp|ROPE|Kcur-6|7|1.604|
|7b|pd0|512|512|1|pp|ROPE|Kcur-7|7|1.540|
|7b|pd0|512|512|1|pp|ROPE|Kcur-8|7|1.478|
|7b|pd0|512|512|1|pp|ROPE|Kcur-9|7|1.476|
|7b|pd0|512|512|1|pp|ROPE|Qcur-0|7|1.627|
|7b|pd0|512|512|1|pp|ROPE|Qcur-1|7|1.498|
|7b|pd0|512|512|1|pp|ROPE|Qcur-10|7|1.544|
|7b|pd0|512|512|1|pp|ROPE|Qcur-11|7|1.503|
|7b|pd0|512|512|1|pp|ROPE|Qcur-12|7|1.569|
|7b|pd0|512|512|1|pp|ROPE|Qcur-13|7|1.516|
|7b|pd0|512|512|1|pp|ROPE|Qcur-14|7|1.511|
|7b|pd0|512|512|1|pp|ROPE|Qcur-15|7|1.543|
|7b|pd0|512|512|1|pp|ROPE|Qcur-16|7|1.486|
|7b|pd0|512|512|1|pp|ROPE|Qcur-17|7|1.485|
|7b|pd0|512|512|1|pp|ROPE|Qcur-18|7|1.440|
|7b|pd0|512|512|1|pp|ROPE|Qcur-19|7|1.467|
|7b|pd0|512|512|1|pp|ROPE|Qcur-2|7|1.489|
|7b|pd0|512|512|1|pp|ROPE|Qcur-20|7|1.508|
|7b|pd0|512|512|1|pp|ROPE|Qcur-21|7|1.583|
|7b|pd0|512|512|1|pp|ROPE|Qcur-22|7|1.566|
|7b|pd0|512|512|1|pp|ROPE|Qcur-23|7|1.504|
|7b|pd0|512|512|1|pp|ROPE|Qcur-24|7|1.552|
|7b|pd0|512|512|1|pp|ROPE|Qcur-25|7|1.457|
|7b|pd0|512|512|1|pp|ROPE|Qcur-26|7|1.459|
|7b|pd0|512|512|1|pp|ROPE|Qcur-27|7|1.499|
|7b|pd0|512|512|1|pp|ROPE|Qcur-28|7|1.390|
|7b|pd0|512|512|1|pp|ROPE|Qcur-29|7|1.466|
|7b|pd0|512|512|1|pp|ROPE|Qcur-3|7|1.520|
|7b|pd0|512|512|1|pp|ROPE|Qcur-30|7|1.475|
|7b|pd0|512|512|1|pp|ROPE|Qcur-31|7|1.476|
|7b|pd0|512|512|1|pp|ROPE|Qcur-4|7|1.446|
|7b|pd0|512|512|1|pp|ROPE|Qcur-5|7|1.502|
|7b|pd0|512|512|1|pp|ROPE|Qcur-6|7|1.404|
|7b|pd0|512|512|1|pp|ROPE|Qcur-7|7|1.527|
|7b|pd0|512|512|1|pp|ROPE|Qcur-8|7|1.465|
|7b|pd0|512|512|1|pp|ROPE|Qcur-9|7|1.390|
|7b|pd0|512|512|1|tg|GET_ROWS|embd|32|0.329|
|7b|pd0|512|512|1|tg|GET_ROWS|node_988|32|0.079|
|7b|pd0|512|512|1|tg|GET_ROWS|node_989|32|0.056|
|7b|pd0|512|512|1|tg|MUL_MAT|result_output|32|113.155|
|7b|pd1|128|128|0|pp|GET_ROWS|embd|29|2.319|
|7b|pd1|128|128|0|pp|GET_ROWS|node_988|29|0.100|
|7b|pd1|128|128|0|pp|GET_ROWS|node_989|29|0.057|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-0|29|14.992|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-1|29|14.687|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-10|29|14.630|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-11|29|14.624|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-12|29|14.608|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-13|29|14.768|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-14|29|14.912|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-15|29|14.842|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-16|29|14.715|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-17|29|14.799|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-18|29|14.812|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-19|29|14.678|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-2|29|15.075|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-20|29|14.796|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-21|29|14.844|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-22|29|14.873|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-23|29|14.649|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-24|29|14.767|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-25|29|14.862|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-26|29|14.672|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-27|29|14.619|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-28|29|14.722|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-29|29|14.776|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-3|29|14.759|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-30|29|14.694|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-31|29|0.352|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-4|29|14.636|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-5|29|14.922|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-6|29|14.717|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-7|29|14.755|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-8|29|14.743|
|7b|pd1|128|128|0|pp|GLU|ffn_swiglu-9|29|14.683|
|7b|pd1|128|128|0|pp|MUL_MAT|result_output|29|105.161|
|7b|pd1|128|128|0|tg|GET_ROWS|embd|32|0.363|
|7b|pd1|128|128|0|tg|GET_ROWS|node_988|32|0.081|
|7b|pd1|128|128|0|tg|GET_ROWS|node_989|32|0.052|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-0|32|0.468|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-1|32|0.431|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-10|32|0.393|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-11|32|0.354|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-12|32|0.432|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-13|32|0.377|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-14|32|0.394|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-15|32|0.414|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-16|32|0.400|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-17|32|0.374|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-18|32|0.412|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-19|32|0.401|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-2|32|0.401|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-20|32|0.385|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-21|32|0.439|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-22|32|0.389|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-23|32|0.451|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-24|32|0.410|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-25|32|0.451|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-26|32|0.400|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-27|32|0.451|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-28|32|0.422|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-29|32|0.393|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-3|32|0.413|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-30|32|0.379|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-31|32|0.378|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-4|32|0.411|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-5|32|0.510|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-6|32|0.411|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-7|32|0.453|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-8|32|0.435|
|7b|pd1|128|128|0|tg|GLU|ffn_swiglu-9|32|0.429|
|7b|pd1|128|128|0|tg|MUL_MAT|result_output|32|113.062|
|7b|pd1|128|128|1|pp|GET_ROWS|embd|29|2.401|
|7b|pd1|128|128|1|pp|GET_ROWS|node_988|29|0.093|
|7b|pd1|128|128|1|pp|GET_ROWS|node_989|29|0.055|
|7b|pd1|128|128|1|pp|MUL_MAT|result_output|29|103.986|
|7b|pd1|128|128|1|tg|GET_ROWS|embd|32|0.376|
|7b|pd1|128|128|1|tg|GET_ROWS|node_988|32|0.081|
|7b|pd1|128|128|1|tg|GET_ROWS|node_989|32|0.053|
|7b|pd1|128|128|1|tg|MUL_MAT|result_output|32|113.150|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|222.002|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|206.016|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|206.184|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|216.379|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|209.261|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|215.254|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|208.214|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|212.479|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|208.935|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|204.622|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|204.838|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|215.713|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|216.771|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|204.746|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|206.324|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|207.856|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|208.535|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|217.854|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|216.120|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|210.663|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|204.585|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_706|7|209.933|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_737|7|206.889|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_768|7|206.781|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_799|7|206.951|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_830|7|208.514|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|213.175|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_861|7|209.227|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_892|7|208.929|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_923|7|204.755|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_954|7|204.358|
|7b|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_985|7|215.942|
|7b|pd1|512|512|0|pp|GET_ROWS|embd|8|3.018|
|7b|pd1|512|512|0|pp|GET_ROWS|node_988|8|0.030|
|7b|pd1|512|512|0|pp|GET_ROWS|node_989|8|0.019|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-0|8|20.133|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-1|8|20.468|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-10|8|20.079|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-11|8|20.142|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-12|8|20.112|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-13|8|20.091|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-14|8|20.147|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-15|8|20.307|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-16|8|20.102|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-17|8|20.204|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-18|8|20.561|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-19|8|20.253|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-2|8|20.198|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-20|8|20.123|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-21|8|20.183|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-22|8|20.131|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-23|8|20.065|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-24|8|20.350|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-25|8|20.063|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-26|8|20.161|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-27|8|20.071|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-28|8|20.245|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-29|8|20.334|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-3|8|20.379|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-30|8|20.282|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-31|8|0.122|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-4|8|20.090|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-5|8|20.188|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-6|8|20.081|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-7|8|20.103|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-8|8|20.220|
|7b|pd1|512|512|0|pp|GLU|ffn_swiglu-9|8|20.150|
|7b|pd1|512|512|0|pp|MUL_MAT|result_output|8|29.093|
|7b|pd1|512|512|0|pp|ROPE|Kcur-0|7|1.552|
|7b|pd1|512|512|0|pp|ROPE|Kcur-1|7|1.649|
|7b|pd1|512|512|0|pp|ROPE|Kcur-10|7|1.614|
|7b|pd1|512|512|0|pp|ROPE|Kcur-11|7|1.562|
|7b|pd1|512|512|0|pp|ROPE|Kcur-12|7|1.556|
|7b|pd1|512|512|0|pp|ROPE|Kcur-13|7|1.460|
|7b|pd1|512|512|0|pp|ROPE|Kcur-14|7|1.641|
|7b|pd1|512|512|0|pp|ROPE|Kcur-15|7|1.562|
|7b|pd1|512|512|0|pp|ROPE|Kcur-16|7|1.517|
|7b|pd1|512|512|0|pp|ROPE|Kcur-17|7|1.558|
|7b|pd1|512|512|0|pp|ROPE|Kcur-18|7|1.662|
|7b|pd1|512|512|0|pp|ROPE|Kcur-19|7|1.651|
|7b|pd1|512|512|0|pp|ROPE|Kcur-2|7|1.689|
|7b|pd1|512|512|0|pp|ROPE|Kcur-20|7|1.525|
|7b|pd1|512|512|0|pp|ROPE|Kcur-21|7|1.570|
|7b|pd1|512|512|0|pp|ROPE|Kcur-22|7|1.527|
|7b|pd1|512|512|0|pp|ROPE|Kcur-23|7|1.570|
|7b|pd1|512|512|0|pp|ROPE|Kcur-24|7|1.689|
|7b|pd1|512|512|0|pp|ROPE|Kcur-25|7|1.698|
|7b|pd1|512|512|0|pp|ROPE|Kcur-26|7|1.728|
|7b|pd1|512|512|0|pp|ROPE|Kcur-27|7|1.563|
|7b|pd1|512|512|0|pp|ROPE|Kcur-28|7|1.589|
|7b|pd1|512|512|0|pp|ROPE|Kcur-29|7|1.610|
|7b|pd1|512|512|0|pp|ROPE|Kcur-3|7|1.478|
|7b|pd1|512|512|0|pp|ROPE|Kcur-30|7|1.508|
|7b|pd1|512|512|0|pp|ROPE|Kcur-31|7|1.578|
|7b|pd1|512|512|0|pp|ROPE|Kcur-4|7|1.584|
|7b|pd1|512|512|0|pp|ROPE|Kcur-5|7|1.556|
|7b|pd1|512|512|0|pp|ROPE|Kcur-6|7|1.535|
|7b|pd1|512|512|0|pp|ROPE|Kcur-7|7|1.658|
|7b|pd1|512|512|0|pp|ROPE|Kcur-8|7|1.670|
|7b|pd1|512|512|0|pp|ROPE|Kcur-9|7|1.705|
|7b|pd1|512|512|0|pp|ROPE|Qcur-0|7|1.514|
|7b|pd1|512|512|0|pp|ROPE|Qcur-1|7|1.531|
|7b|pd1|512|512|0|pp|ROPE|Qcur-10|7|1.461|
|7b|pd1|512|512|0|pp|ROPE|Qcur-11|7|1.423|
|7b|pd1|512|512|0|pp|ROPE|Qcur-12|7|1.494|
|7b|pd1|512|512|0|pp|ROPE|Qcur-13|7|1.543|
|7b|pd1|512|512|0|pp|ROPE|Qcur-14|7|1.526|
|7b|pd1|512|512|0|pp|ROPE|Qcur-15|7|1.534|
|7b|pd1|512|512|0|pp|ROPE|Qcur-16|7|1.450|
|7b|pd1|512|512|0|pp|ROPE|Qcur-17|7|1.538|
|7b|pd1|512|512|0|pp|ROPE|Qcur-18|7|1.584|
|7b|pd1|512|512|0|pp|ROPE|Qcur-19|7|1.535|
|7b|pd1|512|512|0|pp|ROPE|Qcur-2|7|1.574|
|7b|pd1|512|512|0|pp|ROPE|Qcur-20|7|1.615|
|7b|pd1|512|512|0|pp|ROPE|Qcur-21|7|1.441|
|7b|pd1|512|512|0|pp|ROPE|Qcur-22|7|1.485|
|7b|pd1|512|512|0|pp|ROPE|Qcur-23|7|1.376|
|7b|pd1|512|512|0|pp|ROPE|Qcur-24|7|1.485|
|7b|pd1|512|512|0|pp|ROPE|Qcur-25|7|1.512|
|7b|pd1|512|512|0|pp|ROPE|Qcur-26|7|1.440|
|7b|pd1|512|512|0|pp|ROPE|Qcur-27|7|1.530|
|7b|pd1|512|512|0|pp|ROPE|Qcur-28|7|1.585|
|7b|pd1|512|512|0|pp|ROPE|Qcur-29|7|1.469|
|7b|pd1|512|512|0|pp|ROPE|Qcur-3|7|1.564|
|7b|pd1|512|512|0|pp|ROPE|Qcur-30|7|1.400|
|7b|pd1|512|512|0|pp|ROPE|Qcur-31|7|1.553|
|7b|pd1|512|512|0|pp|ROPE|Qcur-4|7|1.597|
|7b|pd1|512|512|0|pp|ROPE|Qcur-5|7|1.587|
|7b|pd1|512|512|0|pp|ROPE|Qcur-6|7|1.428|
|7b|pd1|512|512|0|pp|ROPE|Qcur-7|7|1.578|
|7b|pd1|512|512|0|pp|ROPE|Qcur-8|7|1.557|
|7b|pd1|512|512|0|pp|ROPE|Qcur-9|7|1.615|
|7b|pd1|512|512|0|tg|GET_ROWS|embd|32|0.394|
|7b|pd1|512|512|0|tg|GET_ROWS|node_988|32|0.078|
|7b|pd1|512|512|0|tg|GET_ROWS|node_989|32|0.056|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-0|32|0.436|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-1|32|0.396|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-10|32|0.464|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-11|32|0.454|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-12|32|0.442|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-13|32|0.418|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-14|32|0.458|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-15|32|0.409|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-16|32|0.396|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-17|32|0.432|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-18|32|0.405|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-19|32|0.426|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-2|32|0.422|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-20|32|0.422|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-21|32|0.405|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-22|32|0.402|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-23|32|0.426|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-24|32|0.447|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-25|32|0.446|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-26|32|0.424|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-27|32|0.410|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-28|32|0.416|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-29|32|0.426|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-3|32|0.416|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-30|32|0.397|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-31|32|0.385|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-4|32|0.399|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-5|32|0.422|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-6|32|0.415|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-7|32|0.401|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-8|32|0.452|
|7b|pd1|512|512|0|tg|GLU|ffn_swiglu-9|32|0.414|
|7b|pd1|512|512|0|tg|MUL_MAT|result_output|32|113.042|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|223.010|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|222.570|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|215.503|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|215.025|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|229.218|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|213.766|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|216.583|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|213.896|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|215.357|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|217.610|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|216.145|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|215.963|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|217.465|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|221.513|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|218.702|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|220.655|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|215.261|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|211.648|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|211.891|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|215.432|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|213.302|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_706|7|213.582|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_737|7|216.091|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_768|7|214.479|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_799|7|214.040|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_830|7|212.577|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|222.189|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_861|7|213.174|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_892|7|212.652|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_923|7|213.077|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_954|7|213.454|
|7b|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_985|7|213.056|
|7b|pd1|512|512|1|pp|GET_ROWS|embd|8|3.136|
|7b|pd1|512|512|1|pp|GET_ROWS|node_988|8|0.025|
|7b|pd1|512|512|1|pp|GET_ROWS|node_989|8|0.018|
|7b|pd1|512|512|1|pp|MUL_MAT|result_output|8|29.145|
|7b|pd1|512|512|1|pp|ROPE|Kcur-0|7|1.623|
|7b|pd1|512|512|1|pp|ROPE|Kcur-1|7|1.668|
|7b|pd1|512|512|1|pp|ROPE|Kcur-10|7|1.580|
|7b|pd1|512|512|1|pp|ROPE|Kcur-11|7|1.653|
|7b|pd1|512|512|1|pp|ROPE|Kcur-12|7|1.618|
|7b|pd1|512|512|1|pp|ROPE|Kcur-13|7|1.583|
|7b|pd1|512|512|1|pp|ROPE|Kcur-14|7|1.665|
|7b|pd1|512|512|1|pp|ROPE|Kcur-15|7|1.707|
|7b|pd1|512|512|1|pp|ROPE|Kcur-16|7|1.708|
|7b|pd1|512|512|1|pp|ROPE|Kcur-17|7|1.590|
|7b|pd1|512|512|1|pp|ROPE|Kcur-18|7|1.565|
|7b|pd1|512|512|1|pp|ROPE|Kcur-19|7|1.579|
|7b|pd1|512|512|1|pp|ROPE|Kcur-2|7|1.589|
|7b|pd1|512|512|1|pp|ROPE|Kcur-20|7|1.475|
|7b|pd1|512|512|1|pp|ROPE|Kcur-21|7|1.576|
|7b|pd1|512|512|1|pp|ROPE|Kcur-22|7|1.619|
|7b|pd1|512|512|1|pp|ROPE|Kcur-23|7|1.701|
|7b|pd1|512|512|1|pp|ROPE|Kcur-24|7|1.575|
|7b|pd1|512|512|1|pp|ROPE|Kcur-25|7|1.609|
|7b|pd1|512|512|1|pp|ROPE|Kcur-26|7|1.487|
|7b|pd1|512|512|1|pp|ROPE|Kcur-27|7|1.508|
|7b|pd1|512|512|1|pp|ROPE|Kcur-28|7|1.589|
|7b|pd1|512|512|1|pp|ROPE|Kcur-29|7|1.647|
|7b|pd1|512|512|1|pp|ROPE|Kcur-3|7|1.610|
|7b|pd1|512|512|1|pp|ROPE|Kcur-30|7|1.546|
|7b|pd1|512|512|1|pp|ROPE|Kcur-31|7|1.494|
|7b|pd1|512|512|1|pp|ROPE|Kcur-4|7|1.559|
|7b|pd1|512|512|1|pp|ROPE|Kcur-5|7|1.528|
|7b|pd1|512|512|1|pp|ROPE|Kcur-6|7|1.645|
|7b|pd1|512|512|1|pp|ROPE|Kcur-7|7|1.541|
|7b|pd1|512|512|1|pp|ROPE|Kcur-8|7|1.548|
|7b|pd1|512|512|1|pp|ROPE|Kcur-9|7|1.565|
|7b|pd1|512|512|1|pp|ROPE|Qcur-0|7|1.549|
|7b|pd1|512|512|1|pp|ROPE|Qcur-1|7|1.612|
|7b|pd1|512|512|1|pp|ROPE|Qcur-10|7|1.487|
|7b|pd1|512|512|1|pp|ROPE|Qcur-11|7|1.577|
|7b|pd1|512|512|1|pp|ROPE|Qcur-12|7|1.531|
|7b|pd1|512|512|1|pp|ROPE|Qcur-13|7|1.459|
|7b|pd1|512|512|1|pp|ROPE|Qcur-14|7|1.571|
|7b|pd1|512|512|1|pp|ROPE|Qcur-15|7|1.547|
|7b|pd1|512|512|1|pp|ROPE|Qcur-16|7|1.568|
|7b|pd1|512|512|1|pp|ROPE|Qcur-17|7|1.477|
|7b|pd1|512|512|1|pp|ROPE|Qcur-18|7|1.508|
|7b|pd1|512|512|1|pp|ROPE|Qcur-19|7|1.441|
|7b|pd1|512|512|1|pp|ROPE|Qcur-2|7|1.679|
|7b|pd1|512|512|1|pp|ROPE|Qcur-20|7|1.406|
|7b|pd1|512|512|1|pp|ROPE|Qcur-21|7|1.494|
|7b|pd1|512|512|1|pp|ROPE|Qcur-22|7|1.517|
|7b|pd1|512|512|1|pp|ROPE|Qcur-23|7|1.587|
|7b|pd1|512|512|1|pp|ROPE|Qcur-24|7|1.545|
|7b|pd1|512|512|1|pp|ROPE|Qcur-25|7|1.465|
|7b|pd1|512|512|1|pp|ROPE|Qcur-26|7|1.595|
|7b|pd1|512|512|1|pp|ROPE|Qcur-27|7|1.610|
|7b|pd1|512|512|1|pp|ROPE|Qcur-28|7|1.543|
|7b|pd1|512|512|1|pp|ROPE|Qcur-29|7|1.419|
|7b|pd1|512|512|1|pp|ROPE|Qcur-3|7|1.590|
|7b|pd1|512|512|1|pp|ROPE|Qcur-30|7|1.418|
|7b|pd1|512|512|1|pp|ROPE|Qcur-31|7|1.477|
|7b|pd1|512|512|1|pp|ROPE|Qcur-4|7|1.578|
|7b|pd1|512|512|1|pp|ROPE|Qcur-5|7|1.640|
|7b|pd1|512|512|1|pp|ROPE|Qcur-6|7|1.488|
|7b|pd1|512|512|1|pp|ROPE|Qcur-7|7|1.595|
|7b|pd1|512|512|1|pp|ROPE|Qcur-8|7|1.574|
|7b|pd1|512|512|1|pp|ROPE|Qcur-9|7|1.533|
|7b|pd1|512|512|1|tg|GET_ROWS|embd|32|0.324|
|7b|pd1|512|512|1|tg|GET_ROWS|node_988|32|0.078|
|7b|pd1|512|512|1|tg|GET_ROWS|node_989|32|0.056|
|7b|pd1|512|512|1|tg|MUL_MAT|result_output|32|113.788|
|7b|row|128|128|0|pp|GET_ROWS|embd|29|2.682|
|7b|row|128|128|0|pp|GET_ROWS|node_988|29|0.088|
|7b|row|128|128|0|pp|GET_ROWS|node_989|29|0.055|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-0|29|15.639|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-1|29|15.427|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-10|29|15.417|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-11|29|15.576|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-12|29|15.562|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-13|29|15.703|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-14|29|15.494|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-15|29|15.408|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-16|29|15.569|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-17|29|15.419|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-18|29|15.520|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-19|29|15.575|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-2|29|15.367|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-20|29|15.520|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-21|29|15.423|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-22|29|15.502|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-23|29|15.705|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-24|29|15.599|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-25|29|15.426|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-26|29|15.608|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-27|29|15.607|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-28|29|15.603|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-29|29|15.535|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-3|29|15.566|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-30|29|15.665|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-31|29|0.519|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-4|29|15.377|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-5|29|15.434|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-6|29|15.583|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-7|29|15.555|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-8|29|15.349|
|7b|row|128|128|0|pp|GLU|ffn_swiglu-9|29|15.429|
|7b|row|128|128|0|pp|MUL|attn_norm-0|29|1.976|
|7b|row|128|128|0|pp|MUL|attn_norm-1|29|2.129|
|7b|row|128|128|0|pp|MUL|attn_norm-10|29|2.061|
|7b|row|128|128|0|pp|MUL|attn_norm-11|29|2.019|
|7b|row|128|128|0|pp|MUL|attn_norm-12|29|2.154|
|7b|row|128|128|0|pp|MUL|attn_norm-13|29|1.941|
|7b|row|128|128|0|pp|MUL|attn_norm-14|29|1.893|
|7b|row|128|128|0|pp|MUL|attn_norm-15|29|2.016|
|7b|row|128|128|0|pp|MUL|attn_norm-16|29|1.997|
|7b|row|128|128|0|pp|MUL|attn_norm-17|29|1.977|
|7b|row|128|128|0|pp|MUL|attn_norm-18|29|2.047|
|7b|row|128|128|0|pp|MUL|attn_norm-19|29|1.865|
|7b|row|128|128|0|pp|MUL|attn_norm-2|29|1.951|
|7b|row|128|128|0|pp|MUL|attn_norm-20|29|2.155|
|7b|row|128|128|0|pp|MUL|attn_norm-21|29|2.215|
|7b|row|128|128|0|pp|MUL|attn_norm-22|29|2.086|
|7b|row|128|128|0|pp|MUL|attn_norm-23|29|1.986|
|7b|row|128|128|0|pp|MUL|attn_norm-24|29|2.074|
|7b|row|128|128|0|pp|MUL|attn_norm-25|29|1.909|
|7b|row|128|128|0|pp|MUL|attn_norm-26|29|1.928|
|7b|row|128|128|0|pp|MUL|attn_norm-27|29|1.949|
|7b|row|128|128|0|pp|MUL|attn_norm-28|29|2.218|
|7b|row|128|128|0|pp|MUL|attn_norm-29|29|1.993|
|7b|row|128|128|0|pp|MUL|attn_norm-3|29|2.053|
|7b|row|128|128|0|pp|MUL|attn_norm-30|29|2.062|
|7b|row|128|128|0|pp|MUL|attn_norm-31|29|2.007|
|7b|row|128|128|0|pp|MUL|attn_norm-4|29|2.074|
|7b|row|128|128|0|pp|MUL|attn_norm-5|29|2.000|
|7b|row|128|128|0|pp|MUL|attn_norm-6|29|1.982|
|7b|row|128|128|0|pp|MUL|attn_norm-7|29|1.963|
|7b|row|128|128|0|pp|MUL|attn_norm-8|29|2.011|
|7b|row|128|128|0|pp|MUL|attn_norm-9|29|2.069|
|7b|row|128|128|0|pp|MUL|ffn_norm-0|29|1.946|
|7b|row|128|128|0|pp|MUL|ffn_norm-1|29|1.818|
|7b|row|128|128|0|pp|MUL|ffn_norm-10|29|1.791|
|7b|row|128|128|0|pp|MUL|ffn_norm-11|29|1.975|
|7b|row|128|128|0|pp|MUL|ffn_norm-12|29|1.581|
|7b|row|128|128|0|pp|MUL|ffn_norm-13|29|1.941|
|7b|row|128|128|0|pp|MUL|ffn_norm-14|29|1.779|
|7b|row|128|128|0|pp|MUL|ffn_norm-15|29|2.002|
|7b|row|128|128|0|pp|MUL|ffn_norm-16|29|1.992|
|7b|row|128|128|0|pp|MUL|ffn_norm-17|29|1.847|
|7b|row|128|128|0|pp|MUL|ffn_norm-18|29|1.885|
|7b|row|128|128|0|pp|MUL|ffn_norm-19|29|1.649|
|7b|row|128|128|0|pp|MUL|ffn_norm-2|29|1.901|
|7b|row|128|128|0|pp|MUL|ffn_norm-20|29|1.857|
|7b|row|128|128|0|pp|MUL|ffn_norm-21|29|1.652|
|7b|row|128|128|0|pp|MUL|ffn_norm-22|29|1.745|
|7b|row|128|128|0|pp|MUL|ffn_norm-23|29|1.898|
|7b|row|128|128|0|pp|MUL|ffn_norm-24|29|1.694|
|7b|row|128|128|0|pp|MUL|ffn_norm-25|29|1.907|
|7b|row|128|128|0|pp|MUL|ffn_norm-26|29|1.975|
|7b|row|128|128|0|pp|MUL|ffn_norm-27|29|1.631|
|7b|row|128|128|0|pp|MUL|ffn_norm-28|29|1.887|
|7b|row|128|128|0|pp|MUL|ffn_norm-29|29|1.674|
|7b|row|128|128|0|pp|MUL|ffn_norm-3|29|1.878|
|7b|row|128|128|0|pp|MUL|ffn_norm-30|29|1.977|
|7b|row|128|128|0|pp|MUL|ffn_norm-31|29|0.359|
|7b|row|128|128|0|pp|MUL|ffn_norm-4|29|1.772|
|7b|row|128|128|0|pp|MUL|ffn_norm-5|29|1.938|
|7b|row|128|128|0|pp|MUL|ffn_norm-6|29|1.827|
|7b|row|128|128|0|pp|MUL|ffn_norm-7|29|1.820|
|7b|row|128|128|0|pp|MUL|ffn_norm-8|29|1.814|
|7b|row|128|128|0|pp|MUL|ffn_norm-9|29|1.966|
|7b|row|128|128|0|pp|MUL|result_norm|29|0.388|
|7b|row|128|128|0|pp|MUL_MAT|result_output|29|105.294|
|7b|row|128|128|0|tg|GET_ROWS|embd|32|0.371|
|7b|row|128|128|0|tg|GET_ROWS|node_988|32|0.085|
|7b|row|128|128|0|tg|GET_ROWS|node_989|32|0.056|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-0|32|0.503|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-1|32|0.592|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-10|32|0.604|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-11|32|0.601|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-12|32|0.572|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-13|32|0.630|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-14|32|0.640|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-15|32|0.656|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-16|32|0.621|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-17|32|0.578|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-18|32|0.548|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-19|32|0.589|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-2|32|0.618|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-20|32|0.642|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-21|32|0.578|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-22|32|0.558|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-23|32|0.630|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-24|32|0.517|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-25|32|0.613|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-26|32|0.606|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-27|32|0.555|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-28|32|0.627|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-29|32|0.607|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-3|32|0.500|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-30|32|0.579|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-31|32|0.659|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-4|32|0.636|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-5|32|0.588|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-6|32|0.628|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-7|32|0.615|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-8|32|0.706|
|7b|row|128|128|0|tg|GLU|ffn_swiglu-9|32|0.662|
|7b|row|128|128|0|tg|MUL|attn_norm-0|32|0.283|
|7b|row|128|128|0|tg|MUL|attn_norm-1|32|0.315|
|7b|row|128|128|0|tg|MUL|attn_norm-10|32|0.391|
|7b|row|128|128|0|tg|MUL|attn_norm-11|32|0.336|
|7b|row|128|128|0|tg|MUL|attn_norm-12|32|0.338|
|7b|row|128|128|0|tg|MUL|attn_norm-13|32|0.346|
|7b|row|128|128|0|tg|MUL|attn_norm-14|32|0.328|
|7b|row|128|128|0|tg|MUL|attn_norm-15|32|0.327|
|7b|row|128|128|0|tg|MUL|attn_norm-16|32|0.346|
|7b|row|128|128|0|tg|MUL|attn_norm-17|32|0.369|
|7b|row|128|128|0|tg|MUL|attn_norm-18|32|0.356|
|7b|row|128|128|0|tg|MUL|attn_norm-19|32|0.377|
|7b|row|128|128|0|tg|MUL|attn_norm-2|32|0.379|
|7b|row|128|128|0|tg|MUL|attn_norm-20|32|0.368|
|7b|row|128|128|0|tg|MUL|attn_norm-21|32|0.398|
|7b|row|128|128|0|tg|MUL|attn_norm-22|32|0.310|
|7b|row|128|128|0|tg|MUL|attn_norm-23|32|0.395|
|7b|row|128|128|0|tg|MUL|attn_norm-24|32|0.381|
|7b|row|128|128|0|tg|MUL|attn_norm-25|32|0.431|
|7b|row|128|128|0|tg|MUL|attn_norm-26|32|0.317|
|7b|row|128|128|0|tg|MUL|attn_norm-27|32|0.351|
|7b|row|128|128|0|tg|MUL|attn_norm-28|32|0.358|
|7b|row|128|128|0|tg|MUL|attn_norm-29|32|0.381|
|7b|row|128|128|0|tg|MUL|attn_norm-3|32|0.355|
|7b|row|128|128|0|tg|MUL|attn_norm-30|32|0.368|
|7b|row|128|128|0|tg|MUL|attn_norm-31|32|0.373|
|7b|row|128|128|0|tg|MUL|attn_norm-4|32|0.313|
|7b|row|128|128|0|tg|MUL|attn_norm-5|32|0.341|
|7b|row|128|128|0|tg|MUL|attn_norm-6|32|0.334|
|7b|row|128|128|0|tg|MUL|attn_norm-7|32|0.357|
|7b|row|128|128|0|tg|MUL|attn_norm-8|32|0.371|
|7b|row|128|128|0|tg|MUL|attn_norm-9|32|0.372|
|7b|row|128|128|0|tg|MUL|ffn_norm-0|32|0.349|
|7b|row|128|128|0|tg|MUL|ffn_norm-1|32|0.370|
|7b|row|128|128|0|tg|MUL|ffn_norm-10|32|0.368|
|7b|row|128|128|0|tg|MUL|ffn_norm-11|32|0.392|
|7b|row|128|128|0|tg|MUL|ffn_norm-12|32|0.373|
|7b|row|128|128|0|tg|MUL|ffn_norm-13|32|0.377|
|7b|row|128|128|0|tg|MUL|ffn_norm-14|32|0.356|
|7b|row|128|128|0|tg|MUL|ffn_norm-15|32|0.350|
|7b|row|128|128|0|tg|MUL|ffn_norm-16|32|0.384|
|7b|row|128|128|0|tg|MUL|ffn_norm-17|32|0.375|
|7b|row|128|128|0|tg|MUL|ffn_norm-18|32|0.340|
|7b|row|128|128|0|tg|MUL|ffn_norm-19|32|0.361|
|7b|row|128|128|0|tg|MUL|ffn_norm-2|32|0.352|
|7b|row|128|128|0|tg|MUL|ffn_norm-20|32|0.405|
|7b|row|128|128|0|tg|MUL|ffn_norm-21|32|0.334|
|7b|row|128|128|0|tg|MUL|ffn_norm-22|32|0.392|
|7b|row|128|128|0|tg|MUL|ffn_norm-23|32|0.341|
|7b|row|128|128|0|tg|MUL|ffn_norm-24|32|0.372|
|7b|row|128|128|0|tg|MUL|ffn_norm-25|32|0.399|
|7b|row|128|128|0|tg|MUL|ffn_norm-26|32|0.407|
|7b|row|128|128|0|tg|MUL|ffn_norm-27|32|0.408|
|7b|row|128|128|0|tg|MUL|ffn_norm-28|32|0.357|
|7b|row|128|128|0|tg|MUL|ffn_norm-29|32|0.399|
|7b|row|128|128|0|tg|MUL|ffn_norm-3|32|0.347|
|7b|row|128|128|0|tg|MUL|ffn_norm-30|32|0.339|
|7b|row|128|128|0|tg|MUL|ffn_norm-31|32|0.343|
|7b|row|128|128|0|tg|MUL|ffn_norm-4|32|0.404|
|7b|row|128|128|0|tg|MUL|ffn_norm-5|32|0.385|
|7b|row|128|128|0|tg|MUL|ffn_norm-6|32|0.369|
|7b|row|128|128|0|tg|MUL|ffn_norm-7|32|0.362|
|7b|row|128|128|0|tg|MUL|ffn_norm-8|32|0.348|
|7b|row|128|128|0|tg|MUL|ffn_norm-9|32|0.346|
|7b|row|128|128|0|tg|MUL|result_norm|32|0.359|
|7b|row|128|128|0|tg|MUL_MAT|result_output|32|114.441|
|7b|row|128|128|1|pp|GET_ROWS|embd|29|2.933|
|7b|row|128|128|1|pp|GET_ROWS|node_988|29|0.085|
|7b|row|128|128|1|pp|GET_ROWS|node_989|29|0.053|
|7b|row|128|128|1|pp|MUL|attn_norm-0|29|1.795|
|7b|row|128|128|1|pp|MUL|attn_norm-1|29|1.745|
|7b|row|128|128|1|pp|MUL|attn_norm-10|29|1.906|
|7b|row|128|128|1|pp|MUL|attn_norm-11|29|2.044|
|7b|row|128|128|1|pp|MUL|attn_norm-12|29|1.777|
|7b|row|128|128|1|pp|MUL|attn_norm-13|29|1.851|
|7b|row|128|128|1|pp|MUL|attn_norm-14|29|1.938|
|7b|row|128|128|1|pp|MUL|attn_norm-15|29|1.970|
|7b|row|128|128|1|pp|MUL|attn_norm-16|29|1.916|
|7b|row|128|128|1|pp|MUL|attn_norm-17|29|1.763|
|7b|row|128|128|1|pp|MUL|attn_norm-18|29|1.918|
|7b|row|128|128|1|pp|MUL|attn_norm-19|29|1.793|
|7b|row|128|128|1|pp|MUL|attn_norm-2|29|1.666|
|7b|row|128|128|1|pp|MUL|attn_norm-20|29|1.803|
|7b|row|128|128|1|pp|MUL|attn_norm-21|29|1.602|
|7b|row|128|128|1|pp|MUL|attn_norm-22|29|1.965|
|7b|row|128|128|1|pp|MUL|attn_norm-23|29|1.875|
|7b|row|128|128|1|pp|MUL|attn_norm-24|29|1.695|
|7b|row|128|128|1|pp|MUL|attn_norm-25|29|1.914|
|7b|row|128|128|1|pp|MUL|attn_norm-26|29|2.038|
|7b|row|128|128|1|pp|MUL|attn_norm-27|29|1.863|
|7b|row|128|128|1|pp|MUL|attn_norm-28|29|2.021|
|7b|row|128|128|1|pp|MUL|attn_norm-29|29|1.775|
|7b|row|128|128|1|pp|MUL|attn_norm-3|29|1.761|
|7b|row|128|128|1|pp|MUL|attn_norm-30|29|1.875|
|7b|row|128|128|1|pp|MUL|attn_norm-31|29|1.915|
|7b|row|128|128|1|pp|MUL|attn_norm-4|29|1.780|
|7b|row|128|128|1|pp|MUL|attn_norm-5|29|1.623|
|7b|row|128|128|1|pp|MUL|attn_norm-6|29|1.934|
|7b|row|128|128|1|pp|MUL|attn_norm-7|29|1.894|
|7b|row|128|128|1|pp|MUL|attn_norm-8|29|1.735|
|7b|row|128|128|1|pp|MUL|attn_norm-9|29|2.025|
|7b|row|128|128|1|pp|MUL|ffn_norm-0|29|1.719|
|7b|row|128|128|1|pp|MUL|ffn_norm-1|29|1.887|
|7b|row|128|128|1|pp|MUL|ffn_norm-10|29|1.839|
|7b|row|128|128|1|pp|MUL|ffn_norm-11|29|1.867|
|7b|row|128|128|1|pp|MUL|ffn_norm-12|29|1.875|
|7b|row|128|128|1|pp|MUL|ffn_norm-13|29|1.870|
|7b|row|128|128|1|pp|MUL|ffn_norm-14|29|1.893|
|7b|row|128|128|1|pp|MUL|ffn_norm-15|29|1.809|
|7b|row|128|128|1|pp|MUL|ffn_norm-16|29|1.651|
|7b|row|128|128|1|pp|MUL|ffn_norm-17|29|1.719|
|7b|row|128|128|1|pp|MUL|ffn_norm-18|29|1.840|
|7b|row|128|128|1|pp|MUL|ffn_norm-19|29|1.962|
|7b|row|128|128|1|pp|MUL|ffn_norm-2|29|1.755|
|7b|row|128|128|1|pp|MUL|ffn_norm-20|29|1.768|
|7b|row|128|128|1|pp|MUL|ffn_norm-21|29|1.751|
|7b|row|128|128|1|pp|MUL|ffn_norm-22|29|1.806|
|7b|row|128|128|1|pp|MUL|ffn_norm-23|29|1.835|
|7b|row|128|128|1|pp|MUL|ffn_norm-24|29|1.877|
|7b|row|128|128|1|pp|MUL|ffn_norm-25|29|1.712|
|7b|row|128|128|1|pp|MUL|ffn_norm-26|29|1.867|
|7b|row|128|128|1|pp|MUL|ffn_norm-27|29|2.030|
|7b|row|128|128|1|pp|MUL|ffn_norm-28|29|1.921|
|7b|row|128|128|1|pp|MUL|ffn_norm-29|29|1.853|
|7b|row|128|128|1|pp|MUL|ffn_norm-3|29|1.918|
|7b|row|128|128|1|pp|MUL|ffn_norm-30|29|1.806|
|7b|row|128|128|1|pp|MUL|ffn_norm-31|29|0.434|
|7b|row|128|128|1|pp|MUL|ffn_norm-4|29|1.982|
|7b|row|128|128|1|pp|MUL|ffn_norm-5|29|1.996|
|7b|row|128|128|1|pp|MUL|ffn_norm-6|29|1.857|
|7b|row|128|128|1|pp|MUL|ffn_norm-7|29|1.862|
|7b|row|128|128|1|pp|MUL|ffn_norm-8|29|1.743|
|7b|row|128|128|1|pp|MUL|ffn_norm-9|29|1.754|
|7b|row|128|128|1|pp|MUL|result_norm|29|0.336|
|7b|row|128|128|1|pp|MUL_MAT|result_output|29|103.632|
|7b|row|128|128|1|tg|GET_ROWS|embd|32|0.402|
|7b|row|128|128|1|tg|GET_ROWS|node_988|32|0.080|
|7b|row|128|128|1|tg|GET_ROWS|node_989|32|0.053|
|7b|row|128|128|1|tg|MUL|attn_norm-0|32|0.325|
|7b|row|128|128|1|tg|MUL|attn_norm-1|32|0.340|
|7b|row|128|128|1|tg|MUL|attn_norm-10|32|0.346|
|7b|row|128|128|1|tg|MUL|attn_norm-11|32|0.409|
|7b|row|128|128|1|tg|MUL|attn_norm-12|32|0.321|
|7b|row|128|128|1|tg|MUL|attn_norm-13|32|0.357|
|7b|row|128|128|1|tg|MUL|attn_norm-14|32|0.375|
|7b|row|128|128|1|tg|MUL|attn_norm-15|32|0.347|
|7b|row|128|128|1|tg|MUL|attn_norm-16|32|0.368|
|7b|row|128|128|1|tg|MUL|attn_norm-17|32|0.318|
|7b|row|128|128|1|tg|MUL|attn_norm-18|32|0.357|
|7b|row|128|128|1|tg|MUL|attn_norm-19|32|0.357|
|7b|row|128|128|1|tg|MUL|attn_norm-2|32|0.389|
|7b|row|128|128|1|tg|MUL|attn_norm-20|32|0.331|
|7b|row|128|128|1|tg|MUL|attn_norm-21|32|0.326|
|7b|row|128|128|1|tg|MUL|attn_norm-22|32|0.352|
|7b|row|128|128|1|tg|MUL|attn_norm-23|32|0.413|
|7b|row|128|128|1|tg|MUL|attn_norm-24|32|0.398|
|7b|row|128|128|1|tg|MUL|attn_norm-25|32|0.352|
|7b|row|128|128|1|tg|MUL|attn_norm-26|32|0.322|
|7b|row|128|128|1|tg|MUL|attn_norm-27|32|0.309|
|7b|row|128|128|1|tg|MUL|attn_norm-28|32|0.321|
|7b|row|128|128|1|tg|MUL|attn_norm-29|32|0.410|
|7b|row|128|128|1|tg|MUL|attn_norm-3|32|0.326|
|7b|row|128|128|1|tg|MUL|attn_norm-30|32|0.361|
|7b|row|128|128|1|tg|MUL|attn_norm-31|32|0.379|
|7b|row|128|128|1|tg|MUL|attn_norm-4|32|0.360|
|7b|row|128|128|1|tg|MUL|attn_norm-5|32|0.297|
|7b|row|128|128|1|tg|MUL|attn_norm-6|32|0.339|
|7b|row|128|128|1|tg|MUL|attn_norm-7|32|0.322|
|7b|row|128|128|1|tg|MUL|attn_norm-8|32|0.394|
|7b|row|128|128|1|tg|MUL|attn_norm-9|32|0.314|
|7b|row|128|128|1|tg|MUL|ffn_norm-0|32|0.364|
|7b|row|128|128|1|tg|MUL|ffn_norm-1|32|0.333|
|7b|row|128|128|1|tg|MUL|ffn_norm-10|32|0.337|
|7b|row|128|128|1|tg|MUL|ffn_norm-11|32|0.345|
|7b|row|128|128|1|tg|MUL|ffn_norm-12|32|0.391|
|7b|row|128|128|1|tg|MUL|ffn_norm-13|32|0.332|
|7b|row|128|128|1|tg|MUL|ffn_norm-14|32|0.364|
|7b|row|128|128|1|tg|MUL|ffn_norm-15|32|0.419|
|7b|row|128|128|1|tg|MUL|ffn_norm-16|32|0.370|
|7b|row|128|128|1|tg|MUL|ffn_norm-17|32|0.368|
|7b|row|128|128|1|tg|MUL|ffn_norm-18|32|0.362|
|7b|row|128|128|1|tg|MUL|ffn_norm-19|32|0.368|
|7b|row|128|128|1|tg|MUL|ffn_norm-2|32|0.393|
|7b|row|128|128|1|tg|MUL|ffn_norm-20|32|0.360|
|7b|row|128|128|1|tg|MUL|ffn_norm-21|32|0.394|
|7b|row|128|128|1|tg|MUL|ffn_norm-22|32|0.413|
|7b|row|128|128|1|tg|MUL|ffn_norm-23|32|0.332|
|7b|row|128|128|1|tg|MUL|ffn_norm-24|32|0.347|
|7b|row|128|128|1|tg|MUL|ffn_norm-25|32|0.317|
|7b|row|128|128|1|tg|MUL|ffn_norm-26|32|0.324|
|7b|row|128|128|1|tg|MUL|ffn_norm-27|32|0.404|
|7b|row|128|128|1|tg|MUL|ffn_norm-28|32|0.379|
|7b|row|128|128|1|tg|MUL|ffn_norm-29|32|0.362|
|7b|row|128|128|1|tg|MUL|ffn_norm-3|32|0.332|
|7b|row|128|128|1|tg|MUL|ffn_norm-30|32|0.407|
|7b|row|128|128|1|tg|MUL|ffn_norm-31|32|0.392|
|7b|row|128|128|1|tg|MUL|ffn_norm-4|32|0.369|
|7b|row|128|128|1|tg|MUL|ffn_norm-5|32|0.386|
|7b|row|128|128|1|tg|MUL|ffn_norm-6|32|0.354|
|7b|row|128|128|1|tg|MUL|ffn_norm-7|32|0.383|
|7b|row|128|128|1|tg|MUL|ffn_norm-8|32|0.348|
|7b|row|128|128|1|tg|MUL|ffn_norm-9|32|0.383|
|7b|row|128|128|1|tg|MUL|result_norm|32|0.382|
|7b|row|128|128|1|tg|MUL_MAT|result_output|32|114.222|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|216.903|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|225.215|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|222.430|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|217.379|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|225.882|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|227.390|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|221.351|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|228.255|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|231.149|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|231.056|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|230.182|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|236.539|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|235.722|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|229.929|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|229.826|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|220.024|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|226.043|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|225.642|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|223.866|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|222.999|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|231.095|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_706|7|217.738|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_737|7|217.320|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_768|7|223.264|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_799|7|219.182|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_830|7|218.571|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|219.494|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_861|7|223.216|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_892|7|223.601|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_923|7|219.789|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_954|7|219.112|
|7b|row|512|512|0|pp|FLASH_ATTN_EXT|node_985|7|228.581|
|7b|row|512|512|0|pp|GET_ROWS|embd|8|3.145|
|7b|row|512|512|0|pp|GET_ROWS|node_988|8|0.028|
|7b|row|512|512|0|pp|GET_ROWS|node_989|8|0.018|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-0|8|20.324|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-1|8|20.232|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-10|8|20.235|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-11|8|20.159|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-12|8|20.322|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-13|8|20.317|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-14|8|20.247|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-15|8|20.204|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-16|8|20.129|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-17|8|20.599|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-18|8|20.151|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-19|8|20.044|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-2|8|20.231|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-20|8|20.198|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-21|8|20.081|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-22|8|20.187|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-23|8|20.058|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-24|8|20.206|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-25|8|20.224|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-26|8|20.238|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-27|8|20.299|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-28|8|20.181|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-29|8|20.242|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-3|8|20.265|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-30|8|20.178|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-31|8|0.117|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-4|8|20.314|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-5|8|20.156|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-6|8|20.168|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-7|8|20.236|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-8|8|20.387|
|7b|row|512|512|0|pp|GLU|ffn_swiglu-9|8|20.182|
|7b|row|512|512|0|pp|MUL|attn_norm-0|8|1.530|
|7b|row|512|512|0|pp|MUL|attn_norm-1|8|1.468|
|7b|row|512|512|0|pp|MUL|attn_norm-10|8|1.509|
|7b|row|512|512|0|pp|MUL|attn_norm-11|8|1.490|
|7b|row|512|512|0|pp|MUL|attn_norm-12|8|1.400|
|7b|row|512|512|0|pp|MUL|attn_norm-13|8|1.404|
|7b|row|512|512|0|pp|MUL|attn_norm-14|8|1.475|
|7b|row|512|512|0|pp|MUL|attn_norm-15|8|1.452|
|7b|row|512|512|0|pp|MUL|attn_norm-16|8|1.388|
|7b|row|512|512|0|pp|MUL|attn_norm-17|8|1.421|
|7b|row|512|512|0|pp|MUL|attn_norm-18|8|1.362|
|7b|row|512|512|0|pp|MUL|attn_norm-19|8|1.400|
|7b|row|512|512|0|pp|MUL|attn_norm-2|8|1.428|
|7b|row|512|512|0|pp|MUL|attn_norm-20|8|1.462|
|7b|row|512|512|0|pp|MUL|attn_norm-21|8|1.514|
|7b|row|512|512|0|pp|MUL|attn_norm-22|8|1.532|
|7b|row|512|512|0|pp|MUL|attn_norm-23|8|1.427|
|7b|row|512|512|0|pp|MUL|attn_norm-24|8|1.470|
|7b|row|512|512|0|pp|MUL|attn_norm-25|8|1.408|
|7b|row|512|512|0|pp|MUL|attn_norm-26|8|1.505|
|7b|row|512|512|0|pp|MUL|attn_norm-27|8|1.503|
|7b|row|512|512|0|pp|MUL|attn_norm-28|8|1.521|
|7b|row|512|512|0|pp|MUL|attn_norm-29|8|1.424|
|7b|row|512|512|0|pp|MUL|attn_norm-3|8|1.419|
|7b|row|512|512|0|pp|MUL|attn_norm-30|8|1.428|
|7b|row|512|512|0|pp|MUL|attn_norm-31|8|1.460|
|7b|row|512|512|0|pp|MUL|attn_norm-4|8|1.433|
|7b|row|512|512|0|pp|MUL|attn_norm-5|8|1.438|
|7b|row|512|512|0|pp|MUL|attn_norm-6|8|1.397|
|7b|row|512|512|0|pp|MUL|attn_norm-7|8|1.593|
|7b|row|512|512|0|pp|MUL|attn_norm-8|8|1.510|
|7b|row|512|512|0|pp|MUL|attn_norm-9|8|1.469|
|7b|row|512|512|0|pp|MUL|ffn_norm-0|8|1.430|
|7b|row|512|512|0|pp|MUL|ffn_norm-1|8|1.336|
|7b|row|512|512|0|pp|MUL|ffn_norm-10|8|1.373|
|7b|row|512|512|0|pp|MUL|ffn_norm-11|8|1.381|
|7b|row|512|512|0|pp|MUL|ffn_norm-12|8|1.442|
|7b|row|512|512|0|pp|MUL|ffn_norm-13|8|1.446|
|7b|row|512|512|0|pp|MUL|ffn_norm-14|8|1.430|
|7b|row|512|512|0|pp|MUL|ffn_norm-15|8|1.400|
|7b|row|512|512|0|pp|MUL|ffn_norm-16|8|1.437|
|7b|row|512|512|0|pp|MUL|ffn_norm-17|8|1.375|
|7b|row|512|512|0|pp|MUL|ffn_norm-18|8|1.399|
|7b|row|512|512|0|pp|MUL|ffn_norm-19|8|1.413|
|7b|row|512|512|0|pp|MUL|ffn_norm-2|8|1.405|
|7b|row|512|512|0|pp|MUL|ffn_norm-20|8|1.383|
|7b|row|512|512|0|pp|MUL|ffn_norm-21|8|1.427|
|7b|row|512|512|0|pp|MUL|ffn_norm-22|8|1.435|
|7b|row|512|512|0|pp|MUL|ffn_norm-23|8|1.400|
|7b|row|512|512|0|pp|MUL|ffn_norm-24|8|1.365|
|7b|row|512|512|0|pp|MUL|ffn_norm-25|8|1.466|
|7b|row|512|512|0|pp|MUL|ffn_norm-26|8|1.384|
|7b|row|512|512|0|pp|MUL|ffn_norm-27|8|1.445|
|7b|row|512|512|0|pp|MUL|ffn_norm-28|8|1.448|
|7b|row|512|512|0|pp|MUL|ffn_norm-29|8|1.372|
|7b|row|512|512|0|pp|MUL|ffn_norm-3|8|1.474|
|7b|row|512|512|0|pp|MUL|ffn_norm-30|8|1.373|
|7b|row|512|512|0|pp|MUL|ffn_norm-31|8|0.081|
|7b|row|512|512|0|pp|MUL|ffn_norm-4|8|1.491|
|7b|row|512|512|0|pp|MUL|ffn_norm-5|8|1.462|
|7b|row|512|512|0|pp|MUL|ffn_norm-6|8|1.394|
|7b|row|512|512|0|pp|MUL|ffn_norm-7|8|1.373|
|7b|row|512|512|0|pp|MUL|ffn_norm-8|8|1.413|
|7b|row|512|512|0|pp|MUL|ffn_norm-9|8|1.467|
|7b|row|512|512|0|pp|MUL|result_norm|8|0.085|
|7b|row|512|512|0|pp|MUL_MAT|result_output|8|29.028|
|7b|row|512|512|0|pp|ROPE|Kcur-0|7|1.745|
|7b|row|512|512|0|pp|ROPE|Kcur-1|7|1.580|
|7b|row|512|512|0|pp|ROPE|Kcur-10|7|1.624|
|7b|row|512|512|0|pp|ROPE|Kcur-11|7|1.705|
|7b|row|512|512|0|pp|ROPE|Kcur-12|7|1.683|
|7b|row|512|512|0|pp|ROPE|Kcur-13|7|1.658|
|7b|row|512|512|0|pp|ROPE|Kcur-14|7|1.789|
|7b|row|512|512|0|pp|ROPE|Kcur-15|7|1.636|
|7b|row|512|512|0|pp|ROPE|Kcur-16|7|1.674|
|7b|row|512|512|0|pp|ROPE|Kcur-17|7|1.628|
|7b|row|512|512|0|pp|ROPE|Kcur-18|7|1.697|
|7b|row|512|512|0|pp|ROPE|Kcur-19|7|1.633|
|7b|row|512|512|0|pp|ROPE|Kcur-2|7|1.570|
|7b|row|512|512|0|pp|ROPE|Kcur-20|7|1.659|
|7b|row|512|512|0|pp|ROPE|Kcur-21|7|1.688|
|7b|row|512|512|0|pp|ROPE|Kcur-22|7|1.620|
|7b|row|512|512|0|pp|ROPE|Kcur-23|7|1.628|
|7b|row|512|512|0|pp|ROPE|Kcur-24|7|1.619|
|7b|row|512|512|0|pp|ROPE|Kcur-25|7|1.712|
|7b|row|512|512|0|pp|ROPE|Kcur-26|7|1.641|
|7b|row|512|512|0|pp|ROPE|Kcur-27|7|1.625|
|7b|row|512|512|0|pp|ROPE|Kcur-28|7|1.589|
|7b|row|512|512|0|pp|ROPE|Kcur-29|7|1.666|
|7b|row|512|512|0|pp|ROPE|Kcur-3|7|1.586|
|7b|row|512|512|0|pp|ROPE|Kcur-30|7|1.628|
|7b|row|512|512|0|pp|ROPE|Kcur-31|7|1.679|
|7b|row|512|512|0|pp|ROPE|Kcur-4|7|1.584|
|7b|row|512|512|0|pp|ROPE|Kcur-5|7|1.597|
|7b|row|512|512|0|pp|ROPE|Kcur-6|7|1.633|
|7b|row|512|512|0|pp|ROPE|Kcur-7|7|1.688|
|7b|row|512|512|0|pp|ROPE|Kcur-8|7|1.663|
|7b|row|512|512|0|pp|ROPE|Kcur-9|7|1.723|
|7b|row|512|512|0|pp|ROPE|Qcur-0|7|1.561|
|7b|row|512|512|0|pp|ROPE|Qcur-1|7|1.541|
|7b|row|512|512|0|pp|ROPE|Qcur-10|7|1.697|
|7b|row|512|512|0|pp|ROPE|Qcur-11|7|1.582|
|7b|row|512|512|0|pp|ROPE|Qcur-12|7|1.616|
|7b|row|512|512|0|pp|ROPE|Qcur-13|7|1.489|
|7b|row|512|512|0|pp|ROPE|Qcur-14|7|1.641|
|7b|row|512|512|0|pp|ROPE|Qcur-15|7|1.580|
|7b|row|512|512|0|pp|ROPE|Qcur-16|7|1.577|
|7b|row|512|512|0|pp|ROPE|Qcur-17|7|1.466|
|7b|row|512|512|0|pp|ROPE|Qcur-18|7|1.615|
|7b|row|512|512|0|pp|ROPE|Qcur-19|7|1.559|
|7b|row|512|512|0|pp|ROPE|Qcur-2|7|1.565|
|7b|row|512|512|0|pp|ROPE|Qcur-20|7|1.504|
|7b|row|512|512|0|pp|ROPE|Qcur-21|7|1.564|
|7b|row|512|512|0|pp|ROPE|Qcur-22|7|1.476|
|7b|row|512|512|0|pp|ROPE|Qcur-23|7|1.456|
|7b|row|512|512|0|pp|ROPE|Qcur-24|7|1.509|
|7b|row|512|512|0|pp|ROPE|Qcur-25|7|1.542|
|7b|row|512|512|0|pp|ROPE|Qcur-26|7|1.414|
|7b|row|512|512|0|pp|ROPE|Qcur-27|7|1.432|
|7b|row|512|512|0|pp|ROPE|Qcur-28|7|1.470|
|7b|row|512|512|0|pp|ROPE|Qcur-29|7|1.490|
|7b|row|512|512|0|pp|ROPE|Qcur-3|7|1.473|
|7b|row|512|512|0|pp|ROPE|Qcur-30|7|1.521|
|7b|row|512|512|0|pp|ROPE|Qcur-31|7|1.566|
|7b|row|512|512|0|pp|ROPE|Qcur-4|7|1.551|
|7b|row|512|512|0|pp|ROPE|Qcur-5|7|1.496|
|7b|row|512|512|0|pp|ROPE|Qcur-6|7|1.522|
|7b|row|512|512|0|pp|ROPE|Qcur-7|7|1.571|
|7b|row|512|512|0|pp|ROPE|Qcur-8|7|1.619|
|7b|row|512|512|0|pp|ROPE|Qcur-9|7|1.535|
|7b|row|512|512|0|tg|GET_ROWS|embd|32|0.374|
|7b|row|512|512|0|tg|GET_ROWS|node_988|32|0.074|
|7b|row|512|512|0|tg|GET_ROWS|node_989|32|0.056|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-0|32|0.445|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-1|32|0.458|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-10|32|0.508|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-11|32|0.509|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-12|32|0.476|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-13|32|0.549|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-14|32|0.451|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-15|32|0.461|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-16|32|0.471|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-17|32|0.481|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-18|32|0.426|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-19|32|0.433|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-2|32|0.485|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-20|32|0.475|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-21|32|0.415|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-22|32|0.478|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-23|32|0.468|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-24|32|0.480|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-25|32|0.487|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-26|32|0.446|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-27|32|0.413|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-28|32|0.452|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-29|32|0.448|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-3|32|0.415|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-30|32|0.438|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-31|32|0.494|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-4|32|0.439|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-5|32|0.459|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-6|32|0.415|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-7|32|0.470|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-8|32|0.429|
|7b|row|512|512|0|tg|GLU|ffn_swiglu-9|32|0.477|
|7b|row|512|512|0|tg|MUL|attn_norm-0|32|0.291|
|7b|row|512|512|0|tg|MUL|attn_norm-1|32|0.313|
|7b|row|512|512|0|tg|MUL|attn_norm-10|32|0.283|
|7b|row|512|512|0|tg|MUL|attn_norm-11|32|0.305|
|7b|row|512|512|0|tg|MUL|attn_norm-12|32|0.298|
|7b|row|512|512|0|tg|MUL|attn_norm-13|32|0.302|
|7b|row|512|512|0|tg|MUL|attn_norm-14|32|0.367|
|7b|row|512|512|0|tg|MUL|attn_norm-15|32|0.303|
|7b|row|512|512|0|tg|MUL|attn_norm-16|32|0.270|
|7b|row|512|512|0|tg|MUL|attn_norm-17|32|0.287|
|7b|row|512|512|0|tg|MUL|attn_norm-18|32|0.310|
|7b|row|512|512|0|tg|MUL|attn_norm-19|32|0.358|
|7b|row|512|512|0|tg|MUL|attn_norm-2|32|0.305|
|7b|row|512|512|0|tg|MUL|attn_norm-20|32|0.364|
|7b|row|512|512|0|tg|MUL|attn_norm-21|32|0.286|
|7b|row|512|512|0|tg|MUL|attn_norm-22|32|0.349|
|7b|row|512|512|0|tg|MUL|attn_norm-23|32|0.309|
|7b|row|512|512|0|tg|MUL|attn_norm-24|32|0.308|
|7b|row|512|512|0|tg|MUL|attn_norm-25|32|0.299|
|7b|row|512|512|0|tg|MUL|attn_norm-26|32|0.330|
|7b|row|512|512|0|tg|MUL|attn_norm-27|32|0.281|
|7b|row|512|512|0|tg|MUL|attn_norm-28|32|0.293|
|7b|row|512|512|0|tg|MUL|attn_norm-29|32|0.297|
|7b|row|512|512|0|tg|MUL|attn_norm-3|32|0.333|
|7b|row|512|512|0|tg|MUL|attn_norm-30|32|0.332|
|7b|row|512|512|0|tg|MUL|attn_norm-31|32|0.386|
|7b|row|512|512|0|tg|MUL|attn_norm-4|32|0.368|
|7b|row|512|512|0|tg|MUL|attn_norm-5|32|0.313|
|7b|row|512|512|0|tg|MUL|attn_norm-6|32|0.284|
|7b|row|512|512|0|tg|MUL|attn_norm-7|32|0.390|
|7b|row|512|512|0|tg|MUL|attn_norm-8|32|0.300|
|7b|row|512|512|0|tg|MUL|attn_norm-9|32|0.297|
|7b|row|512|512|0|tg|MUL|ffn_norm-0|32|0.327|
|7b|row|512|512|0|tg|MUL|ffn_norm-1|32|0.304|
|7b|row|512|512|0|tg|MUL|ffn_norm-10|32|0.276|
|7b|row|512|512|0|tg|MUL|ffn_norm-11|32|0.279|
|7b|row|512|512|0|tg|MUL|ffn_norm-12|32|0.339|
|7b|row|512|512|0|tg|MUL|ffn_norm-13|32|0.327|
|7b|row|512|512|0|tg|MUL|ffn_norm-14|32|0.310|
|7b|row|512|512|0|tg|MUL|ffn_norm-15|32|0.301|
|7b|row|512|512|0|tg|MUL|ffn_norm-16|32|0.339|
|7b|row|512|512|0|tg|MUL|ffn_norm-17|32|0.314|
|7b|row|512|512|0|tg|MUL|ffn_norm-18|32|0.314|
|7b|row|512|512|0|tg|MUL|ffn_norm-19|32|0.282|
|7b|row|512|512|0|tg|MUL|ffn_norm-2|32|0.326|
|7b|row|512|512|0|tg|MUL|ffn_norm-20|32|0.315|
|7b|row|512|512|0|tg|MUL|ffn_norm-21|32|0.317|
|7b|row|512|512|0|tg|MUL|ffn_norm-22|32|0.281|
|7b|row|512|512|0|tg|MUL|ffn_norm-23|32|0.342|
|7b|row|512|512|0|tg|MUL|ffn_norm-24|32|0.319|
|7b|row|512|512|0|tg|MUL|ffn_norm-25|32|0.329|
|7b|row|512|512|0|tg|MUL|ffn_norm-26|32|0.307|
|7b|row|512|512|0|tg|MUL|ffn_norm-27|32|0.315|
|7b|row|512|512|0|tg|MUL|ffn_norm-28|32|0.327|
|7b|row|512|512|0|tg|MUL|ffn_norm-29|32|0.356|
|7b|row|512|512|0|tg|MUL|ffn_norm-3|32|0.303|
|7b|row|512|512|0|tg|MUL|ffn_norm-30|32|0.365|
|7b|row|512|512|0|tg|MUL|ffn_norm-31|32|0.288|
|7b|row|512|512|0|tg|MUL|ffn_norm-4|32|0.319|
|7b|row|512|512|0|tg|MUL|ffn_norm-5|32|0.306|
|7b|row|512|512|0|tg|MUL|ffn_norm-6|32|0.349|
|7b|row|512|512|0|tg|MUL|ffn_norm-7|32|0.309|
|7b|row|512|512|0|tg|MUL|ffn_norm-8|32|0.339|
|7b|row|512|512|0|tg|MUL|ffn_norm-9|32|0.317|
|7b|row|512|512|0|tg|MUL|result_norm|32|0.281|
|7b|row|512|512|0|tg|MUL_MAT|result_output|32|113.224|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|230.407|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|216.205|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|230.602|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|248.449|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|223.075|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|220.934|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|220.910|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|217.177|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|225.448|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|230.659|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|216.273|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|218.404|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|217.799|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|229.628|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|241.336|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|231.651|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|224.968|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|224.647|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|218.349|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|231.071|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|230.090|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_706|7|225.110|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_737|7|218.492|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_768|7|222.086|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_799|7|219.935|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_830|7|229.875|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|226.531|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_861|7|226.641|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_892|7|236.487|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_923|7|227.848|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_954|7|227.682|
|7b|row|512|512|1|pp|FLASH_ATTN_EXT|node_985|7|224.776|
|7b|row|512|512|1|pp|GET_ROWS|embd|8|3.088|
|7b|row|512|512|1|pp|GET_ROWS|node_988|8|0.025|
|7b|row|512|512|1|pp|GET_ROWS|node_989|8|0.017|
|7b|row|512|512|1|pp|MUL|attn_norm-0|8|1.421|
|7b|row|512|512|1|pp|MUL|attn_norm-1|8|1.490|
|7b|row|512|512|1|pp|MUL|attn_norm-10|8|1.362|
|7b|row|512|512|1|pp|MUL|attn_norm-11|8|1.367|
|7b|row|512|512|1|pp|MUL|attn_norm-12|8|1.378|
|7b|row|512|512|1|pp|MUL|attn_norm-13|8|1.365|
|7b|row|512|512|1|pp|MUL|attn_norm-14|8|1.333|
|7b|row|512|512|1|pp|MUL|attn_norm-15|8|1.387|
|7b|row|512|512|1|pp|MUL|attn_norm-16|8|1.467|
|7b|row|512|512|1|pp|MUL|attn_norm-17|8|1.540|
|7b|row|512|512|1|pp|MUL|attn_norm-18|8|1.461|
|7b|row|512|512|1|pp|MUL|attn_norm-19|8|1.417|
|7b|row|512|512|1|pp|MUL|attn_norm-2|8|1.442|
|7b|row|512|512|1|pp|MUL|attn_norm-20|8|1.468|
|7b|row|512|512|1|pp|MUL|attn_norm-21|8|1.430|
|7b|row|512|512|1|pp|MUL|attn_norm-22|8|1.493|
|7b|row|512|512|1|pp|MUL|attn_norm-23|8|1.536|
|7b|row|512|512|1|pp|MUL|attn_norm-24|8|1.411|
|7b|row|512|512|1|pp|MUL|attn_norm-25|8|1.499|
|7b|row|512|512|1|pp|MUL|attn_norm-26|8|1.368|
|7b|row|512|512|1|pp|MUL|attn_norm-27|8|1.505|
|7b|row|512|512|1|pp|MUL|attn_norm-28|8|1.364|
|7b|row|512|512|1|pp|MUL|attn_norm-29|8|1.357|
|7b|row|512|512|1|pp|MUL|attn_norm-3|8|1.533|
|7b|row|512|512|1|pp|MUL|attn_norm-30|8|1.381|
|7b|row|512|512|1|pp|MUL|attn_norm-31|8|1.475|
|7b|row|512|512|1|pp|MUL|attn_norm-4|8|1.408|
|7b|row|512|512|1|pp|MUL|attn_norm-5|8|1.364|
|7b|row|512|512|1|pp|MUL|attn_norm-6|8|1.397|
|7b|row|512|512|1|pp|MUL|attn_norm-7|8|1.399|
|7b|row|512|512|1|pp|MUL|attn_norm-8|8|1.399|
|7b|row|512|512|1|pp|MUL|attn_norm-9|8|1.408|
|7b|row|512|512|1|pp|MUL|ffn_norm-0|8|1.492|
|7b|row|512|512|1|pp|MUL|ffn_norm-1|8|1.596|
|7b|row|512|512|1|pp|MUL|ffn_norm-10|8|1.473|
|7b|row|512|512|1|pp|MUL|ffn_norm-11|8|1.474|
|7b|row|512|512|1|pp|MUL|ffn_norm-12|8|1.594|
|7b|row|512|512|1|pp|MUL|ffn_norm-13|8|1.514|
|7b|row|512|512|1|pp|MUL|ffn_norm-14|8|1.533|
|7b|row|512|512|1|pp|MUL|ffn_norm-15|8|1.560|
|7b|row|512|512|1|pp|MUL|ffn_norm-16|8|1.579|
|7b|row|512|512|1|pp|MUL|ffn_norm-17|8|1.449|
|7b|row|512|512|1|pp|MUL|ffn_norm-18|8|1.460|
|7b|row|512|512|1|pp|MUL|ffn_norm-19|8|1.548|
|7b|row|512|512|1|pp|MUL|ffn_norm-2|8|1.474|
|7b|row|512|512|1|pp|MUL|ffn_norm-20|8|1.461|
|7b|row|512|512|1|pp|MUL|ffn_norm-21|8|1.554|
|7b|row|512|512|1|pp|MUL|ffn_norm-22|8|1.453|
|7b|row|512|512|1|pp|MUL|ffn_norm-23|8|1.594|
|7b|row|512|512|1|pp|MUL|ffn_norm-24|8|1.434|
|7b|row|512|512|1|pp|MUL|ffn_norm-25|8|1.617|
|7b|row|512|512|1|pp|MUL|ffn_norm-26|8|1.600|
|7b|row|512|512|1|pp|MUL|ffn_norm-27|8|1.467|
|7b|row|512|512|1|pp|MUL|ffn_norm-28|8|1.623|
|7b|row|512|512|1|pp|MUL|ffn_norm-29|8|1.372|
|7b|row|512|512|1|pp|MUL|ffn_norm-3|8|1.438|
|7b|row|512|512|1|pp|MUL|ffn_norm-30|8|1.480|
|7b|row|512|512|1|pp|MUL|ffn_norm-31|8|0.068|
|7b|row|512|512|1|pp|MUL|ffn_norm-4|8|1.387|
|7b|row|512|512|1|pp|MUL|ffn_norm-5|8|1.479|
|7b|row|512|512|1|pp|MUL|ffn_norm-6|8|1.534|
|7b|row|512|512|1|pp|MUL|ffn_norm-7|8|1.454|
|7b|row|512|512|1|pp|MUL|ffn_norm-8|8|1.565|
|7b|row|512|512|1|pp|MUL|ffn_norm-9|8|1.525|
|7b|row|512|512|1|pp|MUL|result_norm|8|0.078|
|7b|row|512|512|1|pp|MUL_MAT|result_output|8|29.192|
|7b|row|512|512|1|pp|ROPE|Kcur-0|7|1.672|
|7b|row|512|512|1|pp|ROPE|Kcur-1|7|1.724|
|7b|row|512|512|1|pp|ROPE|Kcur-10|7|1.606|
|7b|row|512|512|1|pp|ROPE|Kcur-11|7|1.624|
|7b|row|512|512|1|pp|ROPE|Kcur-12|7|1.563|
|7b|row|512|512|1|pp|ROPE|Kcur-13|7|1.626|
|7b|row|512|512|1|pp|ROPE|Kcur-14|7|1.687|
|7b|row|512|512|1|pp|ROPE|Kcur-15|7|1.646|
|7b|row|512|512|1|pp|ROPE|Kcur-16|7|1.641|
|7b|row|512|512|1|pp|ROPE|Kcur-17|7|1.592|
|7b|row|512|512|1|pp|ROPE|Kcur-18|7|1.625|
|7b|row|512|512|1|pp|ROPE|Kcur-19|7|1.571|
|7b|row|512|512|1|pp|ROPE|Kcur-2|7|1.651|
|7b|row|512|512|1|pp|ROPE|Kcur-20|7|1.590|
|7b|row|512|512|1|pp|ROPE|Kcur-21|7|1.695|
|7b|row|512|512|1|pp|ROPE|Kcur-22|7|1.595|
|7b|row|512|512|1|pp|ROPE|Kcur-23|7|1.611|
|7b|row|512|512|1|pp|ROPE|Kcur-24|7|1.697|
|7b|row|512|512|1|pp|ROPE|Kcur-25|7|1.613|
|7b|row|512|512|1|pp|ROPE|Kcur-26|7|1.641|
|7b|row|512|512|1|pp|ROPE|Kcur-27|7|1.616|
|7b|row|512|512|1|pp|ROPE|Kcur-28|7|1.686|
|7b|row|512|512|1|pp|ROPE|Kcur-29|7|1.758|
|7b|row|512|512|1|pp|ROPE|Kcur-3|7|1.635|
|7b|row|512|512|1|pp|ROPE|Kcur-30|7|1.730|
|7b|row|512|512|1|pp|ROPE|Kcur-31|7|1.684|
|7b|row|512|512|1|pp|ROPE|Kcur-4|7|1.523|
|7b|row|512|512|1|pp|ROPE|Kcur-5|7|1.679|
|7b|row|512|512|1|pp|ROPE|Kcur-6|7|1.699|
|7b|row|512|512|1|pp|ROPE|Kcur-7|7|1.630|
|7b|row|512|512|1|pp|ROPE|Kcur-8|7|1.652|
|7b|row|512|512|1|pp|ROPE|Kcur-9|7|1.620|
|7b|row|512|512|1|pp|ROPE|Qcur-0|7|1.645|
|7b|row|512|512|1|pp|ROPE|Qcur-1|7|1.719|
|7b|row|512|512|1|pp|ROPE|Qcur-10|7|1.577|
|7b|row|512|512|1|pp|ROPE|Qcur-11|7|1.646|
|7b|row|512|512|1|pp|ROPE|Qcur-12|7|1.575|
|7b|row|512|512|1|pp|ROPE|Qcur-13|7|1.555|
|7b|row|512|512|1|pp|ROPE|Qcur-14|7|1.630|
|7b|row|512|512|1|pp|ROPE|Qcur-15|7|1.616|
|7b|row|512|512|1|pp|ROPE|Qcur-16|7|1.653|
|7b|row|512|512|1|pp|ROPE|Qcur-17|7|1.549|
|7b|row|512|512|1|pp|ROPE|Qcur-18|7|1.589|
|7b|row|512|512|1|pp|ROPE|Qcur-19|7|1.689|
|7b|row|512|512|1|pp|ROPE|Qcur-2|7|1.680|
|7b|row|512|512|1|pp|ROPE|Qcur-20|7|1.495|
|7b|row|512|512|1|pp|ROPE|Qcur-21|7|1.626|
|7b|row|512|512|1|pp|ROPE|Qcur-22|7|1.651|
|7b|row|512|512|1|pp|ROPE|Qcur-23|7|1.664|
|7b|row|512|512|1|pp|ROPE|Qcur-24|7|1.515|
|7b|row|512|512|1|pp|ROPE|Qcur-25|7|1.613|
|7b|row|512|512|1|pp|ROPE|Qcur-26|7|1.577|
|7b|row|512|512|1|pp|ROPE|Qcur-27|7|1.587|
|7b|row|512|512|1|pp|ROPE|Qcur-28|7|1.641|
|7b|row|512|512|1|pp|ROPE|Qcur-29|7|1.618|
|7b|row|512|512|1|pp|ROPE|Qcur-3|7|1.605|
|7b|row|512|512|1|pp|ROPE|Qcur-30|7|1.622|
|7b|row|512|512|1|pp|ROPE|Qcur-31|7|1.714|
|7b|row|512|512|1|pp|ROPE|Qcur-4|7|1.643|
|7b|row|512|512|1|pp|ROPE|Qcur-5|7|1.616|
|7b|row|512|512|1|pp|ROPE|Qcur-6|7|1.616|
|7b|row|512|512|1|pp|ROPE|Qcur-7|7|1.527|
|7b|row|512|512|1|pp|ROPE|Qcur-8|7|1.636|
|7b|row|512|512|1|pp|ROPE|Qcur-9|7|1.702|
|7b|row|512|512|1|tg|GET_ROWS|embd|32|0.369|
|7b|row|512|512|1|tg|GET_ROWS|node_988|32|0.077|
|7b|row|512|512|1|tg|GET_ROWS|node_989|32|0.057|
|7b|row|512|512|1|tg|MUL|attn_norm-0|32|0.315|
|7b|row|512|512|1|tg|MUL|attn_norm-1|32|0.316|
|7b|row|512|512|1|tg|MUL|attn_norm-10|32|0.323|
|7b|row|512|512|1|tg|MUL|attn_norm-11|32|0.355|
|7b|row|512|512|1|tg|MUL|attn_norm-12|32|0.295|
|7b|row|512|512|1|tg|MUL|attn_norm-13|32|0.318|
|7b|row|512|512|1|tg|MUL|attn_norm-14|32|0.327|
|7b|row|512|512|1|tg|MUL|attn_norm-15|32|0.333|
|7b|row|512|512|1|tg|MUL|attn_norm-16|32|0.326|
|7b|row|512|512|1|tg|MUL|attn_norm-17|32|0.305|
|7b|row|512|512|1|tg|MUL|attn_norm-18|32|0.312|
|7b|row|512|512|1|tg|MUL|attn_norm-19|32|0.340|
|7b|row|512|512|1|tg|MUL|attn_norm-2|32|0.309|
|7b|row|512|512|1|tg|MUL|attn_norm-20|32|0.359|
|7b|row|512|512|1|tg|MUL|attn_norm-21|32|0.309|
|7b|row|512|512|1|tg|MUL|attn_norm-22|32|0.263|
|7b|row|512|512|1|tg|MUL|attn_norm-23|32|0.296|
|7b|row|512|512|1|tg|MUL|attn_norm-24|32|0.297|
|7b|row|512|512|1|tg|MUL|attn_norm-25|32|0.351|
|7b|row|512|512|1|tg|MUL|attn_norm-26|32|0.299|
|7b|row|512|512|1|tg|MUL|attn_norm-27|32|0.312|
|7b|row|512|512|1|tg|MUL|attn_norm-28|32|0.325|
|7b|row|512|512|1|tg|MUL|attn_norm-29|32|0.345|
|7b|row|512|512|1|tg|MUL|attn_norm-3|32|0.283|
|7b|row|512|512|1|tg|MUL|attn_norm-30|32|0.310|
|7b|row|512|512|1|tg|MUL|attn_norm-31|32|0.320|
|7b|row|512|512|1|tg|MUL|attn_norm-4|32|0.327|
|7b|row|512|512|1|tg|MUL|attn_norm-5|32|0.354|
|7b|row|512|512|1|tg|MUL|attn_norm-6|32|0.327|
|7b|row|512|512|1|tg|MUL|attn_norm-7|32|0.309|
|7b|row|512|512|1|tg|MUL|attn_norm-8|32|0.310|
|7b|row|512|512|1|tg|MUL|attn_norm-9|32|0.359|
|7b|row|512|512|1|tg|MUL|ffn_norm-0|32|0.316|
|7b|row|512|512|1|tg|MUL|ffn_norm-1|32|0.265|
|7b|row|512|512|1|tg|MUL|ffn_norm-10|32|0.302|
|7b|row|512|512|1|tg|MUL|ffn_norm-11|32|0.272|
|7b|row|512|512|1|tg|MUL|ffn_norm-12|32|0.283|
|7b|row|512|512|1|tg|MUL|ffn_norm-13|32|0.286|
|7b|row|512|512|1|tg|MUL|ffn_norm-14|32|0.277|
|7b|row|512|512|1|tg|MUL|ffn_norm-15|32|0.316|
|7b|row|512|512|1|tg|MUL|ffn_norm-16|32|0.317|
|7b|row|512|512|1|tg|MUL|ffn_norm-17|32|0.332|
|7b|row|512|512|1|tg|MUL|ffn_norm-18|32|0.320|
|7b|row|512|512|1|tg|MUL|ffn_norm-19|32|0.339|
|7b|row|512|512|1|tg|MUL|ffn_norm-2|32|0.291|
|7b|row|512|512|1|tg|MUL|ffn_norm-20|32|0.302|
|7b|row|512|512|1|tg|MUL|ffn_norm-21|32|0.282|
|7b|row|512|512|1|tg|MUL|ffn_norm-22|32|0.286|
|7b|row|512|512|1|tg|MUL|ffn_norm-23|32|0.289|
|7b|row|512|512|1|tg|MUL|ffn_norm-24|32|0.348|
|7b|row|512|512|1|tg|MUL|ffn_norm-25|32|0.299|
|7b|row|512|512|1|tg|MUL|ffn_norm-26|32|0.311|
|7b|row|512|512|1|tg|MUL|ffn_norm-27|32|0.303|
|7b|row|512|512|1|tg|MUL|ffn_norm-28|32|0.343|
|7b|row|512|512|1|tg|MUL|ffn_norm-29|32|0.324|
|7b|row|512|512|1|tg|MUL|ffn_norm-3|32|0.319|
|7b|row|512|512|1|tg|MUL|ffn_norm-30|32|0.294|
|7b|row|512|512|1|tg|MUL|ffn_norm-31|32|0.319|
|7b|row|512|512|1|tg|MUL|ffn_norm-4|32|0.335|
|7b|row|512|512|1|tg|MUL|ffn_norm-5|32|0.278|
|7b|row|512|512|1|tg|MUL|ffn_norm-6|32|0.295|
|7b|row|512|512|1|tg|MUL|ffn_norm-7|32|0.331|
|7b|row|512|512|1|tg|MUL|ffn_norm-8|32|0.279|
|7b|row|512|512|1|tg|MUL|ffn_norm-9|32|0.275|
|7b|row|512|512|1|tg|MUL|result_norm|32|0.269|
|7b|row|512|512|1|tg|MUL_MAT|result_output|32|114.311|
|tiny|layer|128|128|0|pp|GET_ROWS|embd|29|1.674|
|tiny|layer|128|128|0|pp|GET_ROWS|node_678|29|0.088|
|tiny|layer|128|128|0|pp|GET_ROWS|node_679|29|0.048|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-0|29|5.655|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-1|29|4.456|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-10|29|4.191|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-11|29|4.223|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-12|29|4.235|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-13|29|4.129|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-14|29|4.130|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-15|29|4.140|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-16|29|4.250|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-17|29|4.118|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-18|29|4.242|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-19|29|4.176|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-2|29|4.344|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-20|29|4.176|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-21|29|0.373|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-3|29|4.128|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-4|29|4.047|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-5|29|4.106|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-6|29|4.328|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-7|29|4.215|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-8|29|4.204|
|tiny|layer|128|128|0|pp|GLU|ffn_swiglu-9|29|4.342|
|tiny|layer|128|128|0|pp|MUL_MAT|result_output|29|55.025|
|tiny|layer|128|128|0|tg|GET_ROWS|embd|32|0.318|
|tiny|layer|128|128|0|tg|GET_ROWS|node_678|32|0.062|
|tiny|layer|128|128|0|tg|GET_ROWS|node_679|32|0.046|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-0|32|0.421|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-1|32|0.395|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-10|32|0.438|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-11|32|0.425|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-12|32|0.420|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-13|32|0.429|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-14|32|0.433|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-15|32|0.470|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-16|32|0.492|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-17|32|0.409|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-18|32|0.421|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-19|32|0.399|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-2|32|0.375|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-20|32|0.406|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-21|32|0.434|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-3|32|0.402|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-4|32|0.413|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-5|32|0.378|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-6|32|0.427|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-7|32|0.436|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-8|32|0.383|
|tiny|layer|128|128|0|tg|GLU|ffn_swiglu-9|32|0.373|
|tiny|layer|128|128|0|tg|MUL_MAT|result_output|32|57.791|
|tiny|layer|128|128|1|pp|GET_ROWS|embd|29|1.631|
|tiny|layer|128|128|1|pp|GET_ROWS|node_678|29|0.068|
|tiny|layer|128|128|1|pp|GET_ROWS|node_679|29|0.045|
|tiny|layer|128|128|1|pp|MUL_MAT|result_output|29|53.448|
|tiny|layer|128|128|1|tg|GET_ROWS|embd|32|0.271|
|tiny|layer|128|128|1|tg|GET_ROWS|node_678|32|0.058|
|tiny|layer|128|128|1|tg|GET_ROWS|node_679|32|0.045|
|tiny|layer|128|128|1|tg|MUL_MAT|result_output|32|56.814|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|107.522|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|107.645|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|107.493|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|107.828|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|108.715|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|113.256|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|111.845|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|107.819|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|107.680|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|109.382|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|109.527|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|107.977|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|108.541|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|108.245|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|108.459|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|111.009|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|108.474|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|108.324|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|107.793|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|111.825|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|108.879|
|tiny|layer|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|107.851|
|tiny|layer|512|512|0|pp|GET_ROWS|embd|8|1.354|
|tiny|layer|512|512|0|pp|GET_ROWS|node_678|8|0.020|
|tiny|layer|512|512|0|pp|GET_ROWS|node_679|8|0.014|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-0|8|9.775|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-1|8|9.731|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-10|8|9.779|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-11|8|9.651|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-12|8|9.715|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-13|8|9.750|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-14|8|9.731|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-15|8|9.788|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-16|8|9.634|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-17|8|9.802|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-18|8|9.732|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-19|8|9.787|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-2|8|9.634|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-20|8|9.651|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-21|8|0.091|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-3|8|9.650|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-4|8|9.730|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-5|8|9.659|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-6|8|9.608|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-7|8|9.737|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-8|8|9.708|
|tiny|layer|512|512|0|pp|GLU|ffn_swiglu-9|8|9.669|
|tiny|layer|512|512|0|pp|MUL_MAT|result_output|8|15.289|
|tiny|layer|512|512|0|pp|ROPE|Kcur-0|7|0.444|
|tiny|layer|512|512|0|pp|ROPE|Kcur-1|7|0.541|
|tiny|layer|512|512|0|pp|ROPE|Kcur-10|7|0.460|
|tiny|layer|512|512|0|pp|ROPE|Kcur-11|7|0.473|
|tiny|layer|512|512|0|pp|ROPE|Kcur-12|7|0.487|
|tiny|layer|512|512|0|pp|ROPE|Kcur-13|7|0.459|
|tiny|layer|512|512|0|pp|ROPE|Kcur-14|7|0.523|
|tiny|layer|512|512|0|pp|ROPE|Kcur-15|7|0.483|
|tiny|layer|512|512|0|pp|ROPE|Kcur-16|7|0.492|
|tiny|layer|512|512|0|pp|ROPE|Kcur-17|7|0.431|
|tiny|layer|512|512|0|pp|ROPE|Kcur-18|7|0.556|
|tiny|layer|512|512|0|pp|ROPE|Kcur-19|7|0.506|
|tiny|layer|512|512|0|pp|ROPE|Kcur-2|7|0.320|
|tiny|layer|512|512|0|pp|ROPE|Kcur-20|7|0.539|
|tiny|layer|512|512|0|pp|ROPE|Kcur-21|7|0.520|
|tiny|layer|512|512|0|pp|ROPE|Kcur-3|7|0.479|
|tiny|layer|512|512|0|pp|ROPE|Kcur-4|7|0.532|
|tiny|layer|512|512|0|pp|ROPE|Kcur-5|7|0.513|
|tiny|layer|512|512|0|pp|ROPE|Kcur-6|7|0.565|
|tiny|layer|512|512|0|pp|ROPE|Kcur-7|7|0.494|
|tiny|layer|512|512|0|pp|ROPE|Kcur-8|7|0.591|
|tiny|layer|512|512|0|pp|ROPE|Kcur-9|7|0.406|
|tiny|layer|512|512|0|pp|ROPE|Qcur-0|7|0.896|
|tiny|layer|512|512|0|pp|ROPE|Qcur-1|7|0.921|
|tiny|layer|512|512|0|pp|ROPE|Qcur-10|7|0.875|
|tiny|layer|512|512|0|pp|ROPE|Qcur-11|7|1.028|
|tiny|layer|512|512|0|pp|ROPE|Qcur-12|7|0.976|
|tiny|layer|512|512|0|pp|ROPE|Qcur-13|7|0.990|
|tiny|layer|512|512|0|pp|ROPE|Qcur-14|7|0.893|
|tiny|layer|512|512|0|pp|ROPE|Qcur-15|7|0.907|
|tiny|layer|512|512|0|pp|ROPE|Qcur-16|7|0.951|
|tiny|layer|512|512|0|pp|ROPE|Qcur-17|7|0.886|
|tiny|layer|512|512|0|pp|ROPE|Qcur-18|7|0.907|
|tiny|layer|512|512|0|pp|ROPE|Qcur-19|7|1.025|
|tiny|layer|512|512|0|pp|ROPE|Qcur-2|7|0.870|
|tiny|layer|512|512|0|pp|ROPE|Qcur-20|7|1.076|
|tiny|layer|512|512|0|pp|ROPE|Qcur-21|7|1.004|
|tiny|layer|512|512|0|pp|ROPE|Qcur-3|7|0.906|
|tiny|layer|512|512|0|pp|ROPE|Qcur-4|7|0.936|
|tiny|layer|512|512|0|pp|ROPE|Qcur-5|7|1.027|
|tiny|layer|512|512|0|pp|ROPE|Qcur-6|7|0.844|
|tiny|layer|512|512|0|pp|ROPE|Qcur-7|7|0.907|
|tiny|layer|512|512|0|pp|ROPE|Qcur-8|7|0.967|
|tiny|layer|512|512|0|pp|ROPE|Qcur-9|7|0.985|
|tiny|layer|512|512|0|tg|GET_ROWS|embd|32|0.387|
|tiny|layer|512|512|0|tg|GET_ROWS|node_678|32|0.062|
|tiny|layer|512|512|0|tg|GET_ROWS|node_679|32|0.048|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-0|32|0.355|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-1|32|0.362|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-10|32|0.374|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-11|32|0.388|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-12|32|0.376|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-13|32|0.337|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-14|32|0.330|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-15|32|0.342|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-16|32|0.364|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-17|32|0.349|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-18|32|0.351|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-19|32|0.363|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-2|32|0.370|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-20|32|0.374|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-21|32|0.364|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-3|32|0.365|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-4|32|0.401|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-5|32|0.367|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-6|32|0.336|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-7|32|0.379|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-8|32|0.327|
|tiny|layer|512|512|0|tg|GLU|ffn_swiglu-9|32|0.326|
|tiny|layer|512|512|0|tg|MUL_MAT|result_output|32|57.483|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|108.076|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|108.408|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|112.085|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|111.960|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|112.634|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|111.118|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|107.624|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|110.050|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|109.825|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|107.272|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|107.117|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|107.163|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|107.368|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|108.507|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|108.393|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|112.176|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|114.372|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|110.328|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|108.580|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|108.811|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|108.792|
|tiny|layer|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|108.571|
|tiny|layer|512|512|1|pp|GET_ROWS|embd|8|1.286|
|tiny|layer|512|512|1|pp|GET_ROWS|node_678|8|0.021|
|tiny|layer|512|512|1|pp|GET_ROWS|node_679|8|0.014|
|tiny|layer|512|512|1|pp|MUL_MAT|result_output|8|15.184|
|tiny|layer|512|512|1|pp|ROPE|Kcur-0|7|0.506|
|tiny|layer|512|512|1|pp|ROPE|Kcur-1|7|0.446|
|tiny|layer|512|512|1|pp|ROPE|Kcur-10|7|0.427|
|tiny|layer|512|512|1|pp|ROPE|Kcur-11|7|0.487|
|tiny|layer|512|512|1|pp|ROPE|Kcur-12|7|0.531|
|tiny|layer|512|512|1|pp|ROPE|Kcur-13|7|0.474|
|tiny|layer|512|512|1|pp|ROPE|Kcur-14|7|0.458|
|tiny|layer|512|512|1|pp|ROPE|Kcur-15|7|0.517|
|tiny|layer|512|512|1|pp|ROPE|Kcur-16|7|0.452|
|tiny|layer|512|512|1|pp|ROPE|Kcur-17|7|0.583|
|tiny|layer|512|512|1|pp|ROPE|Kcur-18|7|0.562|
|tiny|layer|512|512|1|pp|ROPE|Kcur-19|7|0.469|
|tiny|layer|512|512|1|pp|ROPE|Kcur-2|7|0.487|
|tiny|layer|512|512|1|pp|ROPE|Kcur-20|7|0.493|
|tiny|layer|512|512|1|pp|ROPE|Kcur-21|7|0.450|
|tiny|layer|512|512|1|pp|ROPE|Kcur-3|7|0.433|
|tiny|layer|512|512|1|pp|ROPE|Kcur-4|7|0.504|
|tiny|layer|512|512|1|pp|ROPE|Kcur-5|7|0.436|
|tiny|layer|512|512|1|pp|ROPE|Kcur-6|7|0.540|
|tiny|layer|512|512|1|pp|ROPE|Kcur-7|7|0.608|
|tiny|layer|512|512|1|pp|ROPE|Kcur-8|7|0.510|
|tiny|layer|512|512|1|pp|ROPE|Kcur-9|7|0.451|
|tiny|layer|512|512|1|pp|ROPE|Qcur-0|7|0.919|
|tiny|layer|512|512|1|pp|ROPE|Qcur-1|7|0.883|
|tiny|layer|512|512|1|pp|ROPE|Qcur-10|7|0.917|
|tiny|layer|512|512|1|pp|ROPE|Qcur-11|7|0.905|
|tiny|layer|512|512|1|pp|ROPE|Qcur-12|7|0.991|
|tiny|layer|512|512|1|pp|ROPE|Qcur-13|7|0.882|
|tiny|layer|512|512|1|pp|ROPE|Qcur-14|7|0.918|
|tiny|layer|512|512|1|pp|ROPE|Qcur-15|7|0.941|
|tiny|layer|512|512|1|pp|ROPE|Qcur-16|7|0.893|
|tiny|layer|512|512|1|pp|ROPE|Qcur-17|7|0.948|
|tiny|layer|512|512|1|pp|ROPE|Qcur-18|7|0.961|
|tiny|layer|512|512|1|pp|ROPE|Qcur-19|7|0.981|
|tiny|layer|512|512|1|pp|ROPE|Qcur-2|7|0.970|
|tiny|layer|512|512|1|pp|ROPE|Qcur-20|7|0.945|
|tiny|layer|512|512|1|pp|ROPE|Qcur-21|7|0.929|
|tiny|layer|512|512|1|pp|ROPE|Qcur-3|7|0.874|
|tiny|layer|512|512|1|pp|ROPE|Qcur-4|7|0.974|
|tiny|layer|512|512|1|pp|ROPE|Qcur-5|7|1.012|
|tiny|layer|512|512|1|pp|ROPE|Qcur-6|7|0.959|
|tiny|layer|512|512|1|pp|ROPE|Qcur-7|7|1.010|
|tiny|layer|512|512|1|pp|ROPE|Qcur-8|7|0.905|
|tiny|layer|512|512|1|pp|ROPE|Qcur-9|7|0.961|
|tiny|layer|512|512|1|tg|GET_ROWS|embd|32|0.280|
|tiny|layer|512|512|1|tg|GET_ROWS|node_678|32|0.060|
|tiny|layer|512|512|1|tg|GET_ROWS|node_679|32|0.046|
|tiny|layer|512|512|1|tg|MUL_MAT|result_output|32|56.823|
|tiny|pd0|128|128|0|pp|GET_ROWS|embd|29|1.662|
|tiny|pd0|128|128|0|pp|GET_ROWS|node_678|29|0.077|
|tiny|pd0|128|128|0|pp|GET_ROWS|node_679|29|0.047|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-0|29|5.662|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-1|29|4.039|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-10|29|3.952|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-11|29|4.076|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-12|29|3.825|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-13|29|4.036|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-14|29|3.999|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-15|29|4.063|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-16|29|4.285|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-17|29|4.003|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-18|29|4.218|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-19|29|4.007|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-2|29|4.013|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-20|29|4.106|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-21|29|0.414|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-3|29|4.437|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-4|29|4.316|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-5|29|4.156|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-6|29|4.275|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-7|29|4.076|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-8|29|4.251|
|tiny|pd0|128|128|0|pp|GLU|ffn_swiglu-9|29|3.926|
|tiny|pd0|128|128|0|pp|MUL_MAT|result_output|29|55.542|
|tiny|pd0|128|128|0|tg|GET_ROWS|embd|32|0.475|
|tiny|pd0|128|128|0|tg|GET_ROWS|node_678|32|0.060|
|tiny|pd0|128|128|0|tg|GET_ROWS|node_679|32|0.046|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-0|32|0.564|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-1|32|0.614|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-10|32|0.593|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-11|32|0.520|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-12|32|0.522|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-13|32|0.494|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-14|32|0.530|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-15|32|0.487|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-16|32|0.629|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-17|32|0.526|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-18|32|0.522|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-19|32|0.453|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-2|32|0.641|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-20|32|0.564|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-21|32|0.592|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-3|32|0.556|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-4|32|0.457|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-5|32|0.587|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-6|32|0.535|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-7|32|0.625|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-8|32|0.472|
|tiny|pd0|128|128|0|tg|GLU|ffn_swiglu-9|32|0.505|
|tiny|pd0|128|128|0|tg|MUL_MAT|result_output|32|57.849|
|tiny|pd0|128|128|1|pp|GET_ROWS|embd|29|1.617|
|tiny|pd0|128|128|1|pp|GET_ROWS|node_678|29|0.067|
|tiny|pd0|128|128|1|pp|GET_ROWS|node_679|29|0.044|
|tiny|pd0|128|128|1|pp|MUL_MAT|result_output|29|53.995|
|tiny|pd0|128|128|1|tg|GET_ROWS|embd|32|0.314|
|tiny|pd0|128|128|1|tg|GET_ROWS|node_678|32|0.058|
|tiny|pd0|128|128|1|tg|GET_ROWS|node_679|32|0.044|
|tiny|pd0|128|128|1|tg|MUL_MAT|result_output|32|57.622|
|tiny|pd0|512|128|0|pp|GET_ROWS|embd|29|1.388|
|tiny|pd0|512|128|0|pp|GET_ROWS|node_678|8|0.021|
|tiny|pd0|512|128|0|pp|GET_ROWS|node_679|8|0.014|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-0|29|5.984|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-1|29|4.351|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-10|29|4.096|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-11|29|3.985|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-12|29|4.484|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-13|29|4.355|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-14|29|4.318|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-15|29|4.153|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-16|29|4.170|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-17|29|4.194|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-18|29|4.145|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-19|29|3.880|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-2|29|4.222|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-20|29|4.086|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-21|8|0.091|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-3|29|4.215|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-4|29|4.226|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-5|29|4.128|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-6|29|4.250|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-7|29|4.296|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-8|29|4.367|
|tiny|pd0|512|128|0|pp|GLU|ffn_swiglu-9|29|4.122|
|tiny|pd0|512|128|0|pp|MUL_MAT|result_output|8|15.251|
|tiny|pd0|512|128|0|tg|GET_ROWS|embd|32|0.312|
|tiny|pd0|512|128|0|tg|GET_ROWS|node_678|32|0.063|
|tiny|pd0|512|128|0|tg|GET_ROWS|node_679|32|0.052|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-0|32|0.439|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-1|32|0.334|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-10|32|0.394|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-11|32|0.416|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-12|32|0.380|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-13|32|0.407|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-14|32|0.395|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-15|32|0.396|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-16|32|0.374|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-17|32|0.361|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-18|32|0.361|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-19|32|0.390|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-2|32|0.440|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-20|32|0.386|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-21|32|0.386|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-3|32|0.352|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-4|32|0.432|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-5|32|0.386|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-6|32|0.394|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-7|32|0.379|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-8|32|0.358|
|tiny|pd0|512|128|0|tg|GLU|ffn_swiglu-9|32|0.373|
|tiny|pd0|512|128|0|tg|MUL_MAT|result_output|32|57.768|
|tiny|pd0|512|128|1|pp|GET_ROWS|embd|29|1.370|
|tiny|pd0|512|128|1|pp|GET_ROWS|node_678|8|0.017|
|tiny|pd0|512|128|1|pp|GET_ROWS|node_679|8|0.012|
|tiny|pd0|512|128|1|pp|MUL_MAT|result_output|8|14.700|
|tiny|pd0|512|128|1|tg|GET_ROWS|embd|32|0.302|
|tiny|pd0|512|128|1|tg|GET_ROWS|node_678|32|0.058|
|tiny|pd0|512|128|1|tg|GET_ROWS|node_679|32|0.049|
|tiny|pd0|512|128|1|tg|MUL_MAT|result_output|32|57.219|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|106.791|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|107.308|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|104.397|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|104.302|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|107.823|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|104.555|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|105.026|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|105.447|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|105.550|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|105.146|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|109.420|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|105.090|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|104.706|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|107.349|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|107.515|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|105.954|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|107.479|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|105.066|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|104.468|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|104.757|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|104.369|
|tiny|pd0|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|111.815|
|tiny|pd0|512|512|0|pp|GET_ROWS|embd|8|1.312|
|tiny|pd0|512|512|0|pp|GET_ROWS|node_678|8|0.021|
|tiny|pd0|512|512|0|pp|GET_ROWS|node_679|8|0.014|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-0|8|9.692|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-1|8|9.786|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-10|8|9.598|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-11|8|9.700|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-12|8|9.694|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-13|8|9.723|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-14|8|9.807|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-15|8|9.629|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-16|8|9.672|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-17|8|9.636|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-18|8|9.693|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-19|8|9.673|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-2|8|9.681|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-20|8|9.631|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-21|8|0.077|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-3|8|9.755|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-4|8|9.692|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-5|8|9.774|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-6|8|9.717|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-7|8|9.708|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-8|8|9.638|
|tiny|pd0|512|512|0|pp|GLU|ffn_swiglu-9|8|9.599|
|tiny|pd0|512|512|0|pp|MUL_MAT|result_output|8|15.229|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-0|7|0.613|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-1|7|0.538|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-10|7|0.516|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-11|7|0.451|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-12|7|0.466|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-13|7|0.453|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-14|7|0.450|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-15|7|0.475|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-16|7|0.495|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-17|7|0.491|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-18|7|0.495|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-19|7|0.469|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-2|7|0.422|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-20|7|0.445|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-21|7|0.467|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-3|7|0.513|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-4|7|0.438|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-5|7|0.544|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-6|7|0.525|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-7|7|0.550|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-8|7|0.507|
|tiny|pd0|512|512|0|pp|ROPE|Kcur-9|7|0.506|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-0|7|1.018|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-1|7|0.874|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-10|7|0.897|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-11|7|0.827|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-12|7|1.064|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-13|7|0.851|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-14|7|0.924|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-15|7|0.853|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-16|7|0.938|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-17|7|0.933|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-18|7|0.983|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-19|7|0.936|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-2|7|0.991|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-20|7|0.839|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-21|7|0.904|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-3|7|1.032|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-4|7|0.982|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-5|7|0.977|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-6|7|0.961|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-7|7|0.951|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-8|7|0.911|
|tiny|pd0|512|512|0|pp|ROPE|Qcur-9|7|0.940|
|tiny|pd0|512|512|0|tg|GET_ROWS|embd|32|0.278|
|tiny|pd0|512|512|0|tg|GET_ROWS|node_678|32|0.061|
|tiny|pd0|512|512|0|tg|GET_ROWS|node_679|32|0.048|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-0|32|0.334|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-1|32|0.339|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-10|32|0.328|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-11|32|0.345|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-12|32|0.353|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-13|32|0.323|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-14|32|0.357|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-15|32|0.316|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-16|32|0.369|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-17|32|0.312|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-18|32|0.323|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-19|32|0.328|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-2|32|0.339|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-20|32|0.349|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-21|32|0.340|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-3|32|0.338|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-4|32|0.302|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-5|32|0.338|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-6|32|0.389|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-7|32|0.305|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-8|32|0.323|
|tiny|pd0|512|512|0|tg|GLU|ffn_swiglu-9|32|0.364|
|tiny|pd0|512|512|0|tg|MUL_MAT|result_output|32|57.821|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|108.634|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|108.674|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|108.315|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|105.922|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|111.466|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|104.670|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|107.444|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|106.811|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|104.693|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|104.755|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|105.850|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|105.996|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|105.765|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|109.033|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|108.940|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|110.926|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|105.754|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|106.127|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|110.499|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|109.195|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|104.773|
|tiny|pd0|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|111.084|
|tiny|pd0|512|512|1|pp|GET_ROWS|embd|8|1.372|
|tiny|pd0|512|512|1|pp|GET_ROWS|node_678|8|0.019|
|tiny|pd0|512|512|1|pp|GET_ROWS|node_679|8|0.013|
|tiny|pd0|512|512|1|pp|MUL_MAT|result_output|8|15.349|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-0|7|0.572|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-1|7|0.437|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-10|7|0.478|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-11|7|0.527|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-12|7|0.535|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-13|7|0.517|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-14|7|0.484|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-15|7|0.509|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-16|7|0.500|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-17|7|0.502|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-18|7|0.551|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-19|7|0.586|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-2|7|0.482|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-20|7|0.478|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-21|7|0.514|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-3|7|0.404|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-4|7|0.553|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-5|7|0.425|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-6|7|0.514|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-7|7|0.435|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-8|7|0.488|
|tiny|pd0|512|512|1|pp|ROPE|Kcur-9|7|0.470|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-0|7|0.866|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-1|7|0.942|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-10|7|0.963|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-11|7|0.941|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-12|7|0.938|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-13|7|0.993|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-14|7|0.895|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-15|7|0.989|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-16|7|0.926|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-17|7|0.999|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-18|7|0.944|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-19|7|0.950|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-2|7|1.009|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-20|7|0.993|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-21|7|0.934|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-3|7|1.001|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-4|7|0.904|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-5|7|1.079|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-6|7|1.053|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-7|7|1.001|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-8|7|1.030|
|tiny|pd0|512|512|1|pp|ROPE|Qcur-9|7|0.983|
|tiny|pd0|512|512|1|tg|GET_ROWS|embd|32|0.304|
|tiny|pd0|512|512|1|tg|GET_ROWS|node_678|32|0.063|
|tiny|pd0|512|512|1|tg|GET_ROWS|node_679|32|0.046|
|tiny|pd0|512|512|1|tg|MUL_MAT|result_output|32|57.521|
|tiny|pd1|128|128|0|pp|GET_ROWS|embd|29|1.526|
|tiny|pd1|128|128|0|pp|GET_ROWS|node_678|29|0.077|
|tiny|pd1|128|128|0|pp|GET_ROWS|node_679|29|0.048|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-0|29|5.553|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-1|29|4.228|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-10|29|4.006|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-11|29|3.973|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-12|29|4.046|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-13|29|4.122|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-14|29|4.215|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-15|29|4.161|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-16|29|4.309|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-17|29|3.882|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-18|29|4.293|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-19|29|3.972|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-2|29|3.982|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-20|29|3.942|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-21|29|0.296|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-3|29|3.922|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-4|29|4.008|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-5|29|4.034|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-6|29|3.944|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-7|29|4.170|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-8|29|4.044|
|tiny|pd1|128|128|0|pp|GLU|ffn_swiglu-9|29|4.132|
|tiny|pd1|128|128|0|pp|MUL_MAT|result_output|29|54.925|
|tiny|pd1|128|128|0|tg|GET_ROWS|embd|32|0.268|
|tiny|pd1|128|128|0|tg|GET_ROWS|node_678|32|0.062|
|tiny|pd1|128|128|0|tg|GET_ROWS|node_679|32|0.048|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-0|32|0.359|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-1|32|0.326|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-10|32|0.335|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-11|32|0.340|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-12|32|0.368|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-13|32|0.376|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-14|32|0.368|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-15|32|0.364|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-16|32|0.314|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-17|32|0.438|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-18|32|0.330|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-19|32|0.332|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-2|32|0.327|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-20|32|0.338|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-21|32|0.346|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-3|32|0.341|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-4|32|0.325|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-5|32|0.342|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-6|32|0.344|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-7|32|0.325|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-8|32|0.355|
|tiny|pd1|128|128|0|tg|GLU|ffn_swiglu-9|32|0.357|
|tiny|pd1|128|128|0|tg|MUL_MAT|result_output|32|57.311|
|tiny|pd1|128|128|1|pp|GET_ROWS|embd|29|1.605|
|tiny|pd1|128|128|1|pp|GET_ROWS|node_678|29|0.069|
|tiny|pd1|128|128|1|pp|GET_ROWS|node_679|29|0.044|
|tiny|pd1|128|128|1|pp|MUL_MAT|result_output|29|53.395|
|tiny|pd1|128|128|1|tg|GET_ROWS|embd|32|0.300|
|tiny|pd1|128|128|1|tg|GET_ROWS|node_678|32|0.060|
|tiny|pd1|128|128|1|tg|GET_ROWS|node_679|32|0.046|
|tiny|pd1|128|128|1|tg|MUL_MAT|result_output|32|57.542|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|106.801|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|106.793|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|106.109|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|106.298|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|108.964|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|105.357|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|105.271|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|107.992|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|105.225|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|105.631|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|105.257|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|105.656|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|105.541|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|104.780|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|105.626|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|108.023|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|105.537|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|106.368|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|112.723|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|111.835|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|111.684|
|tiny|pd1|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|107.406|
|tiny|pd1|512|512|0|pp|GET_ROWS|embd|8|1.356|
|tiny|pd1|512|512|0|pp|GET_ROWS|node_678|8|0.021|
|tiny|pd1|512|512|0|pp|GET_ROWS|node_679|8|0.014|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-0|8|9.792|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-1|8|9.621|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-10|8|9.768|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-11|8|9.680|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-12|8|9.701|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-13|8|9.634|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-14|8|9.627|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-15|8|9.700|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-16|8|9.687|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-17|8|9.630|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-18|8|9.744|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-19|8|9.745|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-2|8|9.728|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-20|8|9.736|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-21|8|0.108|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-3|8|9.797|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-4|8|9.659|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-5|8|9.648|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-6|8|9.794|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-7|8|9.716|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-8|8|9.824|
|tiny|pd1|512|512|0|pp|GLU|ffn_swiglu-9|8|9.779|
|tiny|pd1|512|512|0|pp|MUL_MAT|result_output|8|15.241|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-0|7|0.499|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-1|7|0.517|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-10|7|0.431|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-11|7|0.457|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-12|7|0.460|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-13|7|0.403|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-14|7|0.487|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-15|7|0.422|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-16|7|0.561|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-17|7|0.374|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-18|7|0.429|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-19|7|0.544|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-2|7|0.507|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-20|7|0.458|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-21|7|0.578|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-3|7|0.566|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-4|7|0.533|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-5|7|0.543|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-6|7|0.521|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-7|7|0.511|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-8|7|0.566|
|tiny|pd1|512|512|0|pp|ROPE|Kcur-9|7|0.470|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-0|7|0.976|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-1|7|0.984|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-10|7|0.938|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-11|7|0.973|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-12|7|0.894|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-13|7|0.970|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-14|7|0.924|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-15|7|0.947|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-16|7|0.837|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-17|7|0.902|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-18|7|0.946|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-19|7|0.961|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-2|7|1.084|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-20|7|1.012|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-21|7|0.890|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-3|7|1.016|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-4|7|0.985|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-5|7|1.032|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-6|7|0.973|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-7|7|0.848|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-8|7|0.911|
|tiny|pd1|512|512|0|pp|ROPE|Qcur-9|7|0.961|
|tiny|pd1|512|512|0|tg|GET_ROWS|embd|32|0.289|
|tiny|pd1|512|512|0|tg|GET_ROWS|node_678|32|0.066|
|tiny|pd1|512|512|0|tg|GET_ROWS|node_679|32|0.048|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-0|32|0.389|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-1|32|0.378|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-10|32|0.398|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-11|32|0.361|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-12|32|0.336|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-13|32|0.352|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-14|32|0.380|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-15|32|0.342|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-16|32|0.437|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-17|32|0.348|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-18|32|0.372|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-19|32|0.346|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-2|32|0.341|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-20|32|0.333|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-21|32|0.362|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-3|32|0.368|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-4|32|0.361|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-5|32|0.368|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-6|32|0.383|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-7|32|0.397|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-8|32|0.333|
|tiny|pd1|512|512|0|tg|GLU|ffn_swiglu-9|32|0.392|
|tiny|pd1|512|512|0|tg|MUL_MAT|result_output|32|57.791|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|105.898|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|108.932|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|106.346|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|106.085|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|108.802|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|106.081|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|106.151|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|110.996|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|109.605|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|106.034|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|104.890|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|105.268|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|105.865|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|105.918|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|112.848|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|105.811|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|105.734|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|106.175|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|106.730|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|107.358|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|105.544|
|tiny|pd1|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|105.458|
|tiny|pd1|512|512|1|pp|GET_ROWS|embd|8|1.291|
|tiny|pd1|512|512|1|pp|GET_ROWS|node_678|8|0.020|
|tiny|pd1|512|512|1|pp|GET_ROWS|node_679|8|0.013|
|tiny|pd1|512|512|1|pp|MUL_MAT|result_output|8|15.047|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-0|7|0.436|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-1|7|0.591|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-10|7|0.510|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-11|7|0.512|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-12|7|0.530|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-13|7|0.473|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-14|7|0.441|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-15|7|0.456|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-16|7|0.474|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-17|7|0.414|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-18|7|0.614|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-19|7|0.501|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-2|7|0.399|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-20|7|0.425|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-21|7|0.446|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-3|7|0.537|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-4|7|0.428|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-5|7|0.570|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-6|7|0.361|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-7|7|0.444|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-8|7|0.588|
|tiny|pd1|512|512|1|pp|ROPE|Kcur-9|7|0.474|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-0|7|1.006|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-1|7|0.991|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-10|7|0.980|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-11|7|0.984|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-12|7|0.913|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-13|7|0.961|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-14|7|1.015|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-15|7|1.060|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-16|7|1.027|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-17|7|0.967|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-18|7|0.905|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-19|7|1.052|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-2|7|0.942|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-20|7|1.065|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-21|7|0.934|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-3|7|0.939|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-4|7|1.033|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-5|7|1.129|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-6|7|0.987|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-7|7|1.027|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-8|7|0.991|
|tiny|pd1|512|512|1|pp|ROPE|Qcur-9|7|0.940|
|tiny|pd1|512|512|1|tg|GET_ROWS|embd|32|0.266|
|tiny|pd1|512|512|1|tg|GET_ROWS|node_678|32|0.059|
|tiny|pd1|512|512|1|tg|GET_ROWS|node_679|32|0.047|
|tiny|pd1|512|512|1|tg|MUL_MAT|result_output|32|57.552|
|tiny|row|128|128|0|pp|GET_ROWS|embd|29|1.406|
|tiny|row|128|128|0|pp|GET_ROWS|node_678|29|0.070|
|tiny|row|128|128|0|pp|GET_ROWS|node_679|29|0.043|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-0|29|6.564|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-1|29|5.448|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-10|29|5.023|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-11|29|5.015|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-12|29|5.576|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-13|29|5.125|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-14|29|5.113|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-15|29|5.120|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-16|29|5.081|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-17|29|5.190|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-18|29|5.184|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-19|29|5.143|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-2|29|5.163|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-20|29|5.112|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-21|29|0.529|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-3|29|5.054|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-4|29|5.229|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-5|29|5.013|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-6|29|5.048|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-7|29|5.136|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-8|29|5.088|
|tiny|row|128|128|0|pp|GLU|ffn_swiglu-9|29|4.990|
|tiny|row|128|128|0|pp|MUL|attn_norm-0|29|1.234|
|tiny|row|128|128|0|pp|MUL|attn_norm-1|29|1.116|
|tiny|row|128|128|0|pp|MUL|attn_norm-10|29|1.115|
|tiny|row|128|128|0|pp|MUL|attn_norm-11|29|1.115|
|tiny|row|128|128|0|pp|MUL|attn_norm-12|29|1.423|
|tiny|row|128|128|0|pp|MUL|attn_norm-13|29|1.215|
|tiny|row|128|128|0|pp|MUL|attn_norm-14|29|1.036|
|tiny|row|128|128|0|pp|MUL|attn_norm-15|29|1.153|
|tiny|row|128|128|0|pp|MUL|attn_norm-16|29|1.150|
|tiny|row|128|128|0|pp|MUL|attn_norm-17|29|1.219|
|tiny|row|128|128|0|pp|MUL|attn_norm-18|29|1.093|
|tiny|row|128|128|0|pp|MUL|attn_norm-19|29|1.126|
|tiny|row|128|128|0|pp|MUL|attn_norm-2|29|1.313|
|tiny|row|128|128|0|pp|MUL|attn_norm-20|29|1.455|
|tiny|row|128|128|0|pp|MUL|attn_norm-21|29|1.239|
|tiny|row|128|128|0|pp|MUL|attn_norm-3|29|1.175|
|tiny|row|128|128|0|pp|MUL|attn_norm-4|29|1.209|
|tiny|row|128|128|0|pp|MUL|attn_norm-5|29|1.091|
|tiny|row|128|128|0|pp|MUL|attn_norm-6|29|1.189|
|tiny|row|128|128|0|pp|MUL|attn_norm-7|29|1.056|
|tiny|row|128|128|0|pp|MUL|attn_norm-8|29|1.272|
|tiny|row|128|128|0|pp|MUL|attn_norm-9|29|1.313|
|tiny|row|128|128|0|pp|MUL|ffn_norm-0|29|1.051|
|tiny|row|128|128|0|pp|MUL|ffn_norm-1|29|0.962|
|tiny|row|128|128|0|pp|MUL|ffn_norm-10|29|1.070|
|tiny|row|128|128|0|pp|MUL|ffn_norm-11|29|1.206|
|tiny|row|128|128|0|pp|MUL|ffn_norm-12|29|1.187|
|tiny|row|128|128|0|pp|MUL|ffn_norm-13|29|1.070|
|tiny|row|128|128|0|pp|MUL|ffn_norm-14|29|1.165|
|tiny|row|128|128|0|pp|MUL|ffn_norm-15|29|1.219|
|tiny|row|128|128|0|pp|MUL|ffn_norm-16|29|1.156|
|tiny|row|128|128|0|pp|MUL|ffn_norm-17|29|1.027|
|tiny|row|128|128|0|pp|MUL|ffn_norm-18|29|1.034|
|tiny|row|128|128|0|pp|MUL|ffn_norm-19|29|1.104|
|tiny|row|128|128|0|pp|MUL|ffn_norm-2|29|0.956|
|tiny|row|128|128|0|pp|MUL|ffn_norm-20|29|1.158|
|tiny|row|128|128|0|pp|MUL|ffn_norm-21|29|0.329|
|tiny|row|128|128|0|pp|MUL|ffn_norm-3|29|1.115|
|tiny|row|128|128|0|pp|MUL|ffn_norm-4|29|1.103|
|tiny|row|128|128|0|pp|MUL|ffn_norm-5|29|1.093|
|tiny|row|128|128|0|pp|MUL|ffn_norm-6|29|1.174|
|tiny|row|128|128|0|pp|MUL|ffn_norm-7|29|1.135|
|tiny|row|128|128|0|pp|MUL|ffn_norm-8|29|1.079|
|tiny|row|128|128|0|pp|MUL|ffn_norm-9|29|0.976|
|tiny|row|128|128|0|pp|MUL|result_norm|29|0.360|
|tiny|row|128|128|0|pp|MUL_MAT|result_output|29|55.050|
|tiny|row|128|128|0|tg|GET_ROWS|embd|32|0.301|
|tiny|row|128|128|0|tg|GET_ROWS|node_678|32|0.057|
|tiny|row|128|128|0|tg|GET_ROWS|node_679|32|0.046|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-0|32|0.426|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-1|32|0.436|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-10|32|0.443|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-11|32|0.461|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-12|32|0.380|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-13|32|0.425|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-14|32|0.369|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-15|32|0.412|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-16|32|0.393|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-17|32|0.402|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-18|32|0.455|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-19|32|0.427|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-2|32|0.469|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-20|32|0.383|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-21|32|0.421|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-3|32|0.485|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-4|32|0.405|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-5|32|0.384|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-6|32|0.415|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-7|32|0.489|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-8|32|0.400|
|tiny|row|128|128|0|tg|GLU|ffn_swiglu-9|32|0.439|
|tiny|row|128|128|0|tg|MUL|attn_norm-0|32|0.292|
|tiny|row|128|128|0|tg|MUL|attn_norm-1|32|0.360|
|tiny|row|128|128|0|tg|MUL|attn_norm-10|32|0.331|
|tiny|row|128|128|0|tg|MUL|attn_norm-11|32|0.390|
|tiny|row|128|128|0|tg|MUL|attn_norm-12|32|0.321|
|tiny|row|128|128|0|tg|MUL|attn_norm-13|32|0.418|
|tiny|row|128|128|0|tg|MUL|attn_norm-14|32|0.300|
|tiny|row|128|128|0|tg|MUL|attn_norm-15|32|0.313|
|tiny|row|128|128|0|tg|MUL|attn_norm-16|32|0.329|
|tiny|row|128|128|0|tg|MUL|attn_norm-17|32|0.353|
|tiny|row|128|128|0|tg|MUL|attn_norm-18|32|0.328|
|tiny|row|128|128|0|tg|MUL|attn_norm-19|32|0.388|
|tiny|row|128|128|0|tg|MUL|attn_norm-2|32|0.318|
|tiny|row|128|128|0|tg|MUL|attn_norm-20|32|0.331|
|tiny|row|128|128|0|tg|MUL|attn_norm-21|32|0.390|
|tiny|row|128|128|0|tg|MUL|attn_norm-3|32|0.339|
|tiny|row|128|128|0|tg|MUL|attn_norm-4|32|0.358|
|tiny|row|128|128|0|tg|MUL|attn_norm-5|32|0.362|
|tiny|row|128|128|0|tg|MUL|attn_norm-6|32|0.343|
|tiny|row|128|128|0|tg|MUL|attn_norm-7|32|0.360|
|tiny|row|128|128|0|tg|MUL|attn_norm-8|32|0.360|
|tiny|row|128|128|0|tg|MUL|attn_norm-9|32|0.328|
|tiny|row|128|128|0|tg|MUL|ffn_norm-0|32|0.364|
|tiny|row|128|128|0|tg|MUL|ffn_norm-1|32|0.378|
|tiny|row|128|128|0|tg|MUL|ffn_norm-10|32|0.356|
|tiny|row|128|128|0|tg|MUL|ffn_norm-11|32|0.383|
|tiny|row|128|128|0|tg|MUL|ffn_norm-12|32|0.367|
|tiny|row|128|128|0|tg|MUL|ffn_norm-13|32|0.314|
|tiny|row|128|128|0|tg|MUL|ffn_norm-14|32|0.338|
|tiny|row|128|128|0|tg|MUL|ffn_norm-15|32|0.372|
|tiny|row|128|128|0|tg|MUL|ffn_norm-16|32|0.382|
|tiny|row|128|128|0|tg|MUL|ffn_norm-17|32|0.363|
|tiny|row|128|128|0|tg|MUL|ffn_norm-18|32|0.322|
|tiny|row|128|128|0|tg|MUL|ffn_norm-19|32|0.310|
|tiny|row|128|128|0|tg|MUL|ffn_norm-2|32|0.413|
|tiny|row|128|128|0|tg|MUL|ffn_norm-20|32|0.325|
|tiny|row|128|128|0|tg|MUL|ffn_norm-21|32|0.316|
|tiny|row|128|128|0|tg|MUL|ffn_norm-3|32|0.377|
|tiny|row|128|128|0|tg|MUL|ffn_norm-4|32|0.356|
|tiny|row|128|128|0|tg|MUL|ffn_norm-5|32|0.346|
|tiny|row|128|128|0|tg|MUL|ffn_norm-6|32|0.351|
|tiny|row|128|128|0|tg|MUL|ffn_norm-7|32|0.330|
|tiny|row|128|128|0|tg|MUL|ffn_norm-8|32|0.321|
|tiny|row|128|128|0|tg|MUL|ffn_norm-9|32|0.355|
|tiny|row|128|128|0|tg|MUL|result_norm|32|0.363|
|tiny|row|128|128|0|tg|MUL_MAT|result_output|32|58.564|
|tiny|row|128|128|1|pp|GET_ROWS|embd|29|1.425|
|tiny|row|128|128|1|pp|GET_ROWS|node_678|29|0.064|
|tiny|row|128|128|1|pp|GET_ROWS|node_679|29|0.044|
|tiny|row|128|128|1|pp|MUL|attn_norm-0|29|1.036|
|tiny|row|128|128|1|pp|MUL|attn_norm-1|29|1.042|
|tiny|row|128|128|1|pp|MUL|attn_norm-10|29|1.074|
|tiny|row|128|128|1|pp|MUL|attn_norm-11|29|1.004|
|tiny|row|128|128|1|pp|MUL|attn_norm-12|29|1.232|
|tiny|row|128|128|1|pp|MUL|attn_norm-13|29|1.079|
|tiny|row|128|128|1|pp|MUL|attn_norm-14|29|1.253|
|tiny|row|128|128|1|pp|MUL|attn_norm-15|29|1.094|
|tiny|row|128|128|1|pp|MUL|attn_norm-16|29|1.113|
|tiny|row|128|128|1|pp|MUL|attn_norm-17|29|1.149|
|tiny|row|128|128|1|pp|MUL|attn_norm-18|29|1.172|
|tiny|row|128|128|1|pp|MUL|attn_norm-19|29|1.107|
|tiny|row|128|128|1|pp|MUL|attn_norm-2|29|1.220|
|tiny|row|128|128|1|pp|MUL|attn_norm-20|29|1.372|
|tiny|row|128|128|1|pp|MUL|attn_norm-21|29|1.186|
|tiny|row|128|128|1|pp|MUL|attn_norm-3|29|1.019|
|tiny|row|128|128|1|pp|MUL|attn_norm-4|29|1.137|
|tiny|row|128|128|1|pp|MUL|attn_norm-5|29|1.212|
|tiny|row|128|128|1|pp|MUL|attn_norm-6|29|1.261|
|tiny|row|128|128|1|pp|MUL|attn_norm-7|29|1.268|
|tiny|row|128|128|1|pp|MUL|attn_norm-8|29|1.288|
|tiny|row|128|128|1|pp|MUL|attn_norm-9|29|1.175|
|tiny|row|128|128|1|pp|MUL|ffn_norm-0|29|1.450|
|tiny|row|128|128|1|pp|MUL|ffn_norm-1|29|1.323|
|tiny|row|128|128|1|pp|MUL|ffn_norm-10|29|1.019|
|tiny|row|128|128|1|pp|MUL|ffn_norm-11|29|1.204|
|tiny|row|128|128|1|pp|MUL|ffn_norm-12|29|1.255|
|tiny|row|128|128|1|pp|MUL|ffn_norm-13|29|1.159|
|tiny|row|128|128|1|pp|MUL|ffn_norm-14|29|1.176|
|tiny|row|128|128|1|pp|MUL|ffn_norm-15|29|1.274|
|tiny|row|128|128|1|pp|MUL|ffn_norm-16|29|1.175|
|tiny|row|128|128|1|pp|MUL|ffn_norm-17|29|1.328|
|tiny|row|128|128|1|pp|MUL|ffn_norm-18|29|1.050|
|tiny|row|128|128|1|pp|MUL|ffn_norm-19|29|1.191|
|tiny|row|128|128|1|pp|MUL|ffn_norm-2|29|1.192|
|tiny|row|128|128|1|pp|MUL|ffn_norm-20|29|1.323|
|tiny|row|128|128|1|pp|MUL|ffn_norm-21|29|0.445|
|tiny|row|128|128|1|pp|MUL|ffn_norm-3|29|1.064|
|tiny|row|128|128|1|pp|MUL|ffn_norm-4|29|1.185|
|tiny|row|128|128|1|pp|MUL|ffn_norm-5|29|1.210|
|tiny|row|128|128|1|pp|MUL|ffn_norm-6|29|1.071|
|tiny|row|128|128|1|pp|MUL|ffn_norm-7|29|1.208|
|tiny|row|128|128|1|pp|MUL|ffn_norm-8|29|1.208|
|tiny|row|128|128|1|pp|MUL|ffn_norm-9|29|1.265|
|tiny|row|128|128|1|pp|MUL|result_norm|29|0.400|
|tiny|row|128|128|1|pp|MUL_MAT|result_output|29|53.961|
|tiny|row|128|128|1|tg|GET_ROWS|embd|32|0.435|
|tiny|row|128|128|1|tg|GET_ROWS|node_678|32|0.057|
|tiny|row|128|128|1|tg|GET_ROWS|node_679|32|0.046|
|tiny|row|128|128|1|tg|MUL|attn_norm-0|32|0.257|
|tiny|row|128|128|1|tg|MUL|attn_norm-1|32|0.332|
|tiny|row|128|128|1|tg|MUL|attn_norm-10|32|0.369|
|tiny|row|128|128|1|tg|MUL|attn_norm-11|32|0.382|
|tiny|row|128|128|1|tg|MUL|attn_norm-12|32|0.375|
|tiny|row|128|128|1|tg|MUL|attn_norm-13|32|0.475|
|tiny|row|128|128|1|tg|MUL|attn_norm-14|32|0.405|
|tiny|row|128|128|1|tg|MUL|attn_norm-15|32|0.356|
|tiny|row|128|128|1|tg|MUL|attn_norm-16|32|0.324|
|tiny|row|128|128|1|tg|MUL|attn_norm-17|32|0.371|
|tiny|row|128|128|1|tg|MUL|attn_norm-18|32|0.366|
|tiny|row|128|128|1|tg|MUL|attn_norm-19|32|0.363|
|tiny|row|128|128|1|tg|MUL|attn_norm-2|32|0.415|
|tiny|row|128|128|1|tg|MUL|attn_norm-20|32|0.477|
|tiny|row|128|128|1|tg|MUL|attn_norm-21|32|0.366|
|tiny|row|128|128|1|tg|MUL|attn_norm-3|32|0.419|
|tiny|row|128|128|1|tg|MUL|attn_norm-4|32|0.381|
|tiny|row|128|128|1|tg|MUL|attn_norm-5|32|0.395|
|tiny|row|128|128|1|tg|MUL|attn_norm-6|32|0.407|
|tiny|row|128|128|1|tg|MUL|attn_norm-7|32|0.400|
|tiny|row|128|128|1|tg|MUL|attn_norm-8|32|0.402|
|tiny|row|128|128|1|tg|MUL|attn_norm-9|32|0.388|
|tiny|row|128|128|1|tg|MUL|ffn_norm-0|32|0.333|
|tiny|row|128|128|1|tg|MUL|ffn_norm-1|32|0.382|
|tiny|row|128|128|1|tg|MUL|ffn_norm-10|32|0.395|
|tiny|row|128|128|1|tg|MUL|ffn_norm-11|32|0.440|
|tiny|row|128|128|1|tg|MUL|ffn_norm-12|32|0.416|
|tiny|row|128|128|1|tg|MUL|ffn_norm-13|32|0.388|
|tiny|row|128|128|1|tg|MUL|ffn_norm-14|32|0.326|
|tiny|row|128|128|1|tg|MUL|ffn_norm-15|32|0.386|
|tiny|row|128|128|1|tg|MUL|ffn_norm-16|32|0.430|
|tiny|row|128|128|1|tg|MUL|ffn_norm-17|32|0.365|
|tiny|row|128|128|1|tg|MUL|ffn_norm-18|32|0.375|
|tiny|row|128|128|1|tg|MUL|ffn_norm-19|32|0.409|
|tiny|row|128|128|1|tg|MUL|ffn_norm-2|32|0.410|
|tiny|row|128|128|1|tg|MUL|ffn_norm-20|32|0.388|
|tiny|row|128|128|1|tg|MUL|ffn_norm-21|32|0.332|
|tiny|row|128|128|1|tg|MUL|ffn_norm-3|32|0.389|
|tiny|row|128|128|1|tg|MUL|ffn_norm-4|32|0.433|
|tiny|row|128|128|1|tg|MUL|ffn_norm-5|32|0.416|
|tiny|row|128|128|1|tg|MUL|ffn_norm-6|32|0.412|
|tiny|row|128|128|1|tg|MUL|ffn_norm-7|32|0.459|
|tiny|row|128|128|1|tg|MUL|ffn_norm-8|32|0.382|
|tiny|row|128|128|1|tg|MUL|ffn_norm-9|32|0.390|
|tiny|row|128|128|1|tg|MUL|result_norm|32|0.385|
|tiny|row|128|128|1|tg|MUL_MAT|result_output|32|58.818|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_117|7|108.130|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_148|7|108.284|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_179|7|115.234|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_210|7|108.438|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_24|7|117.867|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_241|7|108.879|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_272|7|107.861|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_303|7|107.233|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_334|7|110.438|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_365|7|110.328|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_396|7|110.605|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_427|7|110.486|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_458|7|110.924|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_489|7|112.728|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_520|7|113.893|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_55|7|118.042|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_551|7|109.794|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_582|7|109.208|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_613|7|118.356|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_644|7|107.161|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_675|7|107.479|
|tiny|row|512|512|0|pp|FLASH_ATTN_EXT|node_86|7|108.638|
|tiny|row|512|512|0|pp|GET_ROWS|embd|8|1.459|
|tiny|row|512|512|0|pp|GET_ROWS|node_678|8|0.021|
|tiny|row|512|512|0|pp|GET_ROWS|node_679|8|0.014|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-0|8|9.686|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-1|8|9.750|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-10|8|9.714|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-11|8|9.758|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-12|8|9.825|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-13|8|9.784|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-14|8|9.821|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-15|8|9.764|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-16|8|9.802|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-17|8|9.771|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-18|8|9.844|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-19|8|9.752|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-2|8|9.867|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-20|8|9.802|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-21|8|0.092|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-3|8|9.715|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-4|8|9.673|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-5|8|9.843|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-6|8|9.774|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-7|8|9.785|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-8|8|9.796|
|tiny|row|512|512|0|pp|GLU|ffn_swiglu-9|8|9.751|
|tiny|row|512|512|0|pp|MUL|attn_norm-0|8|0.772|
|tiny|row|512|512|0|pp|MUL|attn_norm-1|8|0.687|
|tiny|row|512|512|0|pp|MUL|attn_norm-10|8|0.693|
|tiny|row|512|512|0|pp|MUL|attn_norm-11|8|0.728|
|tiny|row|512|512|0|pp|MUL|attn_norm-12|8|0.728|
|tiny|row|512|512|0|pp|MUL|attn_norm-13|8|0.681|
|tiny|row|512|512|0|pp|MUL|attn_norm-14|8|0.621|
|tiny|row|512|512|0|pp|MUL|attn_norm-15|8|0.750|
|tiny|row|512|512|0|pp|MUL|attn_norm-16|8|0.772|
|tiny|row|512|512|0|pp|MUL|attn_norm-17|8|0.653|
|tiny|row|512|512|0|pp|MUL|attn_norm-18|8|0.684|
|tiny|row|512|512|0|pp|MUL|attn_norm-19|8|0.664|
|tiny|row|512|512|0|pp|MUL|attn_norm-2|8|0.700|
|tiny|row|512|512|0|pp|MUL|attn_norm-20|8|0.652|
|tiny|row|512|512|0|pp|MUL|attn_norm-21|8|0.716|
|tiny|row|512|512|0|pp|MUL|attn_norm-3|8|0.708|
|tiny|row|512|512|0|pp|MUL|attn_norm-4|8|0.700|
|tiny|row|512|512|0|pp|MUL|attn_norm-5|8|0.733|
|tiny|row|512|512|0|pp|MUL|attn_norm-6|8|0.653|
|tiny|row|512|512|0|pp|MUL|attn_norm-7|8|0.697|
|tiny|row|512|512|0|pp|MUL|attn_norm-8|8|0.715|
|tiny|row|512|512|0|pp|MUL|attn_norm-9|8|0.710|
|tiny|row|512|512|0|pp|MUL|ffn_norm-0|8|0.769|
|tiny|row|512|512|0|pp|MUL|ffn_norm-1|8|0.807|
|tiny|row|512|512|0|pp|MUL|ffn_norm-10|8|0.765|
|tiny|row|512|512|0|pp|MUL|ffn_norm-11|8|0.629|
|tiny|row|512|512|0|pp|MUL|ffn_norm-12|8|0.675|
|tiny|row|512|512|0|pp|MUL|ffn_norm-13|8|0.752|
|tiny|row|512|512|0|pp|MUL|ffn_norm-14|8|0.618|
|tiny|row|512|512|0|pp|MUL|ffn_norm-15|8|0.782|
|tiny|row|512|512|0|pp|MUL|ffn_norm-16|8|0.696|
|tiny|row|512|512|0|pp|MUL|ffn_norm-17|8|0.618|
|tiny|row|512|512|0|pp|MUL|ffn_norm-18|8|0.638|
|tiny|row|512|512|0|pp|MUL|ffn_norm-19|8|0.689|
|tiny|row|512|512|0|pp|MUL|ffn_norm-2|8|0.720|
|tiny|row|512|512|0|pp|MUL|ffn_norm-20|8|0.724|
|tiny|row|512|512|0|pp|MUL|ffn_norm-21|8|0.075|
|tiny|row|512|512|0|pp|MUL|ffn_norm-3|8|0.789|
|tiny|row|512|512|0|pp|MUL|ffn_norm-4|8|0.765|
|tiny|row|512|512|0|pp|MUL|ffn_norm-5|8|0.679|
|tiny|row|512|512|0|pp|MUL|ffn_norm-6|8|0.693|
|tiny|row|512|512|0|pp|MUL|ffn_norm-7|8|0.734|
|tiny|row|512|512|0|pp|MUL|ffn_norm-8|8|0.717|
|tiny|row|512|512|0|pp|MUL|ffn_norm-9|8|0.658|
|tiny|row|512|512|0|pp|MUL|result_norm|8|0.079|
|tiny|row|512|512|0|pp|MUL_MAT|result_output|8|15.248|
|tiny|row|512|512|0|pp|ROPE|Kcur-0|7|0.608|
|tiny|row|512|512|0|pp|ROPE|Kcur-1|7|0.490|
|tiny|row|512|512|0|pp|ROPE|Kcur-10|7|0.414|
|tiny|row|512|512|0|pp|ROPE|Kcur-11|7|0.497|
|tiny|row|512|512|0|pp|ROPE|Kcur-12|7|0.502|
|tiny|row|512|512|0|pp|ROPE|Kcur-13|7|0.431|
|tiny|row|512|512|0|pp|ROPE|Kcur-14|7|0.471|
|tiny|row|512|512|0|pp|ROPE|Kcur-15|7|0.446|
|tiny|row|512|512|0|pp|ROPE|Kcur-16|7|0.575|
|tiny|row|512|512|0|pp|ROPE|Kcur-17|7|0.506|
|tiny|row|512|512|0|pp|ROPE|Kcur-18|7|0.532|
|tiny|row|512|512|0|pp|ROPE|Kcur-19|7|0.501|
|tiny|row|512|512|0|pp|ROPE|Kcur-2|7|0.524|
|tiny|row|512|512|0|pp|ROPE|Kcur-20|7|0.542|
|tiny|row|512|512|0|pp|ROPE|Kcur-21|7|0.514|
|tiny|row|512|512|0|pp|ROPE|Kcur-3|7|0.485|
|tiny|row|512|512|0|pp|ROPE|Kcur-4|7|0.481|
|tiny|row|512|512|0|pp|ROPE|Kcur-5|7|0.492|
|tiny|row|512|512|0|pp|ROPE|Kcur-6|7|0.540|
|tiny|row|512|512|0|pp|ROPE|Kcur-7|7|0.506|
|tiny|row|512|512|0|pp|ROPE|Kcur-8|7|0.466|
|tiny|row|512|512|0|pp|ROPE|Kcur-9|7|0.471|
|tiny|row|512|512|0|pp|ROPE|Qcur-0|7|1.055|
|tiny|row|512|512|0|pp|ROPE|Qcur-1|7|0.989|
|tiny|row|512|512|0|pp|ROPE|Qcur-10|7|0.839|
|tiny|row|512|512|0|pp|ROPE|Qcur-11|7|0.964|
|tiny|row|512|512|0|pp|ROPE|Qcur-12|7|0.882|
|tiny|row|512|512|0|pp|ROPE|Qcur-13|7|0.964|
|tiny|row|512|512|0|pp|ROPE|Qcur-14|7|1.002|
|tiny|row|512|512|0|pp|ROPE|Qcur-15|7|0.942|
|tiny|row|512|512|0|pp|ROPE|Qcur-16|7|1.014|
|tiny|row|512|512|0|pp|ROPE|Qcur-17|7|0.922|
|tiny|row|512|512|0|pp|ROPE|Qcur-18|7|0.991|
|tiny|row|512|512|0|pp|ROPE|Qcur-19|7|0.888|
|tiny|row|512|512|0|pp|ROPE|Qcur-2|7|0.950|
|tiny|row|512|512|0|pp|ROPE|Qcur-20|7|0.902|
|tiny|row|512|512|0|pp|ROPE|Qcur-21|7|0.909|
|tiny|row|512|512|0|pp|ROPE|Qcur-3|7|0.856|
|tiny|row|512|512|0|pp|ROPE|Qcur-4|7|0.909|
|tiny|row|512|512|0|pp|ROPE|Qcur-5|7|1.065|
|tiny|row|512|512|0|pp|ROPE|Qcur-6|7|0.955|
|tiny|row|512|512|0|pp|ROPE|Qcur-7|7|0.950|
|tiny|row|512|512|0|pp|ROPE|Qcur-8|7|0.902|
|tiny|row|512|512|0|pp|ROPE|Qcur-9|7|0.916|
|tiny|row|512|512|0|tg|GET_ROWS|embd|32|0.286|
|tiny|row|512|512|0|tg|GET_ROWS|node_678|32|0.061|
|tiny|row|512|512|0|tg|GET_ROWS|node_679|32|0.049|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-0|32|0.363|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-1|32|0.353|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-10|32|0.336|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-11|32|0.352|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-12|32|0.320|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-13|32|0.326|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-14|32|0.337|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-15|32|0.328|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-16|32|0.338|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-17|32|0.339|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-18|32|0.325|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-19|32|0.323|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-2|32|0.333|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-20|32|0.370|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-21|32|0.346|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-3|32|0.348|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-4|32|0.357|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-5|32|0.318|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-6|32|0.325|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-7|32|0.334|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-8|32|0.348|
|tiny|row|512|512|0|tg|GLU|ffn_swiglu-9|32|0.324|
|tiny|row|512|512|0|tg|MUL|attn_norm-0|32|0.252|
|tiny|row|512|512|0|tg|MUL|attn_norm-1|32|0.303|
|tiny|row|512|512|0|tg|MUL|attn_norm-10|32|0.291|
|tiny|row|512|512|0|tg|MUL|attn_norm-11|32|0.312|
|tiny|row|512|512|0|tg|MUL|attn_norm-12|32|0.268|
|tiny|row|512|512|0|tg|MUL|attn_norm-13|32|0.314|
|tiny|row|512|512|0|tg|MUL|attn_norm-14|32|0.283|
|tiny|row|512|512|0|tg|MUL|attn_norm-15|32|0.278|
|tiny|row|512|512|0|tg|MUL|attn_norm-16|32|0.296|
|tiny|row|512|512|0|tg|MUL|attn_norm-17|32|0.295|
|tiny|row|512|512|0|tg|MUL|attn_norm-18|32|0.291|
|tiny|row|512|512|0|tg|MUL|attn_norm-19|32|0.318|
|tiny|row|512|512|0|tg|MUL|attn_norm-2|32|0.309|
|tiny|row|512|512|0|tg|MUL|attn_norm-20|32|0.310|
|tiny|row|512|512|0|tg|MUL|attn_norm-21|32|0.262|
|tiny|row|512|512|0|tg|MUL|attn_norm-3|32|0.288|
|tiny|row|512|512|0|tg|MUL|attn_norm-4|32|0.285|
|tiny|row|512|512|0|tg|MUL|attn_norm-5|32|0.295|
|tiny|row|512|512|0|tg|MUL|attn_norm-6|32|0.292|
|tiny|row|512|512|0|tg|MUL|attn_norm-7|32|0.281|
|tiny|row|512|512|0|tg|MUL|attn_norm-8|32|0.303|
|tiny|row|512|512|0|tg|MUL|attn_norm-9|32|0.269|
|tiny|row|512|512|0|tg|MUL|ffn_norm-0|32|0.309|
|tiny|row|512|512|0|tg|MUL|ffn_norm-1|32|0.273|
|tiny|row|512|512|0|tg|MUL|ffn_norm-10|32|0.303|
|tiny|row|512|512|0|tg|MUL|ffn_norm-11|32|0.291|
|tiny|row|512|512|0|tg|MUL|ffn_norm-12|32|0.309|
|tiny|row|512|512|0|tg|MUL|ffn_norm-13|32|0.298|
|tiny|row|512|512|0|tg|MUL|ffn_norm-14|32|0.303|
|tiny|row|512|512|0|tg|MUL|ffn_norm-15|32|0.291|
|tiny|row|512|512|0|tg|MUL|ffn_norm-16|32|0.277|
|tiny|row|512|512|0|tg|MUL|ffn_norm-17|32|0.284|
|tiny|row|512|512|0|tg|MUL|ffn_norm-18|32|0.298|
|tiny|row|512|512|0|tg|MUL|ffn_norm-19|32|0.271|
|tiny|row|512|512|0|tg|MUL|ffn_norm-2|32|0.280|
|tiny|row|512|512|0|tg|MUL|ffn_norm-20|32|0.294|
|tiny|row|512|512|0|tg|MUL|ffn_norm-21|32|0.326|
|tiny|row|512|512|0|tg|MUL|ffn_norm-3|32|0.315|
|tiny|row|512|512|0|tg|MUL|ffn_norm-4|32|0.303|
|tiny|row|512|512|0|tg|MUL|ffn_norm-5|32|0.277|
|tiny|row|512|512|0|tg|MUL|ffn_norm-6|32|0.288|
|tiny|row|512|512|0|tg|MUL|ffn_norm-7|32|0.304|
|tiny|row|512|512|0|tg|MUL|ffn_norm-8|32|0.308|
|tiny|row|512|512|0|tg|MUL|ffn_norm-9|32|0.322|
|tiny|row|512|512|0|tg|MUL|result_norm|32|0.270|
|tiny|row|512|512|0|tg|MUL_MAT|result_output|32|58.483|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_117|7|108.170|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_148|7|108.048|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_179|7|108.906|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_210|7|110.058|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_24|7|109.429|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_241|7|109.025|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_272|7|109.547|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_303|7|108.834|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_334|7|105.760|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_365|7|112.216|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_396|7|110.457|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_427|7|110.671|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_458|7|105.470|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_489|7|106.471|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_520|7|107.264|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_55|7|111.099|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_551|7|108.243|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_582|7|107.535|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_613|7|106.103|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_644|7|106.137|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_675|7|106.442|
|tiny|row|512|512|1|pp|FLASH_ATTN_EXT|node_86|7|112.245|
|tiny|row|512|512|1|pp|GET_ROWS|embd|8|1.355|
|tiny|row|512|512|1|pp|GET_ROWS|node_678|8|0.020|
|tiny|row|512|512|1|pp|GET_ROWS|node_679|8|0.013|
|tiny|row|512|512|1|pp|MUL|attn_norm-0|8|0.746|
|tiny|row|512|512|1|pp|MUL|attn_norm-1|8|0.677|
|tiny|row|512|512|1|pp|MUL|attn_norm-10|8|0.656|
|tiny|row|512|512|1|pp|MUL|attn_norm-11|8|0.682|
|tiny|row|512|512|1|pp|MUL|attn_norm-12|8|0.660|
|tiny|row|512|512|1|pp|MUL|attn_norm-13|8|0.715|
|tiny|row|512|512|1|pp|MUL|attn_norm-14|8|0.675|
|tiny|row|512|512|1|pp|MUL|attn_norm-15|8|0.633|
|tiny|row|512|512|1|pp|MUL|attn_norm-16|8|0.661|
|tiny|row|512|512|1|pp|MUL|attn_norm-17|8|0.635|
|tiny|row|512|512|1|pp|MUL|attn_norm-18|8|0.640|
|tiny|row|512|512|1|pp|MUL|attn_norm-19|8|0.662|
|tiny|row|512|512|1|pp|MUL|attn_norm-2|8|0.639|
|tiny|row|512|512|1|pp|MUL|attn_norm-20|8|0.733|
|tiny|row|512|512|1|pp|MUL|attn_norm-21|8|0.759|
|tiny|row|512|512|1|pp|MUL|attn_norm-3|8|0.679|
|tiny|row|512|512|1|pp|MUL|attn_norm-4|8|0.655|
|tiny|row|512|512|1|pp|MUL|attn_norm-5|8|0.630|
|tiny|row|512|512|1|pp|MUL|attn_norm-6|8|0.763|
|tiny|row|512|512|1|pp|MUL|attn_norm-7|8|0.696|
|tiny|row|512|512|1|pp|MUL|attn_norm-8|8|0.620|
|tiny|row|512|512|1|pp|MUL|attn_norm-9|8|0.705|
|tiny|row|512|512|1|pp|MUL|ffn_norm-0|8|0.655|
|tiny|row|512|512|1|pp|MUL|ffn_norm-1|8|0.806|
|tiny|row|512|512|1|pp|MUL|ffn_norm-10|8|0.750|
|tiny|row|512|512|1|pp|MUL|ffn_norm-11|8|0.669|
|tiny|row|512|512|1|pp|MUL|ffn_norm-12|8|0.711|
|tiny|row|512|512|1|pp|MUL|ffn_norm-13|8|0.817|
|tiny|row|512|512|1|pp|MUL|ffn_norm-14|8|0.678|
|tiny|row|512|512|1|pp|MUL|ffn_norm-15|8|0.715|
|tiny|row|512|512|1|pp|MUL|ffn_norm-16|8|0.746|
|tiny|row|512|512|1|pp|MUL|ffn_norm-17|8|0.694|
|tiny|row|512|512|1|pp|MUL|ffn_norm-18|8|0.764|
|tiny|row|512|512|1|pp|MUL|ffn_norm-19|8|0.709|
|tiny|row|512|512|1|pp|MUL|ffn_norm-2|8|0.745|
|tiny|row|512|512|1|pp|MUL|ffn_norm-20|8|0.662|
|tiny|row|512|512|1|pp|MUL|ffn_norm-21|8|0.088|
|tiny|row|512|512|1|pp|MUL|ffn_norm-3|8|0.716|
|tiny|row|512|512|1|pp|MUL|ffn_norm-4|8|0.777|
|tiny|row|512|512|1|pp|MUL|ffn_norm-5|8|0.763|
|tiny|row|512|512|1|pp|MUL|ffn_norm-6|8|0.703|
|tiny|row|512|512|1|pp|MUL|ffn_norm-7|8|0.661|
|tiny|row|512|512|1|pp|MUL|ffn_norm-8|8|0.775|
|tiny|row|512|512|1|pp|MUL|ffn_norm-9|8|0.760|
|tiny|row|512|512|1|pp|MUL|result_norm|8|0.071|
|tiny|row|512|512|1|pp|MUL_MAT|result_output|8|15.255|
|tiny|row|512|512|1|pp|ROPE|Kcur-0|7|0.512|
|tiny|row|512|512|1|pp|ROPE|Kcur-1|7|0.542|
|tiny|row|512|512|1|pp|ROPE|Kcur-10|7|0.473|
|tiny|row|512|512|1|pp|ROPE|Kcur-11|7|0.528|
|tiny|row|512|512|1|pp|ROPE|Kcur-12|7|0.546|
|tiny|row|512|512|1|pp|ROPE|Kcur-13|7|0.436|
|tiny|row|512|512|1|pp|ROPE|Kcur-14|7|0.531|
|tiny|row|512|512|1|pp|ROPE|Kcur-15|7|0.587|
|tiny|row|512|512|1|pp|ROPE|Kcur-16|7|0.450|
|tiny|row|512|512|1|pp|ROPE|Kcur-17|7|0.444|
|tiny|row|512|512|1|pp|ROPE|Kcur-18|7|0.460|
|tiny|row|512|512|1|pp|ROPE|Kcur-19|7|0.401|
|tiny|row|512|512|1|pp|ROPE|Kcur-2|7|0.539|
|tiny|row|512|512|1|pp|ROPE|Kcur-20|7|0.483|
|tiny|row|512|512|1|pp|ROPE|Kcur-21|7|0.395|
|tiny|row|512|512|1|pp|ROPE|Kcur-3|7|0.485|
|tiny|row|512|512|1|pp|ROPE|Kcur-4|7|0.484|
|tiny|row|512|512|1|pp|ROPE|Kcur-5|7|0.433|
|tiny|row|512|512|1|pp|ROPE|Kcur-6|7|0.542|
|tiny|row|512|512|1|pp|ROPE|Kcur-7|7|0.485|
|tiny|row|512|512|1|pp|ROPE|Kcur-8|7|0.551|
|tiny|row|512|512|1|pp|ROPE|Kcur-9|7|0.466|
|tiny|row|512|512|1|pp|ROPE|Qcur-0|7|0.995|
|tiny|row|512|512|1|pp|ROPE|Qcur-1|7|1.021|
|tiny|row|512|512|1|pp|ROPE|Qcur-10|7|0.927|
|tiny|row|512|512|1|pp|ROPE|Qcur-11|7|0.971|
|tiny|row|512|512|1|pp|ROPE|Qcur-12|7|0.933|
|tiny|row|512|512|1|pp|ROPE|Qcur-13|7|0.921|
|tiny|row|512|512|1|pp|ROPE|Qcur-14|7|0.914|
|tiny|row|512|512|1|pp|ROPE|Qcur-15|7|0.983|
|tiny|row|512|512|1|pp|ROPE|Qcur-16|7|0.941|
|tiny|row|512|512|1|pp|ROPE|Qcur-17|7|0.980|
|tiny|row|512|512|1|pp|ROPE|Qcur-18|7|1.002|
|tiny|row|512|512|1|pp|ROPE|Qcur-19|7|0.987|
|tiny|row|512|512|1|pp|ROPE|Qcur-2|7|1.002|
|tiny|row|512|512|1|pp|ROPE|Qcur-20|7|0.936|
|tiny|row|512|512|1|pp|ROPE|Qcur-21|7|0.902|
|tiny|row|512|512|1|pp|ROPE|Qcur-3|7|1.038|
|tiny|row|512|512|1|pp|ROPE|Qcur-4|7|0.932|
|tiny|row|512|512|1|pp|ROPE|Qcur-5|7|0.953|
|tiny|row|512|512|1|pp|ROPE|Qcur-6|7|1.013|
|tiny|row|512|512|1|pp|ROPE|Qcur-7|7|1.039|
|tiny|row|512|512|1|pp|ROPE|Qcur-8|7|0.936|
|tiny|row|512|512|1|pp|ROPE|Qcur-9|7|1.021|
|tiny|row|512|512|1|tg|GET_ROWS|embd|32|0.281|
|tiny|row|512|512|1|tg|GET_ROWS|node_678|32|0.060|
|tiny|row|512|512|1|tg|GET_ROWS|node_679|32|0.048|
|tiny|row|512|512|1|tg|MUL|attn_norm-0|32|0.260|
|tiny|row|512|512|1|tg|MUL|attn_norm-1|32|0.275|
|tiny|row|512|512|1|tg|MUL|attn_norm-10|32|0.288|
|tiny|row|512|512|1|tg|MUL|attn_norm-11|32|0.274|
|tiny|row|512|512|1|tg|MUL|attn_norm-12|32|0.296|
|tiny|row|512|512|1|tg|MUL|attn_norm-13|32|0.264|
|tiny|row|512|512|1|tg|MUL|attn_norm-14|32|0.266|
|tiny|row|512|512|1|tg|MUL|attn_norm-15|32|0.279|
|tiny|row|512|512|1|tg|MUL|attn_norm-16|32|0.310|
|tiny|row|512|512|1|tg|MUL|attn_norm-17|32|0.302|
|tiny|row|512|512|1|tg|MUL|attn_norm-18|32|0.268|
|tiny|row|512|512|1|tg|MUL|attn_norm-19|32|0.282|
|tiny|row|512|512|1|tg|MUL|attn_norm-2|32|0.303|
|tiny|row|512|512|1|tg|MUL|attn_norm-20|32|0.278|
|tiny|row|512|512|1|tg|MUL|attn_norm-21|32|0.275|
|tiny|row|512|512|1|tg|MUL|attn_norm-3|32|0.272|
|tiny|row|512|512|1|tg|MUL|attn_norm-4|32|0.282|
|tiny|row|512|512|1|tg|MUL|attn_norm-5|32|0.290|
|tiny|row|512|512|1|tg|MUL|attn_norm-6|32|0.281|
|tiny|row|512|512|1|tg|MUL|attn_norm-7|32|0.262|
|tiny|row|512|512|1|tg|MUL|attn_norm-8|32|0.276|
|tiny|row|512|512|1|tg|MUL|attn_norm-9|32|0.312|
|tiny|row|512|512|1|tg|MUL|ffn_norm-0|32|0.306|
|tiny|row|512|512|1|tg|MUL|ffn_norm-1|32|0.310|
|tiny|row|512|512|1|tg|MUL|ffn_norm-10|32|0.303|
|tiny|row|512|512|1|tg|MUL|ffn_norm-11|32|0.298|
|tiny|row|512|512|1|tg|MUL|ffn_norm-12|32|0.291|
|tiny|row|512|512|1|tg|MUL|ffn_norm-13|32|0.311|
|tiny|row|512|512|1|tg|MUL|ffn_norm-14|32|0.282|
|tiny|row|512|512|1|tg|MUL|ffn_norm-15|32|0.288|
|tiny|row|512|512|1|tg|MUL|ffn_norm-16|32|0.269|
|tiny|row|512|512|1|tg|MUL|ffn_norm-17|32|0.291|
|tiny|row|512|512|1|tg|MUL|ffn_norm-18|32|0.288|
|tiny|row|512|512|1|tg|MUL|ffn_norm-19|32|0.297|
|tiny|row|512|512|1|tg|MUL|ffn_norm-2|32|0.291|
|tiny|row|512|512|1|tg|MUL|ffn_norm-20|32|0.306|
|tiny|row|512|512|1|tg|MUL|ffn_norm-21|32|0.287|
|tiny|row|512|512|1|tg|MUL|ffn_norm-3|32|0.274|
|tiny|row|512|512|1|tg|MUL|ffn_norm-4|32|0.309|
|tiny|row|512|512|1|tg|MUL|ffn_norm-5|32|0.289|
|tiny|row|512|512|1|tg|MUL|ffn_norm-6|32|0.283|
|tiny|row|512|512|1|tg|MUL|ffn_norm-7|32|0.273|
|tiny|row|512|512|1|tg|MUL|ffn_norm-8|32|0.306|
|tiny|row|512|512|1|tg|MUL|ffn_norm-9|32|0.294|
|tiny|row|512|512|1|tg|MUL|result_norm|32|0.267|
|tiny|row|512|512|1|tg|MUL_MAT|result_output|32|58.598|


## 全構成の上位ボトルネック

全経路診断は各長さ1試行、TGは8decode。上記の代表3試行とは統計精度が異なる。転送は方向別のruntime論理bytes。

|model|split|b/ub|GLU|prefix|phase|ms/単位|第1|第2|第3|H2D KiB|D2H KiB|P2P KiB|
|---|---|---|---|---|---|---|---|---|---|---|---|---|
|7b|layer|128/128|0|1024|pp|14.6105|Q4 GEMV/GEMM (70.2%)|KV SET_ROWS XPU (7.6%)|Elementwise/residual XPU (5.6%)|1352.377|2699.551|16.000|
|7b|layer|128/128|0|1024|tg|165.0313|Q4 GEMV/GEMM (69.1%)|FlashAttention XPU (11.5%)|SwiGLU CPU graph (3.4%)|1461.194|2801.250|16.000|
|7b|layer|128/128|0|128|pp|14.2025|Q4 GEMV/GEMM (72.2%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.7%)|1350.885|2699.551|16.000|
|7b|layer|128/128|0|128|tg|154.0418|Q4 GEMV/GEMM (73.5%)|FlashAttention XPU (6.5%)|SwiGLU CPU graph (3.6%)|1457.194|2801.250|16.000|
|7b|layer|128/128|0|2048|pp|15.1240|Q4 GEMV/GEMM (67.8%)|FlashAttention XPU (8.7%)|KV SET_ROWS XPU (7.3%)|1354.377|2699.551|16.000|
|7b|layer|128/128|0|2048|tg|178.9490|Q4 GEMV/GEMM (63.3%)|FlashAttention XPU (17.0%)|RoPE XPU (4.8%)|1465.194|2801.250|16.000|
|7b|layer|128/128|0|512|pp|14.3154|Q4 GEMV/GEMM (71.7%)|KV SET_ROWS XPU (7.7%)|Elementwise/residual XPU (5.6%)|1351.378|2699.551|16.000|
|7b|layer|128/128|0|512|tg|159.2936|Q4 GEMV/GEMM (71.6%)|FlashAttention XPU (7.5%)|KV SET_ROWS XPU (3.9%)|1459.194|2801.250|16.000|
|7b|layer|128/128|1|1024|pp|14.2105|Q4 GEMV/GEMM (72.1%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.6%)|19.041|32.879|16.000|
|7b|layer|128/128|1|1024|tg|155.1538|Q4 GEMV/GEMM (72.9%)|FlashAttention XPU (12.2%)|KV SET_ROWS XPU (3.3%)|85.194|49.250|16.000|
|7b|layer|128/128|1|128|pp|13.8160|Q4 GEMV/GEMM (74.2%)|KV SET_ROWS XPU (8.0%)|Elementwise/residual XPU (5.8%)|17.549|32.879|16.000|
|7b|layer|128/128|1|128|tg|146.2135|Q4 GEMV/GEMM (77.9%)|FlashAttention XPU (6.9%)|Elementwise/residual XPU (3.2%)|81.194|49.250|16.000|
|7b|layer|128/128|1|2048|pp|14.7265|Q4 GEMV/GEMM (69.6%)|FlashAttention XPU (9.0%)|KV SET_ROWS XPU (7.5%)|21.041|32.879|16.000|
|7b|layer|128/128|1|2048|tg|171.7300|Q4 GEMV/GEMM (65.9%)|FlashAttention XPU (20.6%)|KV SET_ROWS XPU (3.0%)|89.194|49.250|16.000|
|7b|layer|128/128|1|512|pp|13.9182|Q4 GEMV/GEMM (73.6%)|KV SET_ROWS XPU (7.9%)|Elementwise/residual XPU (5.8%)|18.042|32.879|16.000|
|7b|layer|128/128|1|512|tg|149.4730|Q4 GEMV/GEMM (75.7%)|FlashAttention XPU (8.7%)|KV SET_ROWS XPU (3.5%)|83.194|49.250|16.000|
|7b|layer|512/512|0|1024|pp|15.1402|Q4 GEMV/GEMM (67.5%)|Attention CPU graph (8.5%)|KV SET_ROWS XPU (7.9%)|2373.240|4490.700|16.000|
|7b|layer|512/512|0|1024|tg|171.3147|Q4 GEMV/GEMM (66.4%)|FlashAttention XPU (14.4%)|Elementwise/residual XPU (4.4%)|1461.039|2801.250|16.000|
|7b|layer|512/512|0|128|pp|13.9451|Q4 GEMV/GEMM (72.7%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.3%)|1350.885|2699.551|16.000|
|7b|layer|512/512|0|128|tg|158.7583|Q4 GEMV/GEMM (72.2%)|FlashAttention XPU (6.4%)|SwiGLU CPU graph (3.9%)|1457.194|2801.250|16.000|
|7b|layer|512/512|0|2048|pp|16.3923|Q4 GEMV/GEMM (62.4%)|Attention CPU graph (15.0%)|KV SET_ROWS XPU (7.3%)|2373.240|5002.700|16.000|
|7b|layer|512/512|0|2048|tg|184.6157|Q4 GEMV/GEMM (61.5%)|FlashAttention XPU (22.2%)|SwiGLU CPU graph (2.9%)|1465.039|2801.250|16.000|
|7b|layer|512/512|0|512|pp|14.5883|Q4 GEMV/GEMM (69.8%)|KV SET_ROWS XPU (8.2%)|Attention CPU graph (5.6%)|2373.240|4234.700|16.000|
|7b|layer|512/512|0|512|tg|168.1620|Q4 GEMV/GEMM (69.4%)|FlashAttention XPU (9.3%)|SwiGLU CPU graph (3.7%)|1459.039|2801.250|16.000|
|7b|layer|512/512|1|1024|pp|14.9103|Q4 GEMV/GEMM (68.6%)|Attention CPU graph (9.3%)|KV SET_ROWS XPU (8.0%)|1040.156|1824.532|16.000|
|7b|layer|512/512|1|1024|tg|162.6844|Q4 GEMV/GEMM (69.6%)|FlashAttention XPU (15.1%)|Elementwise/residual XPU (2.9%)|85.039|49.250|16.000|
|7b|layer|512/512|1|128|pp|13.8234|Q4 GEMV/GEMM (74.2%)|KV SET_ROWS XPU (8.0%)|Elementwise/residual XPU (5.8%)|17.549|32.879|16.000|
|7b|layer|512/512|1|128|tg|146.8271|Q4 GEMV/GEMM (77.8%)|FlashAttention XPU (6.8%)|Elementwise/residual XPU (3.3%)|81.194|49.250|16.000|
|7b|layer|512/512|1|2048|pp|16.1677|Q4 GEMV/GEMM (63.3%)|Attention CPU graph (15.8%)|KV SET_ROWS XPU (7.4%)|1040.156|2336.532|16.000|
|7b|layer|512/512|1|2048|tg|176.5481|Q4 GEMV/GEMM (64.1%)|FlashAttention XPU (23.2%)|Elementwise/residual XPU (2.7%)|89.039|49.250|16.000|
|7b|layer|512/512|1|512|pp|14.3043|Q4 GEMV/GEMM (71.5%)|KV SET_ROWS XPU (8.3%)|Attention CPU graph (5.7%)|1040.156|1568.532|16.000|
|7b|layer|512/512|1|512|tg|152.3752|Q4 GEMV/GEMM (74.4%)|FlashAttention XPU (10.0%)|Elementwise/residual XPU (3.9%)|83.039|49.250|16.000|
|7b|pd0|512/128|0|1024|pp|14.5939|Q4 GEMV/GEMM (70.3%)|KV SET_ROWS XPU (7.6%)|Elementwise/residual XPU (5.6%)|1350.667|2699.141|0.000|
|7b|pd0|512/128|0|1024|tg|163.4105|Q4 GEMV/GEMM (69.7%)|FlashAttention XPU (11.6%)|KV SET_ROWS XPU (3.2%)|1458.597|2801.250|0.000|
|7b|pd0|512/128|0|128|pp|14.2430|Q4 GEMV/GEMM (72.2%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.7%)|1350.360|2699.551|0.000|
|7b|pd0|512/128|0|128|tg|155.1349|Q4 GEMV/GEMM (73.9%)|FlashAttention XPU (6.7%)|SwiGLU CPU graph (3.6%)|1456.597|2801.250|0.000|
|7b|pd0|512/128|0|2048|pp|15.0906|Q4 GEMV/GEMM (68.0%)|FlashAttention XPU (8.5%)|KV SET_ROWS XPU (7.3%)|1351.667|2699.141|0.000|
|7b|pd0|512/128|0|2048|tg|179.9335|Q4 GEMV/GEMM (66.0%)|FlashAttention XPU (16.9%)|SwiGLU CPU graph (3.0%)|1460.597|2801.250|0.000|
|7b|pd0|512/128|0|512|pp|14.3222|Q4 GEMV/GEMM (71.7%)|KV SET_ROWS XPU (7.7%)|Elementwise/residual XPU (5.7%)|1350.168|2699.141|0.000|
|7b|pd0|512/128|0|512|tg|157.4481|Q4 GEMV/GEMM (72.3%)|FlashAttention XPU (7.6%)|KV SET_ROWS XPU (3.3%)|1457.597|2801.250|0.000|
|7b|pd0|512/128|1|1024|pp|14.1271|Q4 GEMV/GEMM (72.3%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.5%)|17.583|32.973|0.000|
|7b|pd0|512/128|1|1024|tg|155.9422|Q4 GEMV/GEMM (73.0%)|FlashAttention XPU (12.1%)|KV SET_ROWS XPU (3.3%)|82.597|49.250|0.000|
|7b|pd0|512/128|1|128|pp|13.8349|Q4 GEMV/GEMM (74.2%)|KV SET_ROWS XPU (8.0%)|Elementwise/residual XPU (5.8%)|17.024|32.879|0.000|
|7b|pd0|512/128|1|128|tg|147.4372|Q4 GEMV/GEMM (77.8%)|FlashAttention XPU (6.9%)|Elementwise/residual XPU (3.3%)|80.597|49.250|0.000|
|7b|pd0|512/128|1|2048|pp|14.5651|Q4 GEMV/GEMM (69.9%)|FlashAttention XPU (8.8%)|KV SET_ROWS XPU (7.5%)|18.583|32.973|0.000|
|7b|pd0|512/128|1|2048|tg|175.7351|Q4 GEMV/GEMM (68.5%)|FlashAttention XPU (17.6%)|KV SET_ROWS XPU (3.1%)|84.597|49.250|0.000|
|7b|pd0|512/128|1|512|pp|13.9436|Q4 GEMV/GEMM (73.6%)|KV SET_ROWS XPU (7.9%)|Elementwise/residual XPU (5.8%)|17.084|32.973|0.000|
|7b|pd0|512/128|1|512|tg|150.0666|Q4 GEMV/GEMM (75.8%)|FlashAttention XPU (8.7%)|KV SET_ROWS XPU (3.5%)|81.597|49.250|0.000|
|7b|pd0|128/128|0|1024|pp|14.4980|Q4 GEMV/GEMM (70.6%)|KV SET_ROWS XPU (7.6%)|Elementwise/residual XPU (5.4%)|1351.107|2699.551|0.000|
|7b|pd0|128/128|0|1024|tg|164.2159|Q4 GEMV/GEMM (69.7%)|FlashAttention XPU (11.6%)|SwiGLU CPU graph (3.4%)|1458.597|2801.250|0.000|
|7b|pd0|128/128|0|128|pp|14.2485|Q4 GEMV/GEMM (72.2%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.7%)|1350.360|2699.551|0.000|
|7b|pd0|128/128|0|128|tg|155.0405|Q4 GEMV/GEMM (74.3%)|FlashAttention XPU (6.7%)|Elementwise/residual XPU (3.1%)|1456.597|2801.250|0.000|
|7b|pd0|128/128|0|2048|pp|14.8888|Q4 GEMV/GEMM (68.4%)|FlashAttention XPU (8.6%)|KV SET_ROWS XPU (7.3%)|1352.106|2699.551|0.000|
|7b|pd0|128/128|0|2048|tg|183.9189|Q4 GEMV/GEMM (62.7%)|FlashAttention XPU (16.8%)|Elementwise/residual XPU (5.6%)|1460.597|2801.250|0.000|
|7b|pd0|128/128|0|512|pp|14.3239|Q4 GEMV/GEMM (71.8%)|KV SET_ROWS XPU (7.7%)|Elementwise/residual XPU (5.6%)|1350.607|2699.551|0.000|
|7b|pd0|128/128|0|512|tg|157.9920|Q4 GEMV/GEMM (73.1%)|FlashAttention XPU (7.6%)|SwiGLU CPU graph (3.4%)|1457.597|2801.250|0.000|
|7b|pd0|128/128|1|1024|pp|14.2163|Q4 GEMV/GEMM (72.4%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.7%)|17.771|32.879|0.000|
|7b|pd0|128/128|1|1024|tg|156.3004|Q4 GEMV/GEMM (73.1%)|FlashAttention XPU (12.1%)|KV SET_ROWS XPU (3.3%)|82.597|49.250|0.000|
|7b|pd0|128/128|1|128|pp|13.8549|Q4 GEMV/GEMM (74.2%)|KV SET_ROWS XPU (8.0%)|Elementwise/residual XPU (5.8%)|17.024|32.879|0.000|
|7b|pd0|128/128|1|128|tg|147.6544|Q4 GEMV/GEMM (77.9%)|FlashAttention XPU (6.9%)|Elementwise/residual XPU (3.3%)|80.597|49.250|0.000|
|7b|pd0|128/128|1|2048|pp|14.7169|Q4 GEMV/GEMM (69.8%)|FlashAttention XPU (8.7%)|KV SET_ROWS XPU (7.5%)|18.770|32.879|0.000|
|7b|pd0|128/128|1|2048|tg|171.9429|Q4 GEMV/GEMM (66.1%)|FlashAttention XPU (17.7%)|Remaining host/scheduler/measurement (3.4%)|84.597|49.250|0.000|
|7b|pd0|128/128|1|512|pp|13.9592|Q4 GEMV/GEMM (73.6%)|KV SET_ROWS XPU (7.9%)|Elementwise/residual XPU (5.8%)|17.271|32.879|0.000|
|7b|pd0|128/128|1|512|tg|150.4331|Q4 GEMV/GEMM (76.7%)|FlashAttention XPU (7.9%)|KV SET_ROWS XPU (3.5%)|81.597|49.250|0.000|
|7b|pd0|512/512|0|1024|pp|15.1446|Q4 GEMV/GEMM (67.5%)|Attention CPU graph (8.5%)|KV SET_ROWS XPU (7.9%)|2373.225|4490.700|0.000|
|7b|pd0|512/512|0|1024|tg|171.1290|Q4 GEMV/GEMM (68.1%)|FlashAttention XPU (14.4%)|SwiGLU CPU graph (3.2%)|1458.520|2801.250|0.000|
|7b|pd0|512/512|0|128|pp|14.2295|Q4 GEMV/GEMM (72.2%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.7%)|1350.360|2699.551|0.000|
|7b|pd0|512/512|0|128|tg|154.9202|Q4 GEMV/GEMM (74.0%)|FlashAttention XPU (6.7%)|SwiGLU CPU graph (3.3%)|1456.597|2801.250|0.000|
|7b|pd0|512/512|0|2048|pp|16.3751|Q4 GEMV/GEMM (62.5%)|Attention CPU graph (14.9%)|KV SET_ROWS XPU (7.3%)|2373.225|5002.700|0.000|
|7b|pd0|512/512|0|2048|tg|184.6602|Q4 GEMV/GEMM (61.7%)|FlashAttention XPU (22.1%)|SwiGLU CPU graph (3.0%)|1460.520|2801.250|0.000|
|7b|pd0|512/512|0|512|pp|14.6162|Q4 GEMV/GEMM (70.0%)|KV SET_ROWS XPU (8.2%)|Attention CPU graph (5.5%)|2373.225|4234.700|0.000|
|7b|pd0|512/512|0|512|tg|160.0357|Q4 GEMV/GEMM (71.2%)|FlashAttention XPU (10.2%)|SwiGLU CPU graph (3.1%)|1457.520|2801.250|0.000|
|7b|pd0|512/512|1|1024|pp|14.8405|Q4 GEMV/GEMM (68.9%)|Attention CPU graph (9.0%)|KV SET_ROWS XPU (8.0%)|1040.141|1824.532|0.000|
|7b|pd0|512/512|1|1024|tg|163.4181|Q4 GEMV/GEMM (71.1%)|FlashAttention XPU (15.1%)|Elementwise/residual XPU (3.0%)|82.520|49.250|0.000|
|7b|pd0|512/512|1|128|pp|13.7445|Q4 GEMV/GEMM (74.4%)|KV SET_ROWS XPU (8.0%)|Elementwise/residual XPU (5.7%)|17.024|32.879|0.000|
|7b|pd0|512/512|1|128|tg|147.2138|Q4 GEMV/GEMM (77.9%)|FlashAttention XPU (6.9%)|Elementwise/residual XPU (3.3%)|80.597|49.250|0.000|
|7b|pd0|512/512|1|2048|pp|16.0352|Q4 GEMV/GEMM (63.4%)|Attention CPU graph (15.8%)|KV SET_ROWS XPU (7.4%)|1040.141|2336.532|0.000|
|7b|pd0|512/512|1|2048|tg|182.1773|Q4 GEMV/GEMM (63.7%)|FlashAttention XPU (22.8%)|Elementwise/residual XPU (3.1%)|84.520|49.250|0.000|
|7b|pd0|512/512|1|512|pp|14.1922|Q4 GEMV/GEMM (71.7%)|KV SET_ROWS XPU (8.4%)|Attention CPU graph (5.5%)|1040.141|1568.532|0.000|
|7b|pd0|512/512|1|512|tg|152.4651|Q4 GEMV/GEMM (75.3%)|FlashAttention XPU (10.0%)|Elementwise/residual XPU (3.1%)|81.520|49.250|0.000|
|7b|pd1|128/128|0|1024|pp|14.5406|Q4 GEMV/GEMM (70.4%)|KV SET_ROWS XPU (7.6%)|Elementwise/residual XPU (5.5%)|1351.107|2699.551|0.000|
|7b|pd1|128/128|0|1024|tg|160.8453|Q4 GEMV/GEMM (69.5%)|FlashAttention XPU (11.7%)|KV SET_ROWS XPU (3.2%)|1458.597|2801.250|0.000|
|7b|pd1|128/128|0|128|pp|14.1609|Q4 GEMV/GEMM (72.3%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.6%)|1350.360|2699.551|0.000|
|7b|pd1|128/128|0|128|tg|151.9153|Q4 GEMV/GEMM (73.9%)|FlashAttention XPU (6.7%)|SwiGLU CPU graph (3.3%)|1456.597|2801.250|0.000|
|7b|pd1|128/128|0|2048|pp|15.0310|Q4 GEMV/GEMM (68.1%)|FlashAttention XPU (8.4%)|KV SET_ROWS XPU (7.4%)|1352.106|2699.551|0.000|
|7b|pd1|128/128|0|2048|tg|177.0357|Q4 GEMV/GEMM (63.0%)|FlashAttention XPU (17.1%)|Elementwise/residual XPU (5.5%)|1460.597|2801.250|0.000|
|7b|pd1|128/128|0|512|pp|14.2566|Q4 GEMV/GEMM (71.8%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.6%)|1350.607|2699.551|0.000|
|7b|pd1|128/128|0|512|tg|155.2822|Q4 GEMV/GEMM (72.7%)|FlashAttention XPU (7.6%)|SwiGLU CPU graph (3.3%)|1457.597|2801.250|0.000|
|7b|pd1|128/128|1|1024|pp|14.0095|Q4 GEMV/GEMM (72.7%)|KV SET_ROWS XPU (7.8%)|FlashAttention XPU (5.4%)|17.771|32.879|0.000|
|7b|pd1|128/128|1|1024|tg|158.1104|Q4 GEMV/GEMM (71.9%)|FlashAttention XPU (12.3%)|Elementwise/residual XPU (3.6%)|82.597|49.250|0.000|
|7b|pd1|128/128|1|128|pp|13.6506|Q4 GEMV/GEMM (74.5%)|KV SET_ROWS XPU (8.0%)|Elementwise/residual XPU (5.6%)|17.024|32.879|0.000|
|7b|pd1|128/128|1|128|tg|144.4023|Q4 GEMV/GEMM (77.7%)|FlashAttention XPU (6.8%)|Elementwise/residual XPU (3.4%)|80.597|49.250|0.000|
|7b|pd1|128/128|1|2048|pp|14.5050|Q4 GEMV/GEMM (70.1%)|FlashAttention XPU (8.7%)|KV SET_ROWS XPU (7.5%)|18.770|32.879|0.000|
|7b|pd1|128/128|1|2048|tg|171.6922|Q4 GEMV/GEMM (65.5%)|FlashAttention XPU (17.8%)|Remaining host/scheduler/measurement (3.4%)|84.597|49.250|0.000|
|7b|pd1|128/128|1|512|pp|13.7397|Q4 GEMV/GEMM (74.0%)|KV SET_ROWS XPU (7.9%)|Elementwise/residual XPU (5.5%)|17.271|32.879|0.000|
|7b|pd1|128/128|1|512|tg|148.9200|Q4 GEMV/GEMM (76.0%)|FlashAttention XPU (8.2%)|KV SET_ROWS XPU (3.5%)|81.597|49.250|0.000|
|7b|pd1|512/512|0|1024|pp|15.1222|Q4 GEMV/GEMM (67.4%)|Attention CPU graph (8.7%)|KV SET_ROWS XPU (7.9%)|2373.225|4490.700|0.000|
|7b|pd1|512/512|0|1024|tg|168.4196|Q4 GEMV/GEMM (67.7%)|FlashAttention XPU (14.5%)|SwiGLU CPU graph (3.3%)|1458.520|2801.250|0.000|
|7b|pd1|512/512|0|128|pp|14.1330|Q4 GEMV/GEMM (72.3%)|KV SET_ROWS XPU (7.8%)|Elementwise/residual XPU (5.5%)|1350.360|2699.551|0.000|
|7b|pd1|512/512|0|128|tg|151.7010|Q4 GEMV/GEMM (73.9%)|FlashAttention XPU (6.7%)|SwiGLU CPU graph (3.2%)|1456.597|2801.250|0.000|
|7b|pd1|512/512|0|2048|pp|16.2859|Q4 GEMV/GEMM (62.3%)|Attention CPU graph (15.1%)|KV SET_ROWS XPU (7.3%)|2373.225|5002.700|0.000|
|7b|pd1|512/512|0|2048|tg|182.2914|Q4 GEMV/GEMM (61.2%)|FlashAttention XPU (22.4%)|SwiGLU CPU graph (3.0%)|1460.520|2801.250|0.000|
|7b|pd1|512/512|0|512|pp|14.5805|Q4 GEMV/GEMM (70.0%)|KV SET_ROWS XPU (8.2%)|Attention CPU graph (5.5%)|2373.225|4234.700|0.000|
|7b|pd1|512/512|0|512|tg|157.5748|Q4 GEMV/GEMM (70.8%)|FlashAttention XPU (10.4%)|SwiGLU CPU graph (3.4%)|1457.520|2801.250|0.000|
|7b|pd1|512/512|1|1024|pp|14.8380|Q4 GEMV/GEMM (68.5%)|Attention CPU graph (9.4%)|KV SET_ROWS XPU (8.0%)|1040.141|1824.532|0.000|
|7b|pd1|512/512|1|1024|tg|160.5964|Q4 GEMV/GEMM (70.8%)|FlashAttention XPU (15.3%)|Elementwise/residual XPU (3.0%)|82.520|49.250|0.000|
|7b|pd1|512/512|1|128|pp|13.7616|Q4 GEMV/GEMM (74.3%)|KV SET_ROWS XPU (8.0%)|Elementwise/residual XPU (5.7%)|17.024|32.879|0.000|
|7b|pd1|512/512|1|128|tg|144.6840|Q4 GEMV/GEMM (77.6%)|FlashAttention XPU (6.8%)|Elementwise/residual XPU (3.4%)|80.597|49.250|0.000|
|7b|pd1|512/512|1|2048|pp|15.9881|Q4 GEMV/GEMM (63.5%)|Attention CPU graph (15.6%)|KV SET_ROWS XPU (7.4%)|1040.141|2336.532|0.000|
|7b|pd1|512/512|1|2048|tg|176.3687|Q4 GEMV/GEMM (63.6%)|FlashAttention XPU (23.3%)|Elementwise/residual XPU (2.9%)|84.520|49.250|0.000|
|7b|pd1|512/512|1|512|pp|14.2430|Q4 GEMV/GEMM (71.5%)|KV SET_ROWS XPU (8.4%)|Attention CPU graph (5.8%)|1040.141|1568.532|0.000|
|7b|pd1|512/512|1|512|tg|150.1746|Q4 GEMV/GEMM (74.9%)|FlashAttention XPU (10.1%)|Elementwise/residual XPU (3.2%)|81.520|49.250|0.000|
|7b|row|128/128|0|1024|pp|21.8237|Q4 GEMV/GEMM (80.1%)|KV SET_ROWS XPU (5.0%)|FlashAttention XPU (3.5%)|2360.502|3715.801|6994.984|
|7b|row|128/128|0|1024|tg|242.1630|Q4 GEMV/GEMM (73.2%)|FlashAttention XPU (8.0%)|Other CPU graph (4.6%)|2485.194|4865.250|7120.000|
|7b|row|128/128|0|128|pp|21.7908|Q4 GEMV/GEMM (81.3%)|KV SET_ROWS XPU (5.1%)|External D2H (2.9%)|2359.010|3715.801|6994.984|
|7b|row|128/128|0|128|tg|230.2676|Q4 GEMV/GEMM (75.5%)|Other CPU graph (4.5%)|FlashAttention XPU (4.4%)|2481.194|4865.250|7120.000|
|7b|row|128/128|0|2048|pp|22.3400|Q4 GEMV/GEMM (78.2%)|FlashAttention XPU (5.7%)|KV SET_ROWS XPU (4.9%)|2362.502|3715.801|6994.984|
|7b|row|128/128|0|2048|tg|254.1693|Q4 GEMV/GEMM (70.8%)|FlashAttention XPU (12.0%)|Other CPU graph (4.1%)|2489.194|4865.250|7120.000|
|7b|row|128/128|0|512|pp|21.8501|Q4 GEMV/GEMM (81.0%)|KV SET_ROWS XPU (5.1%)|External D2H (2.8%)|2359.503|3715.801|6994.984|
|7b|row|128/128|0|512|tg|229.4020|Q4 GEMV/GEMM (75.7%)|FlashAttention XPU (5.2%)|Other CPU graph (4.5%)|2483.194|4865.250|7120.000|
|7b|row|128/128|1|1024|pp|21.7217|Q4 GEMV/GEMM (81.5%)|KV SET_ROWS XPU (5.1%)|FlashAttention XPU (3.5%)|1027.166|1049.129|6994.984|
|7b|row|128/128|1|1024|tg|230.9137|Q4 GEMV/GEMM (73.6%)|FlashAttention XPU (8.2%)|Other CPU graph (4.4%)|1109.194|2113.250|7120.000|
|7b|row|128/128|1|128|pp|21.4012|Q4 GEMV/GEMM (82.8%)|KV SET_ROWS XPU (5.2%)|SwiGLU XPU (2.2%)|1025.674|1049.129|6994.984|
|7b|row|128/128|1|128|tg|218.3976|Q4 GEMV/GEMM (78.2%)|FlashAttention XPU (4.6%)|Other CPU graph (4.6%)|1105.194|2113.250|7120.000|
|7b|row|128/128|1|2048|pp|22.2666|Q4 GEMV/GEMM (79.6%)|FlashAttention XPU (5.7%)|KV SET_ROWS XPU (5.0%)|1029.166|1049.129|6994.984|
|7b|row|128/128|1|2048|tg|247.6648|Q4 GEMV/GEMM (72.7%)|FlashAttention XPU (12.3%)|Other CPU graph (4.0%)|1113.194|2113.250|7120.000|
|7b|row|128/128|1|512|pp|21.4934|Q4 GEMV/GEMM (82.4%)|KV SET_ROWS XPU (5.1%)|FlashAttention XPU (2.3%)|1026.167|1049.129|6994.984|
|7b|row|128/128|1|512|tg|221.5007|Q4 GEMV/GEMM (77.9%)|FlashAttention XPU (5.4%)|Other CPU graph (4.5%)|1107.194|2113.250|7120.000|
|7b|row|512/512|0|1024|pp|22.2436|Q4 GEMV/GEMM (77.1%)|Attention CPU graph (6.6%)|KV SET_ROWS XPU (5.4%)|3381.271|5500.762|6994.246|
|7b|row|512/512|0|1024|tg|246.9176|Q4 GEMV/GEMM (71.9%)|FlashAttention XPU (9.9%)|Other CPU graph (4.1%)|2485.039|4865.250|7120.000|
|7b|row|512/512|0|128|pp|21.8008|Q4 GEMV/GEMM (81.3%)|KV SET_ROWS XPU (5.1%)|External D2H (2.9%)|2359.010|3715.801|6994.984|
|7b|row|512/512|0|128|tg|223.8717|Q4 GEMV/GEMM (76.5%)|FlashAttention XPU (4.5%)|Other CPU graph (4.4%)|2481.194|4865.250|7120.000|
|7b|row|512/512|0|2048|pp|23.5016|Q4 GEMV/GEMM (73.1%)|Attention CPU graph (11.1%)|KV SET_ROWS XPU (5.1%)|3381.271|6012.762|6994.246|
|7b|row|512/512|0|2048|tg|253.4849|Q4 GEMV/GEMM (67.0%)|FlashAttention XPU (16.2%)|Other CPU graph (3.9%)|2489.039|4865.250|7120.000|
|7b|row|512/512|0|512|pp|21.6176|Q4 GEMV/GEMM (79.6%)|KV SET_ROWS XPU (5.5%)|External D2H (3.8%)|3381.271|5244.762|6994.246|
|7b|row|512/512|0|512|tg|229.9907|Q4 GEMV/GEMM (73.8%)|FlashAttention XPU (6.6%)|Other CPU graph (4.4%)|2483.039|4865.250|7120.000|
|7b|row|512/512|1|1024|pp|21.8963|Q4 GEMV/GEMM (78.5%)|Attention CPU graph (6.4%)|KV SET_ROWS XPU (5.5%)|2048.188|2834.594|6994.246|
|7b|row|512/512|1|1024|tg|236.0307|Q4 GEMV/GEMM (74.2%)|FlashAttention XPU (10.4%)|Other CPU graph (4.5%)|1109.039|2113.250|7120.000|
|7b|row|512/512|1|128|pp|21.3647|Q4 GEMV/GEMM (82.8%)|KV SET_ROWS XPU (5.2%)|SwiGLU XPU (2.2%)|1025.674|1049.129|6994.984|
|7b|row|512/512|1|128|tg|218.3361|Q4 GEMV/GEMM (78.2%)|Other CPU graph (4.6%)|FlashAttention XPU (4.6%)|1105.194|2113.250|7120.000|
|7b|row|512/512|1|2048|pp|23.2219|Q4 GEMV/GEMM (74.0%)|Attention CPU graph (11.4%)|KV SET_ROWS XPU (5.1%)|2048.188|3346.594|6994.246|
|7b|row|512/512|1|2048|tg|247.5807|Q4 GEMV/GEMM (68.7%)|FlashAttention XPU (16.6%)|Other CPU graph (4.3%)|1113.039|2113.250|7120.000|
|7b|row|512/512|1|512|pp|21.2001|Q4 GEMV/GEMM (80.7%)|KV SET_ROWS XPU (5.6%)|Attention CPU graph (3.9%)|2048.188|2578.594|6994.246|
|7b|row|512/512|1|512|tg|226.4907|Q4 GEMV/GEMM (75.9%)|FlashAttention XPU (6.8%)|Other CPU graph (4.7%)|1107.039|2113.250|7120.000|
|tiny|layer|128/128|0|1024|pp|3.8396|Q4 GEMV/GEMM (62.5%)|FlashAttention XPU (11.5%)|Elementwise/residual XPU (8.1%)|473.117|940.923|8.000|
|tiny|layer|128/128|0|1024|tg|61.4946|Q4 GEMV/GEMM (52.6%)|FlashAttention XPU (12.8%)|SwiGLU CPU graph (6.1%)|548.866|992.688|8.000|
|tiny|layer|128/128|0|128|pp|3.6015|Q4 GEMV/GEMM (66.4%)|Elementwise/residual XPU (8.5%)|FlashAttention XPU (6.3%)|471.620|940.923|8.000|
|tiny|layer|128/128|0|128|tg|56.8909|Q4 GEMV/GEMM (58.4%)|Elementwise/residual XPU (6.5%)|SwiGLU CPU graph (6.0%)|544.866|992.688|8.000|
|tiny|layer|128/128|0|2048|pp|4.1303|Q4 GEMV/GEMM (58.1%)|FlashAttention XPU (17.8%)|Elementwise/residual XPU (7.5%)|475.117|940.923|8.000|
|tiny|layer|128/128|0|2048|tg|58.8756|Q4 GEMV/GEMM (54.3%)|FlashAttention XPU (11.2%)|SwiGLU CPU graph (6.2%)|552.875|992.688|8.000|
|tiny|layer|128/128|0|512|pp|3.6794|Q4 GEMV/GEMM (65.2%)|Elementwise/residual XPU (8.4%)|FlashAttention XPU (7.8%)|472.117|940.923|8.000|
|tiny|layer|128/128|0|512|tg|56.2181|Q4 GEMV/GEMM (58.6%)|SwiGLU CPU graph (6.3%)|FlashAttention XPU (6.1%)|546.866|992.688|8.000|
|tiny|layer|128/128|1|1024|pp|3.6720|Q4 GEMV/GEMM (65.0%)|FlashAttention XPU (12.0%)|Elementwise/residual XPU (8.2%)|10.945|16.579|8.000|
|tiny|layer|128/128|1|1024|tg|51.7947|Q4 GEMV/GEMM (60.8%)|FlashAttention XPU (10.3%)|Elementwise/residual XPU (6.2%)|64.866|24.688|8.000|
|tiny|layer|128/128|1|128|pp|3.4878|Q4 GEMV/GEMM (68.8%)|Elementwise/residual XPU (8.8%)|FlashAttention XPU (6.5%)|9.448|16.579|8.000|
|tiny|layer|128/128|1|128|tg|48.6942|Q4 GEMV/GEMM (64.4%)|Elementwise/residual XPU (6.6%)|KV SET_ROWS XPU (6.2%)|60.866|24.688|8.000|
|tiny|layer|128/128|1|2048|pp|3.9932|Q4 GEMV/GEMM (59.9%)|FlashAttention XPU (18.4%)|Elementwise/residual XPU (7.6%)|12.945|16.579|8.000|
|tiny|layer|128/128|1|2048|tg|53.0331|Q4 GEMV/GEMM (59.4%)|FlashAttention XPU (12.4%)|Elementwise/residual XPU (6.0%)|68.875|24.688|8.000|
|tiny|layer|128/128|1|512|pp|3.5338|Q4 GEMV/GEMM (67.9%)|Elementwise/residual XPU (8.5%)|FlashAttention XPU (8.1%)|9.945|16.579|8.000|
|tiny|layer|128/128|1|512|tg|51.1152|Q4 GEMV/GEMM (61.5%)|RoPE XPU (8.0%)|FlashAttention XPU (6.7%)|62.866|24.688|8.000|
|tiny|layer|512/512|0|1024|pp|4.4844|Q4 GEMV/GEMM (53.1%)|KV SET_ROWS XPU (18.3%)|Attention CPU graph (10.8%)|668.148|1171.446|8.000|
|tiny|layer|512/512|0|1024|tg|57.9179|Q4 GEMV/GEMM (55.4%)|FlashAttention XPU (9.3%)|SwiGLU CPU graph (6.3%)|548.789|992.688|8.000|
|tiny|layer|512/512|0|128|pp|3.6180|Q4 GEMV/GEMM (66.3%)|Elementwise/residual XPU (8.5%)|FlashAttention XPU (6.3%)|471.620|940.923|8.000|
|tiny|layer|512/512|0|128|tg|54.7450|Q4 GEMV/GEMM (59.1%)|SwiGLU CPU graph (6.4%)|Elementwise/residual XPU (6.0%)|544.866|992.688|8.000|
|tiny|layer|512/512|0|2048|pp|4.8870|Q4 GEMV/GEMM (48.7%)|Attention CPU graph (17.9%)|KV SET_ROWS XPU (16.8%)|668.148|1193.446|8.000|
|tiny|layer|512/512|0|2048|tg|58.0827|Q4 GEMV/GEMM (54.5%)|FlashAttention XPU (11.4%)|SwiGLU CPU graph (6.2%)|552.789|992.688|8.000|
|tiny|layer|512/512|0|512|pp|4.2849|Q4 GEMV/GEMM (55.6%)|KV SET_ROWS XPU (19.1%)|Attention CPU graph (6.6%)|668.148|1160.446|8.000|
|tiny|layer|512/512|0|512|tg|56.7196|Q4 GEMV/GEMM (58.3%)|SwiGLU CPU graph (6.2%)|FlashAttention XPU (6.1%)|546.789|992.688|8.000|
|tiny|layer|512/512|1|1024|pp|4.3632|Q4 GEMV/GEMM (54.6%)|KV SET_ROWS XPU (18.8%)|Attention CPU graph (11.1%)|206.105|247.360|8.000|
|tiny|layer|512/512|1|1024|tg|52.1848|Q4 GEMV/GEMM (60.5%)|FlashAttention XPU (10.3%)|Elementwise/residual XPU (6.2%)|64.789|24.688|8.000|
|tiny|layer|512/512|1|128|pp|3.4641|Q4 GEMV/GEMM (68.9%)|Elementwise/residual XPU (8.7%)|FlashAttention XPU (6.5%)|9.448|16.579|8.000|
|tiny|layer|512/512|1|128|tg|48.9333|Q4 GEMV/GEMM (64.3%)|Elementwise/residual XPU (6.6%)|KV SET_ROWS XPU (6.2%)|60.866|24.688|8.000|
|tiny|layer|512/512|1|2048|pp|4.7599|Q4 GEMV/GEMM (50.1%)|Attention CPU graph (18.4%)|KV SET_ROWS XPU (17.2%)|206.105|269.360|8.000|
|tiny|layer|512/512|1|2048|tg|55.9826|Q4 GEMV/GEMM (60.9%)|FlashAttention XPU (11.8%)|Elementwise/residual XPU (5.8%)|68.789|24.688|8.000|
|tiny|layer|512/512|1|512|pp|4.1787|Q4 GEMV/GEMM (57.0%)|KV SET_ROWS XPU (19.6%)|Attention CPU graph (7.1%)|206.105|236.360|8.000|
|tiny|layer|512/512|1|512|tg|51.1925|Q4 GEMV/GEMM (61.5%)|FlashAttention XPU (6.7%)|Elementwise/residual XPU (6.2%)|62.789|24.688|8.000|
|tiny|pd0|512/128|0|1024|pp|3.7975|Q4 GEMV/GEMM (63.1%)|FlashAttention XPU (11.1%)|Elementwise/residual XPU (8.1%)|471.624|940.712|0.000|
|tiny|pd0|512/128|0|1024|tg|61.6302|Q4 GEMV/GEMM (57.1%)|FlashAttention XPU (8.8%)|SwiGLU CPU graph (5.9%)|546.308|992.688|0.000|
|tiny|pd0|512/128|0|128|pp|3.6030|Q4 GEMV/GEMM (66.5%)|Elementwise/residual XPU (8.4%)|FlashAttention XPU (6.3%)|471.098|940.923|0.000|
|tiny|pd0|512/128|0|128|tg|55.8206|Q4 GEMV/GEMM (58.0%)|Elementwise/residual XPU (7.4%)|SwiGLU CPU graph (6.5%)|544.308|992.688|0.000|
|tiny|pd0|512/128|0|2048|pp|4.0769|Q4 GEMV/GEMM (58.7%)|FlashAttention XPU (17.1%)|Elementwise/residual XPU (7.5%)|472.624|940.712|0.000|
|tiny|pd0|512/128|0|2048|tg|58.5571|Q4 GEMV/GEMM (54.9%)|FlashAttention XPU (11.3%)|SwiGLU CPU graph (5.7%)|548.312|992.688|0.000|
|tiny|pd0|512/128|0|512|pp|3.6384|Q4 GEMV/GEMM (65.6%)|Elementwise/residual XPU (8.3%)|FlashAttention XPU (7.6%)|471.124|940.712|0.000|
|tiny|pd0|512/128|0|512|tg|56.5919|Q4 GEMV/GEMM (58.9%)|SwiGLU CPU graph (6.4%)|FlashAttention XPU (6.2%)|545.308|992.688|0.000|
|tiny|pd0|512/128|1|1024|pp|3.6730|Q4 GEMV/GEMM (65.3%)|FlashAttention XPU (11.4%)|Elementwise/residual XPU (8.4%)|9.581|16.626|0.000|
|tiny|pd0|512/128|1|1024|tg|52.0121|Q4 GEMV/GEMM (61.1%)|FlashAttention XPU (10.3%)|Elementwise/residual XPU (6.2%)|62.308|24.688|0.000|
|tiny|pd0|512/128|1|128|pp|3.4935|Q4 GEMV/GEMM (68.8%)|Elementwise/residual XPU (8.9%)|FlashAttention XPU (6.5%)|8.926|16.579|0.000|
|tiny|pd0|512/128|1|128|tg|49.5228|Q4 GEMV/GEMM (64.5%)|Elementwise/residual XPU (6.6%)|KV SET_ROWS XPU (6.1%)|60.308|24.688|0.000|
|tiny|pd0|512/128|1|2048|pp|3.9579|Q4 GEMV/GEMM (60.5%)|FlashAttention XPU (17.6%)|Elementwise/residual XPU (7.8%)|10.581|16.626|0.000|
|tiny|pd0|512/128|1|2048|tg|53.4847|Q4 GEMV/GEMM (59.6%)|FlashAttention XPU (12.4%)|Elementwise/residual XPU (6.0%)|64.312|24.688|0.000|
|tiny|pd0|512/128|1|512|pp|3.5266|Q4 GEMV/GEMM (68.1%)|Elementwise/residual XPU (8.6%)|FlashAttention XPU (7.8%)|9.081|16.626|0.000|
|tiny|pd0|512/128|1|512|tg|51.5633|Q4 GEMV/GEMM (64.2%)|FlashAttention XPU (6.7%)|Elementwise/residual XPU (6.3%)|61.308|24.688|0.000|
|tiny|pd0|128/128|0|1024|pp|3.7392|Q4 GEMV/GEMM (63.4%)|FlashAttention XPU (11.2%)|Elementwise/residual XPU (7.8%)|471.846|940.923|0.000|
|tiny|pd0|128/128|0|1024|tg|63.6002|Q4 GEMV/GEMM (56.2%)|FlashAttention XPU (9.1%)|SwiGLU CPU graph (6.5%)|546.308|992.688|0.000|
|tiny|pd0|128/128|0|128|pp|3.5220|Q4 GEMV/GEMM (67.0%)|Elementwise/residual XPU (8.1%)|FlashAttention XPU (6.4%)|471.098|940.923|0.000|
|tiny|pd0|128/128|0|128|tg|59.4977|Q4 GEMV/GEMM (56.7%)|Elementwise/residual XPU (7.7%)|SwiGLU CPU graph (6.8%)|544.308|992.688|0.000|
|tiny|pd0|128/128|0|2048|pp|4.0617|Q4 GEMV/GEMM (58.8%)|FlashAttention XPU (17.2%)|Elementwise/residual XPU (7.4%)|472.846|940.923|0.000|
|tiny|pd0|128/128|0|2048|tg|62.2348|Q4 GEMV/GEMM (53.4%)|FlashAttention XPU (11.0%)|SwiGLU CPU graph (6.3%)|548.312|992.688|0.000|
|tiny|pd0|128/128|0|512|pp|3.4836|Q4 GEMV/GEMM (66.6%)|FlashAttention XPU (7.8%)|Elementwise/residual XPU (7.6%)|471.347|940.923|0.000|
|tiny|pd0|128/128|0|512|tg|58.9495|Q4 GEMV/GEMM (55.3%)|RoPE XPU (7.3%)|SwiGLU CPU graph (7.0%)|545.308|992.688|0.000|
|tiny|pd0|128/128|1|1024|pp|3.5290|Q4 GEMV/GEMM (66.2%)|FlashAttention XPU (11.8%)|Elementwise/residual XPU (7.8%)|9.675|16.579|0.000|
|tiny|pd0|128/128|1|1024|tg|55.3443|Q4 GEMV/GEMM (60.0%)|FlashAttention XPU (10.2%)|Elementwise/residual XPU (6.9%)|62.308|24.688|0.000|
|tiny|pd0|128/128|1|128|pp|3.3101|Q4 GEMV/GEMM (70.2%)|Elementwise/residual XPU (8.1%)|FlashAttention XPU (6.7%)|8.926|16.579|0.000|
|tiny|pd0|128/128|1|128|tg|52.9125|Q4 GEMV/GEMM (63.0%)|Elementwise/residual XPU (7.3%)|KV SET_ROWS XPU (6.1%)|60.308|24.688|0.000|
|tiny|pd0|128/128|1|2048|pp|3.8168|Q4 GEMV/GEMM (61.2%)|FlashAttention XPU (18.2%)|Elementwise/residual XPU (7.2%)|10.674|16.579|0.000|
|tiny|pd0|128/128|1|2048|tg|53.8163|Q4 GEMV/GEMM (59.3%)|FlashAttention XPU (12.3%)|Elementwise/residual XPU (6.2%)|64.312|24.688|0.000|
|tiny|pd0|128/128|1|512|pp|3.5475|Q4 GEMV/GEMM (68.0%)|Elementwise/residual XPU (8.7%)|FlashAttention XPU (7.8%)|9.175|16.579|0.000|
|tiny|pd0|128/128|1|512|tg|51.2087|Q4 GEMV/GEMM (64.2%)|FlashAttention XPU (6.8%)|Elementwise/residual XPU (6.2%)|61.308|24.688|0.000|
|tiny|pd0|512/512|0|1024|pp|4.4739|Q4 GEMV/GEMM (53.3%)|KV SET_ROWS XPU (18.3%)|Attention CPU graph (10.5%)|668.133|1171.446|0.000|
|tiny|pd0|512/512|0|1024|tg|56.7176|Q4 GEMV/GEMM (56.0%)|FlashAttention XPU (9.5%)|SwiGLU CPU graph (6.2%)|546.270|992.688|0.000|
|tiny|pd0|512/512|0|128|pp|3.6284|Q4 GEMV/GEMM (66.3%)|Elementwise/residual XPU (8.6%)|FlashAttention XPU (6.3%)|471.098|940.923|0.000|
|tiny|pd0|512/512|0|128|tg|55.1533|Q4 GEMV/GEMM (58.2%)|Elementwise/residual XPU (7.2%)|SwiGLU CPU graph (6.3%)|544.308|992.688|0.000|
|tiny|pd0|512/512|0|2048|pp|4.8666|Q4 GEMV/GEMM (48.9%)|Attention CPU graph (17.5%)|KV SET_ROWS XPU (16.8%)|668.133|1193.446|0.000|
|tiny|pd0|512/512|0|2048|tg|58.4666|Q4 GEMV/GEMM (54.6%)|FlashAttention XPU (11.3%)|SwiGLU CPU graph (6.0%)|548.270|992.688|0.000|
|tiny|pd0|512/512|0|512|pp|4.2799|Q4 GEMV/GEMM (55.8%)|KV SET_ROWS XPU (19.2%)|Attention CPU graph (6.4%)|668.133|1160.446|0.000|
|tiny|pd0|512/512|0|512|tg|56.2033|Q4 GEMV/GEMM (58.9%)|SwiGLU CPU graph (6.3%)|FlashAttention XPU (6.2%)|545.270|992.688|0.000|
|tiny|pd0|512/512|1|1024|pp|4.3633|Q4 GEMV/GEMM (54.7%)|KV SET_ROWS XPU (18.8%)|Attention CPU graph (10.8%)|206.090|247.360|0.000|
|tiny|pd0|512/512|1|1024|tg|52.1847|Q4 GEMV/GEMM (60.8%)|FlashAttention XPU (10.3%)|Elementwise/residual XPU (6.2%)|62.270|24.688|0.000|
|tiny|pd0|512/512|1|128|pp|3.4717|Q4 GEMV/GEMM (68.9%)|Elementwise/residual XPU (8.8%)|FlashAttention XPU (6.5%)|8.926|16.579|0.000|
|tiny|pd0|512/512|1|128|tg|49.5120|Q4 GEMV/GEMM (64.3%)|Elementwise/residual XPU (6.5%)|KV SET_ROWS XPU (6.2%)|60.308|24.688|0.000|
|tiny|pd0|512/512|1|2048|pp|4.7228|Q4 GEMV/GEMM (50.1%)|Attention CPU graph (18.3%)|KV SET_ROWS XPU (17.3%)|206.090|269.360|0.000|
|tiny|pd0|512/512|1|2048|tg|57.2133|Q4 GEMV/GEMM (56.5%)|FlashAttention XPU (11.8%)|Elementwise/residual XPU (6.1%)|64.270|24.688|0.000|
|tiny|pd0|512/512|1|512|pp|4.1754|Q4 GEMV/GEMM (57.1%)|KV SET_ROWS XPU (19.6%)|Attention CPU graph (6.9%)|206.090|236.360|0.000|
|tiny|pd0|512/512|1|512|tg|50.1162|Q4 GEMV/GEMM (63.2%)|FlashAttention XPU (6.9%)|Elementwise/residual XPU (6.4%)|61.270|24.688|0.000|
|tiny|pd1|128/128|0|1024|pp|3.7894|Q4 GEMV/GEMM (63.1%)|FlashAttention XPU (11.1%)|Elementwise/residual XPU (8.0%)|471.846|940.923|0.000|
|tiny|pd1|128/128|0|1024|tg|59.0054|Q4 GEMV/GEMM (57.7%)|FlashAttention XPU (9.0%)|SwiGLU CPU graph (6.2%)|546.308|992.688|0.000|
|tiny|pd1|128/128|0|128|pp|3.6121|Q4 GEMV/GEMM (66.4%)|Elementwise/residual XPU (8.5%)|FlashAttention XPU (6.2%)|471.098|940.923|0.000|
|tiny|pd1|128/128|0|128|tg|54.5021|Q4 GEMV/GEMM (58.0%)|Elementwise/residual XPU (7.2%)|SwiGLU CPU graph (6.4%)|544.308|992.688|0.000|
|tiny|pd1|128/128|0|2048|pp|4.0570|Q4 GEMV/GEMM (58.8%)|FlashAttention XPU (17.1%)|Elementwise/residual XPU (7.4%)|472.846|940.923|0.000|
|tiny|pd1|128/128|0|2048|tg|57.5755|Q4 GEMV/GEMM (54.6%)|FlashAttention XPU (11.4%)|SwiGLU CPU graph (6.2%)|548.312|992.688|0.000|
|tiny|pd1|128/128|0|512|pp|3.6323|Q4 GEMV/GEMM (65.6%)|Elementwise/residual XPU (8.3%)|FlashAttention XPU (7.6%)|471.347|940.923|0.000|
|tiny|pd1|128/128|0|512|tg|56.0850|Q4 GEMV/GEMM (56.3%)|RoPE XPU (7.5%)|SwiGLU CPU graph (6.4%)|545.308|992.688|0.000|
|tiny|pd1|128/128|1|1024|pp|3.6560|Q4 GEMV/GEMM (65.3%)|FlashAttention XPU (11.4%)|Elementwise/residual XPU (8.3%)|9.675|16.579|0.000|
|tiny|pd1|128/128|1|1024|tg|51.2587|Q4 GEMV/GEMM (60.8%)|FlashAttention XPU (10.4%)|Elementwise/residual XPU (6.2%)|62.308|24.688|0.000|
|tiny|pd1|128/128|1|128|pp|3.4811|Q4 GEMV/GEMM (68.8%)|Elementwise/residual XPU (8.8%)|FlashAttention XPU (6.5%)|8.926|16.579|0.000|
|tiny|pd1|128/128|1|128|tg|48.8164|Q4 GEMV/GEMM (64.2%)|Elementwise/residual XPU (6.6%)|KV SET_ROWS XPU (6.2%)|60.308|24.688|0.000|
|tiny|pd1|128/128|1|2048|pp|3.9464|Q4 GEMV/GEMM (60.6%)|FlashAttention XPU (17.6%)|Elementwise/residual XPU (7.7%)|10.674|16.579|0.000|
|tiny|pd1|128/128|1|2048|tg|52.9293|Q4 GEMV/GEMM (59.2%)|FlashAttention XPU (12.4%)|Elementwise/residual XPU (6.0%)|64.312|24.688|0.000|
|tiny|pd1|128/128|1|512|pp|3.5309|Q4 GEMV/GEMM (68.1%)|Elementwise/residual XPU (8.6%)|FlashAttention XPU (7.8%)|9.175|16.579|0.000|
|tiny|pd1|128/128|1|512|tg|50.8036|Q4 GEMV/GEMM (64.0%)|FlashAttention XPU (6.7%)|Elementwise/residual XPU (6.3%)|61.308|24.688|0.000|
|tiny|pd1|512/512|0|1024|pp|4.4529|Q4 GEMV/GEMM (53.3%)|KV SET_ROWS XPU (18.4%)|Attention CPU graph (10.5%)|668.133|1171.446|0.000|
|tiny|pd1|512/512|0|1024|tg|56.5560|Q4 GEMV/GEMM (55.7%)|FlashAttention XPU (9.4%)|SwiGLU CPU graph (6.2%)|546.270|992.688|0.000|
|tiny|pd1|512/512|0|128|pp|3.6089|Q4 GEMV/GEMM (66.4%)|Elementwise/residual XPU (8.5%)|FlashAttention XPU (6.3%)|471.098|940.923|0.000|
|tiny|pd1|512/512|0|128|tg|59.1371|Q4 GEMV/GEMM (56.6%)|Elementwise/residual XPU (7.7%)|SwiGLU CPU graph (6.5%)|544.308|992.688|0.000|
|tiny|pd1|512/512|0|2048|pp|4.7816|Q4 GEMV/GEMM (48.2%)|Attention CPU graph (18.0%)|KV SET_ROWS XPU (17.1%)|668.133|1193.446|0.000|
|tiny|pd1|512/512|0|2048|tg|57.6686|Q4 GEMV/GEMM (54.6%)|FlashAttention XPU (11.4%)|SwiGLU CPU graph (6.1%)|548.270|992.688|0.000|
|tiny|pd1|512/512|0|512|pp|4.2719|Q4 GEMV/GEMM (55.6%)|KV SET_ROWS XPU (19.2%)|Attention CPU graph (6.7%)|668.133|1160.446|0.000|
|tiny|pd1|512/512|0|512|tg|55.7916|Q4 GEMV/GEMM (58.7%)|FlashAttention XPU (6.1%)|SwiGLU CPU graph (6.1%)|545.270|992.688|0.000|
|tiny|pd1|512/512|1|1024|pp|4.3408|Q4 GEMV/GEMM (54.8%)|KV SET_ROWS XPU (18.9%)|Attention CPU graph (10.7%)|206.090|247.360|0.000|
|tiny|pd1|512/512|1|1024|tg|51.6004|Q4 GEMV/GEMM (60.7%)|FlashAttention XPU (10.3%)|Elementwise/residual XPU (6.2%)|62.270|24.688|0.000|
|tiny|pd1|512/512|1|128|pp|3.4503|Q4 GEMV/GEMM (69.0%)|Elementwise/residual XPU (8.7%)|FlashAttention XPU (6.5%)|8.926|16.579|0.000|
|tiny|pd1|512/512|1|128|tg|48.9126|Q4 GEMV/GEMM (64.1%)|Elementwise/residual XPU (6.6%)|KV SET_ROWS XPU (6.2%)|60.308|24.688|0.000|
|tiny|pd1|512/512|1|2048|pp|4.7351|Q4 GEMV/GEMM (50.2%)|Attention CPU graph (18.2%)|KV SET_ROWS XPU (17.3%)|206.090|269.360|0.000|
|tiny|pd1|512/512|1|2048|tg|55.6528|Q4 GEMV/GEMM (56.4%)|FlashAttention XPU (11.8%)|Remaining host/scheduler/measurement (6.0%)|64.270|24.688|0.000|
|tiny|pd1|512/512|1|512|pp|4.1519|Q4 GEMV/GEMM (57.1%)|KV SET_ROWS XPU (19.7%)|Attention CPU graph (6.9%)|206.090|236.360|0.000|
|tiny|pd1|512/512|1|512|tg|49.6302|Q4 GEMV/GEMM (63.0%)|FlashAttention XPU (6.9%)|Elementwise/residual XPU (6.4%)|61.270|24.688|0.000|
|tiny|row|128/128|0|1024|pp|6.1950|Q4 GEMV/GEMM (75.6%)|FlashAttention XPU (6.8%)|External D2H (3.8%)|817.179|1287.798|2254.500|
|tiny|row|128/128|0|1024|tg|99.8741|Q4 GEMV/GEMM (64.7%)|Other CPU graph (7.1%)|FlashAttention XPU (5.5%)|900.866|1704.688|2318.000|
|tiny|row|128/128|0|128|pp|5.9882|Q4 GEMV/GEMM (78.1%)|External D2H (3.9%)|FlashAttention XPU (3.8%)|815.683|1287.798|2254.500|
|tiny|row|128/128|0|128|tg|91.6475|Q4 GEMV/GEMM (67.2%)|Other CPU graph (7.0%)|SwiGLU CPU graph (3.6%)|896.866|1704.688|2318.000|
|tiny|row|128/128|0|2048|pp|6.4958|Q4 GEMV/GEMM (72.5%)|FlashAttention XPU (10.7%)|External D2H (3.6%)|819.179|1287.798|2254.500|
|tiny|row|128/128|0|2048|tg|96.8637|Q4 GEMV/GEMM (63.7%)|Other CPU graph (7.0%)|FlashAttention XPU (6.8%)|904.875|1704.688|2318.000|
|tiny|row|128/128|0|512|pp|6.0733|Q4 GEMV/GEMM (77.3%)|FlashAttention XPU (4.9%)|External D2H (3.9%)|816.180|1287.798|2254.500|
|tiny|row|128/128|0|512|tg|98.5053|Q4 GEMV/GEMM (64.2%)|Other CPU graph (6.3%)|RoPE XPU (5.6%)|898.866|1704.688|2318.000|
|tiny|row|128/128|1|1024|pp|6.0718|Q4 GEMV/GEMM (77.2%)|FlashAttention XPU (6.9%)|SwiGLU XPU (2.8%)|355.007|363.454|2254.500|
|tiny|row|128/128|1|1024|tg|94.2703|Q4 GEMV/GEMM (67.1%)|Other CPU graph (7.4%)|FlashAttention XPU (5.7%)|416.866|736.688|2318.000|
|tiny|row|128/128|1|128|pp|5.8321|Q4 GEMV/GEMM (80.1%)|FlashAttention XPU (3.9%)|SwiGLU XPU (2.9%)|353.511|363.454|2254.500|
|tiny|row|128/128|1|128|tg|88.9606|Q4 GEMV/GEMM (69.6%)|Other CPU graph (7.7%)|KV SET_ROWS XPU (3.5%)|412.866|736.688|2318.000|
|tiny|row|128/128|1|2048|pp|6.3485|Q4 GEMV/GEMM (73.9%)|FlashAttention XPU (11.0%)|SwiGLU XPU (2.7%)|357.007|363.454|2254.500|
|tiny|row|128/128|1|2048|tg|95.6489|Q4 GEMV/GEMM (66.6%)|Other CPU graph (7.4%)|FlashAttention XPU (7.0%)|420.875|736.688|2318.000|
|tiny|row|128/128|1|512|pp|5.9232|Q4 GEMV/GEMM (79.1%)|FlashAttention XPU (4.7%)|SwiGLU XPU (2.9%)|354.008|363.454|2254.500|
|tiny|row|128/128|1|512|tg|88.9017|Q4 GEMV/GEMM (69.2%)|Other CPU graph (7.5%)|FlashAttention XPU (3.9%)|414.866|736.688|2318.000|
|tiny|row|512/512|0|1024|pp|6.5273|Q4 GEMV/GEMM (67.6%)|KV SET_ROWS XPU (12.6%)|Attention CPU graph (7.4%)|1012.164|1516.164|2254.125|
|tiny|row|512/512|0|1024|tg|94.7796|Q4 GEMV/GEMM (64.8%)|Other CPU graph (7.1%)|FlashAttention XPU (5.7%)|900.789|1704.688|2318.000|
|tiny|row|512/512|0|128|pp|5.9861|Q4 GEMV/GEMM (78.1%)|External D2H (3.9%)|FlashAttention XPU (3.8%)|815.683|1287.798|2254.500|
|tiny|row|512/512|0|128|tg|92.2138|Q4 GEMV/GEMM (67.2%)|Other CPU graph (6.9%)|SwiGLU CPU graph (3.6%)|896.866|1704.688|2318.000|
|tiny|row|512/512|0|2048|pp|6.9253|Q4 GEMV/GEMM (63.5%)|Attention CPU graph (12.9%)|KV SET_ROWS XPU (11.8%)|1012.164|1538.164|2254.125|
|tiny|row|512/512|0|2048|tg|100.0972|Q4 GEMV/GEMM (66.4%)|FlashAttention XPU (6.6%)|Other CPU graph (5.7%)|904.789|1704.688|2318.000|
|tiny|row|512/512|0|512|pp|6.3305|Q4 GEMV/GEMM (69.7%)|KV SET_ROWS XPU (13.0%)|Attention CPU graph (4.6%)|1012.164|1505.164|2254.125|
|tiny|row|512/512|0|512|tg|96.5480|Q4 GEMV/GEMM (66.5%)|Other CPU graph (6.7%)|Remaining host/scheduler/measurement (3.9%)|898.789|1704.688|2318.000|
|tiny|row|512/512|1|1024|pp|6.4124|Q4 GEMV/GEMM (68.8%)|KV SET_ROWS XPU (12.8%)|Attention CPU graph (7.8%)|550.121|592.078|2254.125|
|tiny|row|512/512|1|1024|tg|90.3310|Q4 GEMV/GEMM (67.6%)|Other CPU graph (7.8%)|FlashAttention XPU (5.9%)|416.789|736.688|2318.000|
|tiny|row|512/512|1|128|pp|5.8565|Q4 GEMV/GEMM (80.0%)|FlashAttention XPU (3.9%)|SwiGLU XPU (2.9%)|353.511|363.454|2254.500|
|tiny|row|512/512|1|128|tg|88.5896|Q4 GEMV/GEMM (69.7%)|Other CPU graph (7.7%)|KV SET_ROWS XPU (3.5%)|412.866|736.688|2318.000|
|tiny|row|512/512|1|2048|pp|6.7544|Q4 GEMV/GEMM (65.0%)|Attention CPU graph (12.7%)|KV SET_ROWS XPU (12.1%)|550.121|614.078|2254.125|
|tiny|row|512/512|1|2048|tg|97.0179|Q4 GEMV/GEMM (68.3%)|Other CPU graph (7.4%)|FlashAttention XPU (6.8%)|420.789|736.688|2318.000|
|tiny|row|512/512|1|512|pp|6.2358|Q4 GEMV/GEMM (70.5%)|KV SET_ROWS XPU (13.2%)|Attention CPU graph (5.1%)|550.121|581.078|2254.125|
|tiny|row|512/512|1|512|tg|88.5875|Q4 GEMV/GEMM (69.0%)|Other CPU graph (8.1%)|FlashAttention XPU (3.9%)|414.789|736.688|2318.000|


## 分離kernel診断

起動後にstream待機を追加した参考診断。通常0.7.6の並列性を保った性能値ではない。以下を通常wall内訳へ加算しない。

|model|GLU|prefix|phase|kernel分類|回/単位|enqueue→完了 ms/単位|
|---|---|---|---|---|---|---|
|7b|0|128|pp|Other elementwise|1.50|1.04974|
|7b|0|128|pp|Device copy_rows|1.02|0.11974|
|7b|0|128|pp|Q4 Activation quantization|1.80|2.65051|
|7b|0|128|pp|Q4 GEMM MAC+block scales+DMA|1.73|7.48381|
|7b|0|128|pp|RoPE|1.00|0.33714|
|7b|0|128|pp|Other kernel|0.50|1.07262|
|7b|0|128|pp|Attention pack/copy|0.25|0.02647|
|7b|0|128|pp|SDK cast/findmax (unmapped)|0.75|0.04836|
|7b|0|128|pp|Flash Q ranges|0.25|0.01412|
|7b|0|128|pp|Flash KQ+DMA|0.25|0.05558|
|7b|0|128|pp|Flash Online/normalize/reduce|0.50|0.28639|
|7b|0|128|pp|Flash PV+DMA|0.25|0.05035|
|7b|0|128|pp|Q4 GEMM slice join|0.24|0.08105|
|7b|0|128|pp|RMSNorm|0.03|0.00088|
|7b|0|128|pp|Q4 GEMV MAC+DMA|0.02|0.01364|
|7b|0|128|pp|Q4 GEMV FP32 scale/sum|0.02|0.00230|
|7b|0|128|tg|RMSNorm|130.00|1.50650|
|7b|0|128|tg|Device copy_rows|130.00|1.09349|
|7b|0|128|tg|Other elementwise|129.00|2.25356|
|7b|0|128|tg|Q4 Activation quantization|672.00|18.09681|
|7b|0|128|tg|Q4 GEMV MAC+DMA|224.00|82.83703|
|7b|0|128|tg|Q4 GEMV FP32 scale/sum|224.00|10.47039|
|7b|0|128|tg|RoPE|128.00|3.05311|
|7b|0|128|tg|Other kernel|64.00|1.80455|
|7b|0|128|tg|SDK cast/findmax (unmapped)|96.00|2.10275|
|7b|0|128|tg|Flash Q ranges|32.00|0.49996|
|7b|0|128|tg|Flash KQ+DMA|32.00|3.44653|
|7b|0|128|tg|Flash Online/normalize/reduce|64.00|1.40221|
|7b|0|128|tg|Flash PV+DMA|32.00|3.56215|
|7b|0|2048|pp|Other elementwise|1.50|1.04918|
|7b|0|2048|pp|Device copy_rows|1.02|0.11922|
|7b|0|2048|pp|Q4 Activation quantization|1.80|2.65007|
|7b|0|2048|pp|Q4 GEMM MAC+block scales+DMA|1.73|7.48691|
|7b|0|2048|pp|RoPE|1.00|0.33781|
|7b|0|2048|pp|Other kernel|0.50|1.07330|
|7b|0|2048|pp|Attention pack/copy|0.25|0.02636|
|7b|0|2048|pp|SDK cast/findmax (unmapped)|0.75|0.08723|
|7b|0|2048|pp|Flash Q ranges|0.25|0.01445|
|7b|0|2048|pp|Flash KQ+DMA|0.62|0.21666|
|7b|0|2048|pp|Flash Online/normalize/reduce|0.88|0.86454|
|7b|0|2048|pp|Flash PV+DMA|0.62|0.18611|
|7b|0|2048|pp|Q4 GEMM slice join|0.24|0.08105|
|7b|0|2048|pp|RMSNorm|0.03|0.00097|
|7b|0|2048|pp|Q4 GEMV MAC+DMA|0.02|0.01301|
|7b|0|2048|pp|Q4 GEMV FP32 scale/sum|0.02|0.00212|
|7b|0|2048|tg|RMSNorm|130.00|1.48452|
|7b|0|2048|tg|Device copy_rows|130.00|1.07416|
|7b|0|2048|tg|Other elementwise|129.00|2.24569|
|7b|0|2048|tg|Q4 Activation quantization|672.00|18.00586|
|7b|0|2048|tg|Q4 GEMV MAC+DMA|224.00|82.76908|
|7b|0|2048|tg|Q4 GEMV FP32 scale/sum|224.00|10.59712|
|7b|0|2048|tg|RoPE|128.00|3.05988|
|7b|0|2048|tg|Other kernel|96.00|2.91536|
|7b|0|2048|tg|SDK cast/findmax (unmapped)|48.00|3.35623|
|7b|0|2048|tg|Flash Q ranges|32.00|0.51178|
|7b|0|2048|tg|Flash KQ+DMA|96.00|11.61133|
|7b|0|2048|tg|Flash Online/normalize/reduce|128.00|6.87302|
|7b|0|2048|tg|Flash PV+DMA|96.00|13.64334|
|7b|1|128|pp|Other elementwise|1.50|1.05380|
|7b|1|128|pp|Device copy_rows|1.02|0.12091|
|7b|1|128|pp|Q4 Activation quantization|1.80|2.64988|
|7b|1|128|pp|Q4 GEMM MAC+block scales+DMA|1.73|7.48757|
|7b|1|128|pp|RoPE|1.00|0.33826|
|7b|1|128|pp|Other kernel|0.50|1.07347|
|7b|1|128|pp|Attention pack/copy|0.25|0.02654|
|7b|1|128|pp|SDK cast/findmax (unmapped)|0.75|0.04924|
|7b|1|128|pp|Flash Q ranges|0.25|0.01469|
|7b|1|128|pp|Flash KQ+DMA|0.25|0.05567|
|7b|1|128|pp|Flash Online/normalize/reduce|0.50|0.28627|
|7b|1|128|pp|Flash PV+DMA|0.25|0.05035|
|7b|1|128|pp|SwiGLU|0.25|0.47325|
|7b|1|128|pp|Q4 GEMM slice join|0.24|0.08036|
|7b|1|128|pp|RMSNorm|0.03|0.00089|
|7b|1|128|pp|Q4 GEMV MAC+DMA|0.02|0.01350|
|7b|1|128|pp|Q4 GEMV FP32 scale/sum|0.02|0.00233|
|7b|1|128|tg|RMSNorm|130.00|1.43895|
|7b|1|128|tg|Device copy_rows|130.00|1.04086|
|7b|1|128|tg|Other elementwise|129.00|2.21452|
|7b|1|128|tg|Q4 Activation quantization|672.00|17.87633|
|7b|1|128|tg|Q4 GEMV MAC+DMA|224.00|82.75449|
|7b|1|128|tg|Q4 GEMV FP32 scale/sum|224.00|10.52927|
|7b|1|128|tg|RoPE|128.00|3.00329|
|7b|1|128|tg|Other kernel|64.00|1.82803|
|7b|1|128|tg|SDK cast/findmax (unmapped)|96.00|2.14190|
|7b|1|128|tg|Flash Q ranges|32.00|0.51555|
|7b|1|128|tg|Flash KQ+DMA|32.00|3.45038|
|7b|1|128|tg|Flash Online/normalize/reduce|64.00|1.40309|
|7b|1|128|tg|Flash PV+DMA|32.00|3.56377|
|7b|1|128|tg|SwiGLU|32.00|0.95435|
|7b|1|2048|pp|Other elementwise|1.50|1.05101|
|7b|1|2048|pp|Device copy_rows|1.02|0.11854|
|7b|1|2048|pp|Q4 Activation quantization|1.80|2.64736|
|7b|1|2048|pp|Q4 GEMM MAC+block scales+DMA|1.73|7.48419|
|7b|1|2048|pp|RoPE|1.00|0.33722|
|7b|1|2048|pp|Other kernel|0.50|1.07278|
|7b|1|2048|pp|Attention pack/copy|0.25|0.02605|
|7b|1|2048|pp|SDK cast/findmax (unmapped)|0.75|0.08640|
|7b|1|2048|pp|Flash Q ranges|0.25|0.01408|
|7b|1|2048|pp|Flash KQ+DMA|0.62|0.21564|
|7b|1|2048|pp|Flash Online/normalize/reduce|0.88|0.86339|
|7b|1|2048|pp|Flash PV+DMA|0.62|0.18504|
|7b|1|2048|pp|SwiGLU|0.25|0.47276|
|7b|1|2048|pp|Q4 GEMM slice join|0.24|0.08005|
|7b|1|2048|pp|RMSNorm|0.03|0.00094|
|7b|1|2048|pp|Q4 GEMV MAC+DMA|0.02|0.01298|
|7b|1|2048|pp|Q4 GEMV FP32 scale/sum|0.02|0.00204|
|7b|1|2048|tg|RMSNorm|130.00|1.49845|
|7b|1|2048|tg|Device copy_rows|130.00|1.07558|
|7b|1|2048|tg|Other elementwise|129.00|2.25322|
|7b|1|2048|tg|Q4 Activation quantization|672.00|17.97514|
|7b|1|2048|tg|Q4 GEMV MAC+DMA|224.00|82.80795|
|7b|1|2048|tg|Q4 GEMV FP32 scale/sum|224.00|10.51576|
|7b|1|2048|tg|RoPE|128.00|3.04643|
|7b|1|2048|tg|Other kernel|96.00|2.88527|
|7b|1|2048|tg|SDK cast/findmax (unmapped)|48.00|3.34095|
|7b|1|2048|tg|Flash Q ranges|32.00|0.50282|
|7b|1|2048|tg|Flash KQ+DMA|96.00|11.59144|
|7b|1|2048|tg|Flash Online/normalize/reduce|128.00|6.88681|
|7b|1|2048|tg|Flash PV+DMA|96.00|13.65902|
|7b|1|2048|tg|SwiGLU|32.00|0.96281|
|tiny|0|128|pp|Other elementwise|1.03|0.37414|
|tiny|0|128|pp|Device copy_rows|0.70|0.05110|
|tiny|0|128|pp|Q4 Activation quantization|1.25|0.92920|
|tiny|0|128|pp|Q4 GEMM MAC+block scales+DMA|1.18|1.39966|
|tiny|0|128|pp|RoPE|0.69|0.09348|
|tiny|0|128|pp|SET_ROWS|0.34|0.06324|
|tiny|0|128|pp|Attention pack/copy|0.17|0.01558|
|tiny|0|128|pp|SDK cast/findmax (unmapped)|0.52|0.02553|
|tiny|0|128|pp|Flash Q ranges|0.17|0.00851|
|tiny|0|128|pp|Flash KQ+DMA|0.17|0.02328|
|tiny|0|128|pp|Flash Online/normalize/reduce|0.34|0.18468|
|tiny|0|128|pp|Flash PV+DMA|0.17|0.01626|
|tiny|0|128|pp|Q4 GEMM slice join|0.16|0.03108|
|tiny|0|128|pp|RMSNorm|0.03|0.00081|
|tiny|0|128|pp|Q4 GEMV MAC+DMA|0.02|0.00421|
|tiny|0|128|pp|Q4 GEMV FP32 scale/sum|0.02|0.00074|
|tiny|0|128|tg|RMSNorm|90.00|0.96128|
|tiny|0|128|tg|Device copy_rows|90.00|0.69500|
|tiny|0|128|tg|Other elementwise|89.00|1.51125|
|tiny|0|128|tg|Q4 Activation quantization|462.00|10.72800|
|tiny|0|128|tg|Q4 GEMV MAC+DMA|154.00|16.92695|
|tiny|0|128|tg|Q4 GEMV FP32 scale/sum|154.00|2.76491|
|tiny|0|128|tg|RoPE|88.00|1.45600|
|tiny|0|128|tg|SET_ROWS|44.00|1.17407|
|tiny|0|128|tg|SDK cast/findmax (unmapped)|66.00|0.58278|
|tiny|0|128|tg|Flash Q ranges|22.00|0.34453|
|tiny|0|128|tg|Flash KQ+DMA|22.00|0.48978|
|tiny|0|128|tg|Flash Online/normalize/reduce|44.00|0.96411|
|tiny|0|128|tg|Flash PV+DMA|22.00|0.33802|
|tiny|0|2048|pp|Other elementwise|1.03|0.36233|
|tiny|0|2048|pp|Device copy_rows|0.70|0.04226|
|tiny|0|2048|pp|Q4 Activation quantization|1.25|0.91589|
|tiny|0|2048|pp|Q4 GEMM MAC+block scales+DMA|1.18|1.38571|
|tiny|0|2048|pp|RoPE|0.69|0.08513|
|tiny|0|2048|pp|SET_ROWS|0.34|0.05902|
|tiny|0|2048|pp|Attention pack/copy|0.17|0.01577|
|tiny|0|2048|pp|SDK cast/findmax (unmapped)|0.52|0.01835|
|tiny|0|2048|pp|Flash Q ranges|0.17|0.00621|
|tiny|0|2048|pp|Flash KQ+DMA|0.43|0.07944|
|tiny|0|2048|pp|Flash Online/normalize/reduce|0.60|0.57052|
|tiny|0|2048|pp|Flash PV+DMA|0.43|0.04581|
|tiny|0|2048|pp|Q4 GEMM slice join|0.16|0.02904|
|tiny|0|2048|pp|RMSNorm|0.03|0.00063|
|tiny|0|2048|pp|Q4 GEMV MAC+DMA|0.02|0.00413|
|tiny|0|2048|pp|Q4 GEMV FP32 scale/sum|0.02|0.00069|
|tiny|0|2048|tg|RMSNorm|90.00|1.14085|
|tiny|0|2048|tg|Device copy_rows|90.00|0.89297|
|tiny|0|2048|tg|Other elementwise|89.00|1.69115|
|tiny|0|2048|tg|Q4 Activation quantization|462.00|11.66264|
|tiny|0|2048|tg|Q4 GEMV MAC+DMA|154.00|17.25743|
|tiny|0|2048|tg|Q4 GEMV FP32 scale/sum|154.00|3.08703|
|tiny|0|2048|tg|RoPE|88.00|1.63346|
|tiny|0|2048|tg|SET_ROWS|44.00|1.26720|
|tiny|0|2048|tg|SDK cast/findmax (unmapped)|66.00|1.17974|
|tiny|0|2048|tg|Flash Q ranges|22.00|0.38700|
|tiny|0|2048|tg|Flash KQ+DMA|22.00|1.97918|
|tiny|0|2048|tg|Flash Online/normalize/reduce|44.00|2.56775|
|tiny|0|2048|tg|Flash PV+DMA|22.00|1.47418|
|tiny|1|128|pp|Other elementwise|1.03|0.36862|
|tiny|1|128|pp|Device copy_rows|0.70|0.04579|
|tiny|1|128|pp|Q4 Activation quantization|1.25|0.92351|
|tiny|1|128|pp|Q4 GEMM MAC+block scales+DMA|1.18|1.39235|
|tiny|1|128|pp|RoPE|0.69|0.08932|
|tiny|1|128|pp|SET_ROWS|0.34|0.06137|
|tiny|1|128|pp|Attention pack/copy|0.17|0.01434|
|tiny|1|128|pp|SDK cast/findmax (unmapped)|0.52|0.02437|
|tiny|1|128|pp|Flash Q ranges|0.17|0.00876|
|tiny|1|128|pp|Flash KQ+DMA|0.17|0.02219|
|tiny|1|128|pp|Flash Online/normalize/reduce|0.34|0.18203|
|tiny|1|128|pp|Flash PV+DMA|0.17|0.01531|
|tiny|1|128|pp|SwiGLU|0.17|0.17050|
|tiny|1|128|pp|Q4 GEMM slice join|0.16|0.02982|
|tiny|1|128|pp|RMSNorm|0.03|0.00078|
|tiny|1|128|pp|Q4 GEMV MAC+DMA|0.02|0.00419|
|tiny|1|128|pp|Q4 GEMV FP32 scale/sum|0.02|0.00074|
|tiny|1|128|tg|RMSNorm|90.00|1.01848|
|tiny|1|128|tg|Device copy_rows|90.00|0.73747|
|tiny|1|128|tg|Other elementwise|89.00|1.53972|
|tiny|1|128|tg|Q4 Activation quantization|462.00|10.66762|
|tiny|1|128|tg|Q4 GEMV MAC+DMA|154.00|16.92816|
|tiny|1|128|tg|Q4 GEMV FP32 scale/sum|154.00|2.77091|
|tiny|1|128|tg|RoPE|88.00|1.44303|
|tiny|1|128|tg|SET_ROWS|44.00|1.17045|
|tiny|1|128|tg|SDK cast/findmax (unmapped)|66.00|0.58207|
|tiny|1|128|tg|Flash Q ranges|22.00|0.34463|
|tiny|1|128|tg|Flash KQ+DMA|22.00|0.47696|
|tiny|1|128|tg|Flash Online/normalize/reduce|44.00|0.95122|
|tiny|1|128|tg|Flash PV+DMA|22.00|0.32914|
|tiny|1|128|tg|SwiGLU|22.00|0.66499|
|tiny|1|2048|pp|Other elementwise|1.03|0.37853|
|tiny|1|2048|pp|Device copy_rows|0.70|0.05190|
|tiny|1|2048|pp|Q4 Activation quantization|1.25|0.93285|
|tiny|1|2048|pp|Q4 GEMM MAC+block scales+DMA|1.18|1.40101|
|tiny|1|2048|pp|RoPE|0.69|0.09441|
|tiny|1|2048|pp|SET_ROWS|0.34|0.06370|
|tiny|1|2048|pp|Attention pack/copy|0.17|0.01576|
|tiny|1|2048|pp|SDK cast/findmax (unmapped)|0.52|0.02662|
|tiny|1|2048|pp|Flash Q ranges|0.17|0.00865|
|tiny|1|2048|pp|Flash KQ+DMA|0.43|0.08778|
|tiny|1|2048|pp|Flash Online/normalize/reduce|0.60|0.58109|
|tiny|1|2048|pp|Flash PV+DMA|0.43|0.05414|
|tiny|1|2048|pp|SwiGLU|0.17|0.17071|
|tiny|1|2048|pp|Q4 GEMM slice join|0.16|0.03087|
|tiny|1|2048|pp|RMSNorm|0.03|0.00092|
|tiny|1|2048|pp|Q4 GEMV MAC+DMA|0.02|0.00422|
|tiny|1|2048|pp|Q4 GEMV FP32 scale/sum|0.02|0.00076|
|tiny|1|2048|tg|RMSNorm|90.00|0.96930|
|tiny|1|2048|tg|Device copy_rows|90.00|0.70385|
|tiny|1|2048|tg|Other elementwise|89.00|1.50503|
|tiny|1|2048|tg|Q4 Activation quantization|462.00|10.61131|
|tiny|1|2048|tg|Q4 GEMV MAC+DMA|154.00|16.92355|
|tiny|1|2048|tg|Q4 GEMV FP32 scale/sum|154.00|2.74774|
|tiny|1|2048|tg|RoPE|88.00|1.43132|
|tiny|1|2048|tg|SET_ROWS|44.00|1.17860|
|tiny|1|2048|tg|SDK cast/findmax (unmapped)|66.00|1.03507|
|tiny|1|2048|tg|Flash Q ranges|22.00|0.33658|
|tiny|1|2048|tg|Flash KQ+DMA|22.00|1.93235|
|tiny|1|2048|tg|Flash Online/normalize/reduce|44.00|2.46935|
|tiny|1|2048|tg|Flash PV+DMA|22.00|1.43158|
|tiny|1|2048|tg|SwiGLU|22.00|0.66428|


## 排他的割合を使ったAmdahl仮定

対象区間全体を2倍、または費用ゼロにした楽観的な全体倍率。実装後の速度ではない。別候補の値は足し合わせない。

|model|prefix|phase|対象|割合 %|仮に対象2倍: 全体倍率|費用ゼロ上限倍率|
|---|---|---|---|---|---|---|
|tiny|128|pp|Q4 GEMV/GEMM|68.83|1.525|3.208|
|tiny|128|pp|FlashAttention XPU|6.51|1.034|1.070|
|tiny|128|pp|SwiGLU XPU|4.99|1.026|1.053|
|tiny|128|pp|KV SET_ROWS XPU|2.50|1.013|1.026|
|tiny|128|pp|Q6_K output CPU graph|0.45|1.002|1.005|
|tiny|128|pp|GEMM MAC, block-scale correction and device DMA|40.37|1.253|1.677|
|tiny|128|pp|GEMV MAC and device DMA|0.12|1.001|1.001|
|tiny|128|tg|Q4 GEMV/GEMM|64.80|1.479|2.841|
|tiny|128|tg|KV SET_ROWS XPU|6.12|1.032|1.065|
|tiny|128|tg|FlashAttention XPU|4.45|1.023|1.047|
|tiny|128|tg|Q6_K output CPU graph|3.81|1.019|1.040|
|tiny|128|tg|SwiGLU XPU|1.70|1.009|1.017|
|tiny|128|tg|GEMV MAC and device DMA|35.47|1.216|1.550|
|tiny|2048|pp|Q4 GEMV/GEMM|60.48|1.433|2.530|
|tiny|2048|pp|FlashAttention XPU|17.60|1.096|1.214|
|tiny|2048|pp|SwiGLU XPU|4.38|1.022|1.046|
|tiny|2048|pp|KV SET_ROWS XPU|2.19|1.011|1.022|
|tiny|2048|pp|Q6_K output CPU graph|0.39|1.002|1.004|
|tiny|2048|pp|GEMM MAC, block-scale correction and device DMA|35.46|1.215|1.549|
|tiny|2048|pp|GEMV MAC and device DMA|0.11|1.001|1.001|
|tiny|2048|tg|Q4 GEMV/GEMM|59.71|1.426|2.482|
|tiny|2048|tg|FlashAttention XPU|12.43|1.066|1.142|
|tiny|2048|tg|KV SET_ROWS XPU|5.73|1.029|1.061|
|tiny|2048|tg|Q6_K output CPU graph|3.54|1.018|1.037|
|tiny|2048|tg|SwiGLU XPU|1.56|1.008|1.016|
|tiny|2048|tg|GEMV MAC and device DMA|32.69|1.195|1.486|
|7b|128|pp|Q4 GEMV/GEMM|74.18|1.590|3.873|
|7b|128|pp|KV SET_ROWS XPU|7.99|1.042|1.087|
|7b|128|pp|SwiGLU XPU|3.45|1.018|1.036|
|7b|128|pp|FlashAttention XPU|2.95|1.015|1.030|
|7b|128|pp|Q6_K output CPU graph|0.21|1.001|1.002|
|7b|128|pp|GEMM MAC, block-scale correction and device DMA|54.14|1.371|2.180|
|7b|128|pp|GEMV MAC and device DMA|0.10|1.000|1.001|
|7b|128|tg|Q4 GEMV/GEMM|77.80|1.637|4.505|
|7b|128|tg|FlashAttention XPU|6.90|1.036|1.074|
|7b|128|tg|KV SET_ROWS XPU|3.20|1.016|1.033|
|7b|128|tg|Q6_K output CPU graph|2.48|1.013|1.025|
|7b|128|tg|SwiGLU XPU|0.83|1.004|1.008|
|7b|128|tg|GEMV MAC and device DMA|56.85|1.397|2.317|
|7b|2048|pp|Q4 GEMV/GEMM|69.81|1.536|3.312|
|7b|2048|pp|FlashAttention XPU|8.69|1.045|1.095|
|7b|2048|pp|KV SET_ROWS XPU|7.51|1.039|1.081|
|7b|2048|pp|SwiGLU XPU|3.24|1.016|1.034|
|7b|2048|pp|Q6_K output CPU graph|0.20|1.001|1.002|
|7b|2048|pp|GEMM MAC, block-scale correction and device DMA|50.87|1.341|2.035|
|7b|2048|pp|GEMV MAC and device DMA|0.09|1.000|1.001|
|7b|2048|tg|Q4 GEMV/GEMM|67.77|1.512|3.103|
|7b|2048|tg|FlashAttention XPU|17.47|1.096|1.212|
|7b|2048|tg|KV SET_ROWS XPU|3.12|1.016|1.032|
|7b|2048|tg|Q6_K output CPU graph|2.17|1.011|1.022|
|7b|2048|tg|SwiGLU XPU|0.73|1.004|1.007|
|7b|2048|tg|GEMV MAC and device DMA|49.77|1.331|1.991|
