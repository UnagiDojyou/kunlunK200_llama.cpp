# 軽量observerによる主時間内訳

XPU runtimeのinterposeを外し、GGMLの演算・CPU worker・scheduler・tensor操作scopeだけを収集した。追加XPU同期なし、同じ0.7.6 library/shader、同じGGML演算ソース。全runtime traceは転送bytes・kernel段階の補助資料へ分離する。

## ABBA校正

Single PD0、SwiGLU=1、prefix128。各実行3試行、A/B各6試行。全32位置のlogitsは4実行間でビット一致。

|model|observer|phase|n|平均 ms/token|中央値|SD|通常比増加 %|
|---|---|---|---|---|---|---|---|
|7b|none|pp|6|13.74979|13.74954|0.02122|—|
|7b|none|tg|6|135.87551|135.87036|0.08145|—|
|7b|light|pp|6|13.76726|13.76389|0.01816|0.13|
|7b|light|tg|6|138.26291|138.23617|0.10984|1.76|
|tiny|none|pp|6|3.37275|3.42512|0.08334|—|
|tiny|none|tg|6|41.62209|41.10541|0.86487|—|
|tiny|light|pp|6|3.40625|3.43307|0.06841|0.99|
|tiny|light|tg|6|43.35199|43.12361|0.58203|4.16|


## 排他的区間

PD0、batch/ubatch128/128、SwiGLU=1、context容量4096、各3試行。PPの単位はprompt token、TGはdecode。外部tensor操作にはCPU bufferコピーも含むため、このscopeのbytesをH2D/D2Hへ読み替えない。

### tiny / prefix 128 / PP

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|440.172|3.43885 ± 0.00329|100.00|1|
|Q4 GEMV/GEMM|304.713|2.38057 ± 0.00154|69.23|154|
|Elementwise/residual XPU|38.218|0.29858 ± 0.00100|8.68|89|
|FlashAttention XPU|28.938|0.22608 ± 0.00006|6.57|22|
|SwiGLU XPU|21.750|0.16992 ± 0.00044|4.94|22|
|RMSNorm XPU|17.451|0.13633 ± 0.00027|3.96|45|
|RoPE XPU|13.682|0.10689 ± 0.00081|3.11|44|
|KV SET_ROWS XPU|10.115|0.07903 ± 0.00047|2.30|44|
|Q6_K output CPU graph|1.919|0.01499 ± 0.00028|0.44|1|
|Remaining host/scheduler/measurement|1.041|0.00813 ± 0.00061|0.24|1|
|External tensor access/copy|0.867|0.00677 ± 0.00011|0.20|68|
|Scheduler split/allocation|0.596|0.00466 ± 0.00026|0.14|2|
|XPU metadata|0.523|0.00408 ± 0.00030|0.12|264|
|Embedding CPU graph|0.323|0.00252 ± 0.00058|0.07|1|
|External backend synchronization|0.028|0.00022 ± 0.00000|0.01|25|
|Output selection GET_ROWS CPU graph|0.010|0.00008 ± 0.00001|0.00|1|


### tiny / prefix 128 / TG

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|1328.128|42.84283 ± 0.09120|100.00|31|
|Q4 GEMV/GEMM|906.692|29.24813 ± 0.00606|68.27|4774|
|Elementwise/residual XPU|73.241|2.36261 ± 0.00307|5.51|2759|
|RoPE XPU|68.217|2.20056 ± 0.00254|5.14|1364|
|KV SET_ROWS XPU|67.503|2.17753 ± 0.00529|5.08|1364|
|FlashAttention XPU|58.404|1.88399 ± 0.00207|4.40|682|
|Q6_K output CPU graph|54.972|1.77328 ± 0.00950|4.14|31|
|RMSNorm XPU|32.131|1.03649 ± 0.00169|2.42|1395|
|Remaining host/scheduler/measurement|21.872|0.70554 ± 0.06137|1.65|31|
|SwiGLU XPU|21.743|0.70138 ± 0.00042|1.64|682|
|XPU metadata|10.248|0.33057 ± 0.00121|0.77|8184|
|Embedding CPU graph|6.632|0.21394 ± 0.00598|0.50|31|
|External tensor access/copy|4.345|0.14016 ± 0.00244|0.33|2108|
|External backend synchronization|1.004|0.03238 ± 0.00095|0.08|775|
|Scheduler split/allocation|0.897|0.02894 ± 0.01682|0.07|2|
|Output selection GET_ROWS CPU graph|0.227|0.00733 ± 0.00022|0.02|31|


### tiny / prefix 2048 / PP

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|7973.983|3.89355 ± 0.00414|100.00|16|
|Q4 GEMV/GEMM|4860.070|2.37308 ± 0.00173|60.95|2464|
|FlashAttention XPU|1427.665|0.69710 ± 0.00021|17.90|352|
|Elementwise/residual XPU|603.496|0.29468 ± 0.00093|7.57|1424|
|SwiGLU XPU|350.129|0.17096 ± 0.00013|4.39|352|
|RMSNorm XPU|276.680|0.13510 ± 0.00020|3.47|720|
|RoPE XPU|215.021|0.10499 ± 0.00062|2.70|704|
|KV SET_ROWS XPU|159.838|0.07805 ± 0.00038|2.00|704|
|Q6_K output CPU graph|31.101|0.01519 ± 0.00009|0.39|16|
|Remaining host/scheduler/measurement|17.541|0.00856 ± 0.00002|0.22|16|
|External tensor access/copy|14.093|0.00688 ± 0.00004|0.18|1088|
|XPU metadata|8.232|0.00402 ± 0.00010|0.10|4224|
|Scheduler split/allocation|5.515|0.00269 ± 0.00031|0.07|16|
|Embedding CPU graph|3.861|0.00189 ± 0.00004|0.05|16|
|External backend synchronization|0.585|0.00029 ± 0.00003|0.01|400|
|Output selection GET_ROWS CPU graph|0.152|0.00007 ± 0.00001|0.00|16|


### tiny / prefix 2048 / TG

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|1473.385|47.52856 ± 0.21543|100.00|31|
|Q4 GEMV/GEMM|908.805|29.31630 ± 0.01967|61.68|4774|
|FlashAttention XPU|198.699|6.40966 ± 0.00282|13.49|682|
|Elementwise/residual XPU|73.753|2.37914 ± 0.01038|5.01|2759|
|RoPE XPU|69.011|2.22617 ± 0.00311|4.68|1364|
|KV SET_ROWS XPU|67.709|2.18415 ± 0.00276|4.60|1364|
|Q6_K output CPU graph|55.707|1.79699 ± 0.01445|3.78|31|
|RMSNorm XPU|32.411|1.04552 ± 0.00501|2.20|1395|
|Remaining host/scheduler/measurement|22.774|0.73465 ± 0.18192|1.54|31|
|SwiGLU XPU|21.805|0.70338 ± 0.00194|1.48|682|
|XPU metadata|10.371|0.33455 ± 0.00310|0.70|8184|
|Embedding CPU graph|5.556|0.17923 ± 0.01177|0.38|31|
|External tensor access/copy|4.848|0.15638 ± 0.00872|0.33|2108|
|External backend synchronization|1.096|0.03536 ± 0.00361|0.07|775|
|Scheduler split/allocation|0.611|0.01972 ± 0.00142|0.04|2|
|Output selection GET_ROWS CPU graph|0.229|0.00737 ± 0.00012|0.02|31|


### 7b / prefix 128 / PP

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|1763.655|13.77855 ± 0.00526|100.00|1|
|Q4 GEMV/GEMM|1309.991|10.23431 ± 0.00255|74.28|224|
|KV SET_ROWS XPU|140.559|1.09812 ± 0.00025|7.97|64|
|Elementwise/residual XPU|101.950|0.79649 ± 0.00151|5.78|129|
|SwiGLU XPU|60.651|0.47384 ± 0.00043|3.44|32|
|FlashAttention XPU|52.072|0.40681 ± 0.00016|2.95|32|
|RMSNorm XPU|47.895|0.37418 ± 0.00020|2.72|65|
|RoPE XPU|42.365|0.33097 ± 0.00024|2.40|64|
|Q6_K output CPU graph|3.677|0.02873 ± 0.00040|0.21|1|
|Remaining host/scheduler/measurement|1.415|0.01106 ± 0.00024|0.08|1|
|External tensor access/copy|1.327|0.01037 ± 0.00006|0.08|88|
|Scheduler split/allocation|0.823|0.00643 ± 0.00021|0.05|2|
|XPU metadata|0.529|0.00413 ± 0.00024|0.03|384|
|Embedding CPU graph|0.355|0.00278 ± 0.00030|0.02|1|
|External backend synchronization|0.032|0.00025 ± 0.00006|0.00|25|
|Output selection GET_ROWS CPU graph|0.013|0.00010 ± 0.00000|0.00|1|


### 7b / prefix 128 / TG

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|4285.210|138.23257 ± 0.05848|100.00|31|
|Q4 GEMV/GEMM|3413.757|110.12120 ± 0.02627|79.66|6944|
|FlashAttention XPU|305.064|9.84079 ± 0.00209|7.12|992|
|Elementwise/residual XPU|112.708|3.63573 ± 0.00756|2.63|3999|
|Q6_K output CPU graph|110.077|3.55087 ± 0.01460|2.57|31|
|KV SET_ROWS XPU|101.925|3.28789 ± 0.00621|2.38|1984|
|RoPE XPU|101.832|3.28490 ± 0.00421|2.38|1984|
|RMSNorm XPU|50.181|1.61874 ± 0.00696|1.17|2015|
|SwiGLU XPU|32.802|1.05815 ± 0.00309|0.77|992|
|Remaining host/scheduler/measurement|27.850|0.89840 ± 0.09084|0.65|31|
|XPU metadata|14.990|0.48356 ± 0.00103|0.35|11904|
|Embedding CPU graph|6.213|0.20041 ± 0.01050|0.14|31|
|External tensor access/copy|5.762|0.18586 ± 0.03867|0.13|2728|
|External backend synchronization|0.950|0.03066 ± 0.00107|0.02|775|
|Scheduler split/allocation|0.842|0.02715 ± 0.00135|0.02|2|
|Output selection GET_ROWS CPU graph|0.256|0.00825 ± 0.00027|0.01|31|


### 7b / prefix 2048 / PP

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|29971.856|14.63470 ± 0.00306|100.00|16|
|Q4 GEMV/GEMM|20948.730|10.22887 ± 0.00168|69.89|3584|
|FlashAttention XPU|2616.281|1.27748 ± 0.00001|8.73|512|
|KV SET_ROWS XPU|2247.884|1.09760 ± 0.00013|7.50|1024|
|Elementwise/residual XPU|1624.929|0.79342 ± 0.00082|5.42|2064|
|SwiGLU XPU|969.562|0.47342 ± 0.00005|3.23|512|
|RMSNorm XPU|765.014|0.37354 ± 0.00024|2.55|1040|
|RoPE XPU|675.933|0.33005 ± 0.00019|2.26|1024|
|Q6_K output CPU graph|59.229|0.02892 ± 0.00011|0.20|16|
|External tensor access/copy|21.991|0.01074 ± 0.00005|0.07|1408|
|Remaining host/scheduler/measurement|21.901|0.01069 ± 0.00007|0.07|16|
|XPU metadata|8.369|0.00409 ± 0.00002|0.03|6144|
|Scheduler split/allocation|6.642|0.00324 ± 0.00001|0.02|16|
|Embedding CPU graph|4.564|0.00223 ± 0.00005|0.02|16|
|External backend synchronization|0.625|0.00031 ± 0.00003|0.00|400|
|Output selection GET_ROWS CPU graph|0.201|0.00010 ± 0.00000|0.00|16|


### 7b / prefix 2048 / TG

|区間|区間計 ms|ms/単位 ± SD|割合 %|呼出/試行|
|---|---|---|---|---|
|TOTAL|4887.056|157.64697 ± 0.36841|100.00|31|
|Q4 GEMV/GEMM|3410.789|110.02546 ± 0.02317|69.79|6944|
|FlashAttention XPU|880.159|28.39224 ± 0.01764|18.01|992|
|KV SET_ROWS XPU|133.283|4.29946 ± 0.37382|2.73|1984|
|Q6_K output CPU graph|110.502|3.56458 ± 0.00650|2.26|31|
|Elementwise/residual XPU|109.677|3.53797 ± 0.01705|2.24|3999|
|RoPE XPU|102.582|3.30911 ± 0.00153|2.10|1984|
|RMSNorm XPU|48.618|1.56831 ± 0.00856|0.99|2015|
|SwiGLU XPU|32.158|1.03735 ± 0.00103|0.66|992|
|Remaining host/scheduler/measurement|30.567|0.98604 ± 0.15877|0.63|31|
|XPU metadata|15.115|0.48758 ± 0.00187|0.31|11904|
|Embedding CPU graph|6.288|0.20283 ± 0.01022|0.13|31|
|External tensor access/copy|5.251|0.16937 ± 0.00509|0.11|2728|
|External backend synchronization|0.990|0.03195 ± 0.00057|0.02|775|
|Scheduler split/allocation|0.816|0.02631 ± 0.00064|0.02|2|
|Output selection GET_ROWS CPU graph|0.261|0.00842 ± 0.00007|0.01|31|


## Amdahl仮定

軽量診断の排他的区間を使う。対象を2倍、または費用ゼロと置いた全体倍率。実装後の実測値ではない。observerの費用も含まれ、別候補の倍率を足したり掛けたりしない。

|model|prefix|phase|対象|割合 %|対象2倍の全体倍率|費用ゼロ上限|
|---|---|---|---|---|---|---|
|tiny|128|pp|Elementwise/residual XPU|8.68|1.045|1.095|
|tiny|128|pp|FlashAttention XPU|6.57|1.034|1.070|
|tiny|128|pp|KV SET_ROWS XPU|2.30|1.012|1.024|
|tiny|128|pp|Q4 GEMV/GEMM|69.23|1.529|3.249|
|tiny|128|pp|Q6_K output CPU graph|0.44|1.002|1.004|
|tiny|128|pp|RMSNorm XPU|3.96|1.020|1.041|
|tiny|128|pp|SwiGLU XPU|4.94|1.025|1.052|
|tiny|128|tg|Elementwise/residual XPU|5.51|1.028|1.058|
|tiny|128|tg|FlashAttention XPU|4.40|1.022|1.046|
|tiny|128|tg|KV SET_ROWS XPU|5.08|1.026|1.054|
|tiny|128|tg|Q4 GEMV/GEMM|68.27|1.518|3.151|
|tiny|128|tg|Q6_K output CPU graph|4.14|1.021|1.043|
|tiny|128|tg|RMSNorm XPU|2.42|1.012|1.025|
|tiny|128|tg|SwiGLU XPU|1.64|1.008|1.017|
|tiny|2048|pp|Elementwise/residual XPU|7.57|1.039|1.082|
|tiny|2048|pp|FlashAttention XPU|17.90|1.098|1.218|
|tiny|2048|pp|KV SET_ROWS XPU|2.00|1.010|1.020|
|tiny|2048|pp|Q4 GEMV/GEMM|60.95|1.438|2.561|
|tiny|2048|pp|Q6_K output CPU graph|0.39|1.002|1.004|
|tiny|2048|pp|RMSNorm XPU|3.47|1.018|1.036|
|tiny|2048|pp|SwiGLU XPU|4.39|1.022|1.046|
|tiny|2048|tg|Elementwise/residual XPU|5.01|1.026|1.053|
|tiny|2048|tg|FlashAttention XPU|13.49|1.072|1.156|
|tiny|2048|tg|KV SET_ROWS XPU|4.60|1.024|1.048|
|tiny|2048|tg|Q4 GEMV/GEMM|61.68|1.446|2.610|
|tiny|2048|tg|Q6_K output CPU graph|3.78|1.019|1.039|
|tiny|2048|tg|RMSNorm XPU|2.20|1.011|1.022|
|tiny|2048|tg|SwiGLU XPU|1.48|1.007|1.015|
|7b|128|pp|Elementwise/residual XPU|5.78|1.030|1.061|
|7b|128|pp|FlashAttention XPU|2.95|1.015|1.030|
|7b|128|pp|KV SET_ROWS XPU|7.97|1.042|1.087|
|7b|128|pp|Q4 GEMV/GEMM|74.28|1.591|3.888|
|7b|128|pp|Q6_K output CPU graph|0.21|1.001|1.002|
|7b|128|pp|RMSNorm XPU|2.72|1.014|1.028|
|7b|128|pp|SwiGLU XPU|3.44|1.017|1.036|
|7b|128|tg|Elementwise/residual XPU|2.63|1.013|1.027|
|7b|128|tg|FlashAttention XPU|7.12|1.037|1.077|
|7b|128|tg|KV SET_ROWS XPU|2.38|1.012|1.024|
|7b|128|tg|Q4 GEMV/GEMM|79.66|1.662|4.917|
|7b|128|tg|Q6_K output CPU graph|2.57|1.013|1.026|
|7b|128|tg|RMSNorm XPU|1.17|1.006|1.012|
|7b|128|tg|SwiGLU XPU|0.77|1.004|1.008|
|7b|2048|pp|Elementwise/residual XPU|5.42|1.028|1.057|
|7b|2048|pp|FlashAttention XPU|8.73|1.046|1.096|
|7b|2048|pp|KV SET_ROWS XPU|7.50|1.039|1.081|
|7b|2048|pp|Q4 GEMV/GEMM|69.89|1.537|3.322|
|7b|2048|pp|Q6_K output CPU graph|0.20|1.001|1.002|
|7b|2048|pp|RMSNorm XPU|2.55|1.013|1.026|
|7b|2048|pp|SwiGLU XPU|3.23|1.016|1.033|
|7b|2048|tg|Elementwise/residual XPU|2.24|1.011|1.023|
|7b|2048|tg|FlashAttention XPU|18.01|1.099|1.220|
|7b|2048|tg|KV SET_ROWS XPU|2.73|1.014|1.028|
|7b|2048|tg|Q4 GEMV/GEMM|69.79|1.536|3.310|
|7b|2048|tg|Q6_K output CPU graph|2.26|1.011|1.023|
|7b|2048|tg|RMSNorm XPU|0.99|1.005|1.010|
|7b|2048|tg|SwiGLU XPU|0.66|1.003|1.007|
