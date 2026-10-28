# 集計データの保存範囲

Gitには、小さい集計CSV/JSONとレポート図を保存する。生trace、raw logits、
実行log、SDK、モデル、ビルド済shared object/kernel imageは含めない。

artifact版のこのディレクトリには、次の大きい詳細CSVも同じ相対pathで付属する：
q4-stages.csv、transfers.csv、device-copies.csv、setrows-overhead.csv、
inclusive-api.csv、cpu-workers.csv、wall-partition.csv、overlap.csv、
ordinary-samples.csv、dispatch-logs.csv、Flash per-core work/timeline等。
Git保存版のレポートからこれらへのリンクは、ローカル成果物中のpathを示す。

実機の保存先は `/home/unagidojyou/kunlun-k200/results/profiling-20261028`。
解析は別のWindows環境で行い、report/data packetを実機へ戻して保持する。
通常性能と診断時間は分け、主判断にはlight-wall-partition/light-amdahlを使う。
全runtimeの割合には、主にTGで大きいobserver負荷が含まれる。

再生成はtools/kunlun-profilingのREADME、元の固定条件は上位のREPRODUCE.ja.mdを参照。
