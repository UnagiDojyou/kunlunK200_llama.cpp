# Q4 Activation 共有（opt-in）

KunlunK200 0.7.7 以上を使用します。共通 weight ABI v1、Q4 GEMV/GEMM、Attention 0.7.6、SET_ROWS、SwiGLU、CDNN の演算方法は維持しています。

モデル／backend の初期化前に設定してください。

```sh
export GGML_KUNLUN_COMMON_Q4=1
export GGML_KUNLUN_Q4_ACT_SHARE=1
# 他の Attention、KV、SwiGLU 設定は比較条件と同じに保つ。
llama-bench -m /path/model-q4_0.gguf -ngl 99 -dev KUNLUN0 -sm none -p 512 -n 128 -r 5 -b 128 -ub 128 -t 6 -ctk f16 -ctv f16 -fa on
```

共有を無効にするには `GGML_KUNLUN_Q4_ACT_SHARE=0` または unset を使用します。既定値は無効です。COMMON_Q4=1 のときだけ共有可能であり、旧 native／CDNN／直接 Q4 は従来のままです。COMMON_Q4 が無効で ACT_SHARE=1 の場合には理由をログに出して共有を無効にします。0/1 以外も明示的に無効にします。共通 pack と旧 pack の二重常駐は追加しません。

## 再利用条件と寿命

GGML の MUL_MAT に渡される実際の src[1] tensor object をグラフ内で数え、複数の common-Q4 weight が使う入力を選びます。現在の GGML 対応型は既存 MUL_MAT と同じ F32、連続した行列です。ライブラリ自体の F16 API は別に検証していますが、GGML の対応範囲は広げません。

- graph_compute の期間内に限定し、グラフの開始／終了で全 Lease を破棄します。tensor object、buffer、data pointer、dtype、全 ne/nb と byte extent を検証し、グループごとに新しい 64 bit generation を発行します。pointer だけのキャッシュではありません。
- 実際に書き込むノードの出力 byte span が共有元と重なる場合は Lease を失効させます。VIEW/RESHAPE/PERMUTE/TRANSPOSE はメタデータのみのため書込み扱いにしません。別名 view 経由の書込みは span で検出します。
- runtime_mutex がグラフ実行と backend の buffer 書込みを直列化します。外部書込み／context 再利用は次の graph_compute で新しい世代になります。raw SDK による mutex を使わない外部書込みはサポートしません。
- PD 別に独立した workspace/Lease を持ち、Layer Split は各所有 PD 内で共有、Row Split は各 PD で一度ずつ量子化します。PD 間で scale／量子化値をコピーする方式ではありません。
- Row Split で同じ入力が既に remote staging にある場合、入力 P2P も省略します。staging pointer、byte extent と論理 generation を検証し、F32/F16 など別の Row matmul が staging を上書き／再確保した場合は再転送します。shared API が安全に拒否した場合にも、従来経路が読む元 Activation を有効に保ちます。
- 最後の consumer が完了した時点で Lease を破棄します。現 API は同期的に完了するため追加 wait はありません。将来非同期 API に変更する際には完了イベントまで Lease を保持する設計が別途必要です。

同時に保持する共有グループは backend あたり一つです。別の入力グループや通常 common matmul が workspace を再利用した場合、次の consumer は再準備または従来経路を使用します。CPU scheduler の分割を跨いで寿命を伸ばしません。QKV と gate/up の通常グラフではそれぞれ共有が成立することを実 LLM のカウンタで検証します。

## メモリ・診断・試験

weight の pack/upload は変更しません。各 workspace に量子化値と scale の allocation は各一つです。M>128 の共有は全 chunk を保持するため HBM peak が増える可能性があります。ライブラリは quant buffer 16 MiB を上限とし、超える場合は従来の逐次 chunk 経路を使います。追加の L3 は不要です。容量拡張時の一時的な再確保は既存 allocator の方式を継承しており、定常 workspace byte と SDK API peak は別に比較します。

終了ログの `KUNLUN activation` に prepares/reuse/fallback、量子化 chunk／kernel 起動数、HBM 使用量を表示します。`KUNLUN activation graph` に graph/group、書込みによる失効、削減した入力 P2P の回数／byte を表示します。reuse は prepared MAC が成功した再利用だけを数えます。最初の拒否では PD・shape・generation と prepare/prepared-MAC の段階も記録します。fallback は準備／prepared dispatch が拒否された回数であり、共有対象外の通常 matmul 全件を意味しません。AO/down のような単独 consumer は意図的に従来の量子化を使います。backend context ごとの集計なので、Row Split で両 PD の backend が作られた場合は全 context を集計してください。

通常性能測定では diagnostic hooks、LD_PRELOAD、K200_TRACE_OPS、KUNLUN_Q4_ACT_PROFILE=1 を使用しません。診断ビルド（C/CXX flags に -DK200_DIAGNOSTIC_HOOKS）と KUNLUN_Q4_ACT_PROFILE=1 を組み合わせた別実行で、PP/TG 別の quantization 時間と管理イベントを取得できます。追加の GPU 同期は入れませんが、ホスト計測の負荷があるので通常 token/s と区別します。quantization 時間は既存 wait を含むホスト経過時間であり、xpu.node の時間に重複して加算しないでください。

`kunlun-activation-share-test 0` および `1` は Single/Row、M=1/9/129、3世代の同一アドレス再利用、別名書込み、view、新旧演算混在・staging resize を共有 off/on で bitwise 比較します。`kunlun-backend-contract` も引き続き実行してください。GPU 実行エラーは CPU fallback で隠さず、既存 thermal guard／persistent abort latch を使用します。
