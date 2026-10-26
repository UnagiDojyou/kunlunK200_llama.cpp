# K200 FlashAttention KQ reuse

KunlunK200 0.7.2 adds `KUNLUN_FLASH_KQ=auto/legacy/reuse1024`. Auto (also unset) reuses quantized Q in L1D and K in L1W when each SDNN part has multiple 32-key tiles. Legacy retains 0.7.1 KQ. Unknown values also select legacy. This does not change graph construction, quantization, KV writer, Online Softmax or KQV.

For CLI comparison, enable `GGML_KUNLUN_ATTENTION=1 GGML_KUNLUN_FLASH=1 KUNLUN_FLASH_KV=1 KUNLUN_ATTENTION_MATMUL=auto KUNLUN_FLASH_ONLINE=auto` and `-fa on`, and vary only `KUNLUN_FLASH_KQ`. Keep model, batch/ubatch, split, block, query tile and GQA options identical. Unset diagnostic tracing/audit flags for token/s measurements. Test tools use `K200_FLASH=1` instead of `-fa on`.

See `vendor/KunlunK200/docs/flash-kq.md` and `vendor/KunlunK200/docs/reports/flash-kq-20261026/REPORT.ja.md` for exact measurements and limitations. The source baseline includes uncommitted 0.7.1 changes; use the saved snapshot/working source when reproducing it.
