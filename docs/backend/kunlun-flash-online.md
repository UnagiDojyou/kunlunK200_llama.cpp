# Kunlun K200 FlashAttention Online Softmax

KunlunK200 0.7.1 preserves the existing fused attention graph and SDNN/KV-writer paths. The CL Online Softmax kernel defaults to the optimized ordered-sum implementation. Compare it with the original 0.7.0 implementation using:

```sh
GGML_KUNLUN_ATTENTION=1 GGML_KUNLUN_FLASH=1 KUNLUN_ATTENTION_MATMUL=auto KUNLUN_FLASH_KV=1 KUNLUN_FLASH_ONLINE=legacy ... -fa on
GGML_KUNLUN_ATTENTION=1 GGML_KUNLUN_FLASH=1 KUNLUN_ATTENTION_MATMUL=auto KUNLUN_FLASH_KV=1 KUNLUN_FLASH_ONLINE=auto ... -fa on
```

Use identical model, context, batch/ubatch, split, block, query tile and KV writer settings. Do not enable `KUNLUN_FLASH_TRACE`, operation diagnostics or preload audit libraries in normal token/s measurements. The standalone library benchmarks and Japanese report are in `vendor/KunlunK200/docs/flash-online.md` and `vendor/KunlunK200/docs/reports/flash-online-20261026/REPORT.ja.md`. The optimization changes no graph construction or public tensor layout.
