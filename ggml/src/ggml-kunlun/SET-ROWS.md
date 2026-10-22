# F16 KV cache SET_ROWS

The backend requires KunlunK200 0.4.1 for contiguous transposed V-cache updates with more than 32 tokens. This applies to both the bundled submodule and `KUNLUN_USE_SYSTEM_LIBRARY=ON`. The original GGML F16 KV allocation, tensor views and cache contents are unchanged.

The previous backend limited the detected contiguous mapping to 32 tokens. PP128 therefore used one two-byte output DMA per V element and sorted a large index list. The backend now recognizes any positive token count that fits the cache stride. The library splits it into bounded 32-token tiles. The complete host index check and device validation remain; arbitrary indices, duplicates, non-contiguous layouts and CPU fallback keep their previous behavior.

No Q4 weight layout, GEMV/GEMM selection, CDNN configuration or default quantization profile is changed. Single PD, Layer Split and Row Split use the same owning-PD cache update. The existing `GGML_KUNLUN_SET_ROWS=legacy` CPU roundtrip comparison remains available.

```
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S . -B build -G Ninja -DGGML_KUNLUN=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j4 --target kunlun-llm-smoke
OMP_WAIT_POLICY=PASSIVE GGML_KUNLUN_COMMON_Q4=1 KUNLUN_BROADCAST_VECTOR=1 \
  K200_CTX=2048 K200_BATCH=128 K200_UBATCH=128 K200_IGNORE_EOS=1 \
  build/bin/kunlun-llm-smoke /path/model-q4_0.gguf none 32 logits.bin
```

Use `layer` or `row` instead of `none` to test split execution. Use `GGML_KUNLUN_COMMON_Q4=0 GGML_KUNLUN_REQUANT_I8=1` for CDNN or add `GGML_KUNLUN_NATIVE_Q4=1` for the old native Q4 profile. Run all commands under the existing thermal guard.

Measure ordinary PP/TG without `K200_TRACE_OPS`, preload profiling or graph callbacks. `K200_TRACE_OPS=1` synchronizes observed graph nodes and is a separate diagnostic. Runtime-copy tracing that adds no new waits is still a separate instrumented measurement and must not replace the ordinary baseline.

Attention KQ and KQV are batched/head-strided F16 views; they remain on the CPU because the current K200 MUL_MAT contract supports contiguous two-dimensional matrices. Moving attention requires separate batched-stride/GQA matmul and masked softmax support, or a validated fused attention implementation. Reusing integer Q4 kernels or forcing these views through the existing scalar matrix fallback is not appropriate.
