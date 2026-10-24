# Resident FFN SwiGLU

KunlunK200 0.5.0 is required by both bundled and system-library builds.

`GGML_KUNLUN_SWIGLU=1` opts into plain `GGML_OP_GLU/GGML_GLU_OP_SWIGLU`. The default and `=0` keep CPU GLU. `GGML_KUNLUN_SWIGLU=pp` uses the GPU only for multiple rows; one-row operations keep CPU GLU, including any pruned PP operation. This can retain PP transfer savings while avoiding small-shape TG regressions. Invalid values disable the feature and are logged. Set the mode before context creation and keep it fixed for that context. This setting is independent of the existing common/native/CDNN Q4 settings and does not change their priority or weight ABI.

Supported inputs are matching F32 or F16 tensors with contiguous inner elements and flattenable outer rows. Separate gate/up, packed input, packed swapped input and padded rows are supported. The output has the same type. Views are accepted only when their shape/strides satisfy the library contract. GEGLU, ReGLU, clamped/OAI variants, BF16, noncontiguous inner elements, incompatible outer strides and split-buffer inputs retain CPU placement. The output and both activation inputs must be normal device buffers on the executing PD. Row Split uses the existing gathered owning-PD activations; Layer Split runs GLU on each layer's PD.

The backend calls the library once. Gate/up remain on the XPU and output feeds the down projection without a CPU readback. There is no intermediate SiLU tensor or activation quantization in GLU. F16 GLU support does not make the existing F16-input MUL_MAT contract wider; an unsupported downstream operation still falls back normally.

```
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S . -B build -G Ninja -DGGML_KUNLUN=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j4 --target kunlun-llm-smoke kunlun-swiglu-test kunlun-quality-eval
OMP_WAIT_POLICY=PASSIVE GGML_KUNLUN_COMMON_Q4=1 GGML_KUNLUN_SWIGLU=1 \
 KUNLUN_BROADCAST_VECTOR=1 K200_CTX=2048 K200_BATCH=128 K200_UBATCH=128 \
 K200_IGNORE_EOS=1 build/bin/kunlun-llm-smoke /path/model-q4_0.gguf none 32 logits.bin
```

Use the existing guarded runner for device execution. `layer` and `row` select the existing split modes. Set `GGML_KUNLUN_COMMON_Q4=0 GGML_KUNLUN_REQUANT_I8=1` for CDNN, or set legacy `GGML_KUNLUN_NATIVE_Q4=1`. Keep SWIGLU=0 for the old CPU-GLU comparison.

Runtime totals report actual GPU calls/elements and phase counts per PD. `GGML_KUNLUN_DIAGNOSTICS=1` logs a bounded set of unsupported placement reasons; those queries are not CPU execution counts. Use separate `K200_TRACE_OPS=1` runs to count actual CPU fallback nodes. That callback synchronizes nodes and must not be used for normal PP/TG performance.

Finite scalar CPU and SIMD CPU implementations have slightly different rounding. GPU GLU can therefore change logits and greedy paths after existing A8 quantization. The opt-in is not a claim of bitwise model equivalence or universally faster TG. Compare CPU-reference KL, fixed-token perplexity, generation and the actual workload before selecting it.

Diagnostic-only harness options: `K200_CAPTURE_GLU=<existing-dir>` with `K200_TRACE_OPS=1` captures selected F32 FFN intermediates. `K200_CAPTURE_LAYER` selects one layer, or -1 selects all. `K200_CHECK_STATE=1` checks save/restore by comparing next-token logits bitwise; exclude those runs from performance. `K200_QUALITY_BATCH` changes the quality-eval batch/ubatch size (default8).
