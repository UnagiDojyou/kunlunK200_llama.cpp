# Q4 MAC/DMA stream GEMV

Requires KunlunK200 0.7.8. Enable before model/backend initialization:

~~~sh
export GGML_KUNLUN_COMMON_Q4=1
export KUNLUN_Q4_MAC_DMA=1
export GGML_KUNLUN_Q4_ACT_SHARE=1  # optional; independently controlled
~~~

The library owns this option and dispatch; GGML does not duplicate the kernel. 0 or unset retains the stable kernel. Unsupported shapes use stable GEMV with a reason in the log. Legacy native, CDNN, direct Q4 and CPU paths are unchanged. Existing M>1 SD GEMM kernels are unchanged. Attention remains at the 0.7.6 implementation.

The new path reuses common ABI v1 weights, original K32 scales, quantized Activation, Lease generation and workspace ownership. Model-load pack/upload and PP-to-TG weight lifetime are unchanged. Single and Layer use their owning PD; Row uses the same existing PD-local input/output and join paths.

The stream keeps up to 128 K32 Activation blocks in L1D and removes pipeline drain/refill between those windows. Each core executes a specialized owner loop. Existing DMA/DS/MAC/EW/RS completion fences and scale/weight slot lifetimes remain intact. Per-output MAC and final summation order are unchanged. No GPU waits or transfers are added by the adapter.

End-of-backend logs include KUNLUN common MAC PD*: owner, baseline, stream, shape_fallback and resident_l1d. Row may create multiple backend contexts; aggregate all owners. Diagnostic builds additionally report per-phase MAC path counters. Those per-call queries are compiled out of normal builds. Path counters are not physical engine busy cycles.

Keep Attention, SwiGLU, Activation sharing, context, batch/ubatch and split fixed for a MAC comparison. Toggle KUNLUN_Q4_MAC_DMA alone. Compare Activation sharing separately in a 2x2 measurement. Do not use instrumented kernel timings as normal LLM throughput.

~~~sh
llama-bench -m /path/model-q4_0.gguf -ngl 99 -dev KUNLUN0 -sm none \
 -p 512 -n 128 -r 5 -b 128 -ub 128 -t 6 -ctk f16 -ctv f16 -fa on
# PD1: -dev KUNLUN1
# Layer: -dev KUNLUN0/KUNLUN1 -sm layer -ts 1/1
# Row: -dev KUNLUN0/KUNLUN1 -sm row -ts 1/1
~~~

Run hardware tests through the existing thermal guard and persistent abort latch. Never continue after a device error without state checks and incident preservation. No driver, firmware, clock or OS changes are needed.

Measured accuracy, end-to-end throughput, memory and reproducible commands are in [the Japanese evaluation report](reports/q4-mac-dma-20261029/REPORT.ja.md).

Default-off rebuild throughput differed from frozen 0.7.7 in some measured configurations. Keep the stable build and review the separate control tables before replacing it.
