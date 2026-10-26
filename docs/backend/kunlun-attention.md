# Kunlun resident Attention

This backend requires KunlunK2000.6.0 for bundled and system-library builds. Default operation is unchanged. Existing Q4 common/native/CDNN/direct selection and F16 KV cache storage remain unchanged.

## Selection

| Setting | Behavior |
| --- | --- |
| `GGML_KUNLUN_ATTENTION=0` or unset | Existing CPU Attention |
| `GGML_KUNLUN_ATTENTION=1` | Supported KQ, Softmax, KQV, CONT and RoPE on the owning PD |
| `GGML_KUNLUN_ATTENTION=tg` | Native Attention for one-query shapes; multiple-query PP retains the existing CPU route |
| `GGML_KUNLUN_ROPE=0` | Keep CPU RoPE while Attention is enabled, for staged comparisons |
| `GGML_KUNLUN_ROPE=1` or unset | Enable supported RoPE with Attention |

Invalid values disable the affected feature and are logged. Set configuration before context creation and keep it fixed. The `tg` profile is based on query count, so a one-token PP tail can also use the native route. It is independent of `GGML_KUNLUN_SWIGLU=0/1/pp`. Compare each Attention mode with the same SwiGLU setting; do not attribute a combined change to Attention alone.

Full native PP can be slower than CPU Attention because KL1 CL F32 arithmetic is slower than host SIMD at these matrix shapes. The `tg` profile preserves the existing PP route. TG benefit depends on model, context and split: GQA models have less KV readback to remove, while the query-head arithmetic remains substantial. Do not make a universal speed or quality claim.

## Graph and placement

The normal graph is:

```
Q/K RoPE -> F16 SET_ROWS KV update -> KQ MUL_MAT
 -> scale/mask SOFT_MAX -> KQV MUL_MAT -> permute/CONT -> output projection
```

The backend accepts validated non-quantized4D/view matrix operations before the legacy contiguous-matrix check. K/V views stay in their original F16 cache. The library handles stride-aware gather, half conversion and F32 accumulation. Attention output is made contiguous on device and passed to the existing Q4 projection. Regular non-view contiguous2D F32/F16 linear weights retain their existing path.

TinyLlama uses Q32/KV4 and head dimension64. Llama2 7B uses Q32/KV32 and head dimension128. Supported outer strides can include cache padding and head permutations. Query batches above256 and unsupported data/layout/mask/rotary contracts retain CPU placement. Library/API bounds are documented in the library Attention guide. Sinks, BF16, incompatible inner strides, frequency-factor tensors and YaRN extrapolation are not silently reinterpreted.

Layer Split executes each layer's Attention on its owning PD. Row Split uses the existing gathered activations and owning-PD KV cache; it does not distribute Attention heads across PDs. Device/normal-buffer ownership is checked at execution. Partial supported/unsupported graphs can still incur scheduler copies; validate unusual architectures separately.

The64 MiB per-input F32 pack limit also restricts active sequence length. With the default half-input rule, Llama2 7B's32 KV heads of width128 limit its active K/V sequence to4096. TinyLlama's4 KV heads of width64 fit longer sequences, but its KQV probability pack also requires `S * M * Q_heads * batch *4 <=64 MiB`; at M128 and Q32 this permits S<=4096. The maximum sequence contract16384 is therefore not a promise for every model/query shape. Context capacity and active sequence length differ. Workspace rejection is safe, but the current scheduler can place other supported stages on GPU and copy between stages; no whole-graph fallback planner is provided for every resource limit.

RoPE reads only I32 position IDs for its bounded range check. It does not read Q/K activations or KV payloads to the CPU. Extreme rotary positions use the library's explicit host-trig fallback. Attention core KQ/Softmax/KQV remains on device for supported shapes.

## Diagnostics and correctness

Runtime logs report per-PD actual calls for KQ/KQV, Softmax, CONT and RoPE, phase totals, pack calls, explicit HBM/L3 workspace and host trig fallbacks. `GGML_KUNLUN_DIAGNOSTICS=1` records bounded placement reasons, not CPU execution counts.

Use `K200_TRACE_OPS=1` only in separate diagnostic runs. It synchronizes each observed node and changes performance. `K200_TRACE_LAYOUT=1` with that callback prints tensor extents, byte strides, types, buffers and operation parameters. Ordinary measurements must disable both trace/capture and diagnostic preload libraries.

Unit tests compare each stage and the complete graph against the GGML CPU backend. Floating reduction order can change logits through existing Row A8 thresholds even when per-operation error is small. Compare fixed-token KL/perplexity, greedy generation and KV state save/restore; bitwise model equivalence is not promised.

## Build and execution

```
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S . -B build -G Ninja -DGGML_KUNLUN=ON -DLLAMA_BUILD_EXAMPLES=ON
cmake --build build -j4 --target kunlun-attention-test kunlun-llm-smoke kunlun-quality-eval
GGML_KUNLUN_ATTENTION=1 build/bin/kunlun-attention-test 0
GGML_KUNLUN_ATTENTION=1 build/bin/kunlun-attention-test 1
OMP_WAIT_POLICY=PASSIVE GGML_KUNLUN_COMMON_Q4=1 \
 GGML_KUNLUN_ATTENTION=tg GGML_KUNLUN_SWIGLU=0 KUNLUN_BROADCAST_VECTOR=1 \
 K200_CTX=2048 K200_BATCH=128 K200_UBATCH=128 K200_IGNORE_EOS=1 \
 build/bin/kunlun-llm-smoke /path/model-q4_0.gguf none 32 logits.bin
```

Wrap device commands with the existing thermal guard and fail-stop runner. `none`, `layer`, `row` select existing splits. Unset Attention or use0 for the old comparison. Use1 to measure full native PP/TG. Keep Q4/SwiGLU/context/batch/threads/KV/clock conditions identical. Attention does not repack or upload Q4 weights when PP changes to TG.

## Bundled Attention acceleration (library 0.6.3)

The bundled library includes SDNN matrix multiplication, local-memory CL Softmax and DMA preprocessing. Resident Attention remains opt-in: `GGML_KUNLUN_ATTENTION=1` enables supported PP/TG, `tg` keeps multi-query PP on CPU, and unset/0 preserves CPU Attention. Q4 and SwiGLU settings remain independent.

With library 0.6.3, `KUNLUN_ATTENTION_MATMUL=auto/sdnn/cl` selects measured dispatch, forced supported SDNN or CL reference. The SDNN interface converts half inputs to symmetric INT16 MAC operands with visible F32 accumulation/output; it is not a claim of IEEE half MAC arithmetic. Auto retains CL for the measured long-context GQA KQ GEMV case. `KUNLUN_ATTENTION_SOFTMAX=legacy` and `KUNLUN_ATTENTION_PREP=legacy` retain the corresponding comparison paths; their defaults use validated fast Softmax and shape-aware preprocessing. Unsupported SDNN shapes retain CL fallback.

See the [library Attention guide](../../vendor/KunlunK200/docs/attention.md) for exact layout, rounding, workspace, lifetime and dispatch contracts. The current backend requires library 0.7.0 or later for system-library builds; the original separate-path acceleration controls were introduced in 0.6.3. Omit timing, trace, verification and profiling flags from normal performance measurements. Model/context-dependent regressions and CPU-reference numerical differences remain possible; the explicit backend opt-in is unchanged.

## Experimental FlashAttention (library 0.7.0)

Use `GGML_KUNLUN_ATTENTION=1 GGML_KUNLUN_FLASH=1 KUNLUN_FLASH_KV=1` with `-fa on` to enable the shared-L3 block path. The existing separate Attention path remains the default with FA disabled. See the [library configuration and contracts](../../vendor/KunlunK200/docs/flash-attention.md) and [Japanese hardware evaluation](../../vendor/KunlunK200/docs/reports/flash-attention-20261026/REPORT.ja.md). KV-update optimization is separately controlled; enable it on both comparison paths to isolate the fused-Attention effect.
