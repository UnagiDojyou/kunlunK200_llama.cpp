# Kunlun K200 fork

Repository: [UnagiDojyou/kunlunK200_llama.cpp](https://github.com/UnagiDojyou/kunlunK200_llama.cpp).

This fork adds the experimental K200 GGML backend. The independent userspace device library is the Git submodule [`vendor/KunlunK200`](vendor/KunlunK200). Driver source belongs to the separate `kunlun_k200` fork.

Initialize with `git submodule update --init --recursive`, set external `XRE`, `XDNN` and `XTDK_DIR`, then configure `-DGGML_KUNLUN=ON`. CMake builds the submodule automatically and links the backend to `KunlunK200::kunlunk200`. See [the backend guide](ggml/src/ggml-kunlun/README.md) for build commands, precision profiles, XPU KV cache, single-PD and layer/row split execution, and hardware tests.

The submodule URL points to `UnagiDojyou/kunlunK200_lib`. Publish its referenced library commit before the parent llama.cpp commit. SDKs, models, build products and test logs belong outside both source trees.

## Q4 Activation sharing (opt-in, library 0.7.7)

Set GGML_KUNLUN_COMMON_Q4=1 and GGML_KUNLUN_Q4_ACT_SHARE=1 before loading the model to share graph-local, per-PD activation snapshots across QKV and FFN gate/up. The default remains disabled. See [conditions, fallback, memory and diagnostics](docs/backend/kunlun-q4-activation-sharing.md). Attention remains the validated 0.7.6 implementation.

## Q4 MAC/DMA stream GEMV (0.7.8)

KUNLUN_Q4_MAC_DMA=1 opts into the library stream GEMV for validated common-Q4 shapes. The default is disabled. Activation sharing is independently controlled by GGML_KUNLUN_Q4_ACT_SHARE. See [configuration and diagnostic counters](docs/backend/kunlun-q4-mac-dma.md). Existing GEMM, Attention, SwiGLU, SET_ROWS and fallback paths remain unchanged.

PP Q4 GEMM opt-in 0.7.10: [評価レポート](docs/backend/kunlun-q4-gemm-next-20261029.md).

Large PP offload 0.7.11: [評価と実行設定](docs/backend/kunlun-large-pp-20261030.md).

## 0.7.12 optimization defaults

See [selection, shape guards and disabling options](ggml/src/ggml-kunlun/DEFAULTS.md). Activation sharing, GEMV stream/N-major and M128 Owners are automatic only within the existing common-Q4 profile and validated shapes. Common-Q4/Attention profiles and Large PP remain opt-in; supported SwiGLU is enabled by default. Explicit 0 retains the preceding path.

## 0.7.12 model-quality assessment

See [fixed-token PPL, logits, tasks and query boundaries](docs/backend/kunlun-quality-20261030.md). SwiGLU and Large PP remain opt-in; small greedy differences alone are not treated as a quality failure. Their independent quality limits and evaluation scope are documented. [Evaluator usage](examples/kunlun/QUALITY.md).

## Requested SwiGLU default update (2026-10-30)

GGML_KUNLUN_SWIGLU unset now means 1: supported PP/TG GLU uses XPU. Explicit 0 restores CPU, pp remains multi-row-only, and invalid values disable it. Existing type/shape/stride/PD/alias guards remain. KUNLUN_LARGE_PP stays unset=0. CPU/XPU logits and generation may differ; the previous quality reports are historical evidence, including the 7B PPL interaction. This change follows the user request and does not claim universal quality equivalence.

## Q6_K output SDNN opt-in

KUNLUN_Q6_K_SDNN=1 uses a lossless W8/K16-scale pack and SDNN for the validated Q6_K output shapes. Unset or 0 retains CPU. EW/L1E aggregation improves TG for the two tested models; this remains experimental. KUNLUN_Q6_K_LEGACY=1 selects the previous HBM-partial implementation. See [on-chip results](docs/backend/kunlun-q6-onchip-20261030.md). See [contracts and limitations](ggml/src/ggml-kunlun/Q6_K_SDNN.md). SwiGLU remains enabled by default; Large PP remains opt-in.

## Embedding GET_ROWS opt-in

Set GGML_KUNLUN_GET_ROWS=1 together with GGML_KUNLUN_COMMON_Q4=1 to gather supported Q4 rows on XPU. Unset or 0 keeps CPU input. No extra F16/F32 weight copy is stored. Current PP/TG changes are small, so the default is unchanged. Keep KUNLUN_Q6_K_SDNN=1 / KUNLUN_Q6_K_LEGACY=0, SwiGLU=1 and Large PP=0 for the measured configuration. See [contracts](ggml/src/ggml-kunlun/GET_ROWS.md) and [results](ggml/src/ggml-kunlun/GET_ROWS_REPORT.ja.md).

## F32 GET_ROWS opt-in

GGML_KUNLUN_GET_ROWS_F32=1 keeps supported F32 activation row selection on XPU. Unset or 0 retains CPU F32 GET_ROWS. It is independent of GGML_KUNLUN_GET_ROWS for Q4 embedding and changes no Q6/SwiGLU/Large PP defaults. Transfers and graph splits decrease; overall PP/TG speedups are not established, so it remains opt-in. See [contracts](ggml/src/ggml-kunlun/GET_ROWS_F32.md) and [evaluation](ggml/src/ggml-kunlun/GET_ROWS_F32_REPORT.ja.md).

## Pure-token input graph opt-in

GGML_KUNLUN_PRUNE_INACTIVE_MIXED=1 omits inactive mixed-input branches in supported Kunlun LLAMA text graphs. Unset or 0 retains the legacy builder. External embeddings and mixed batches keep the original path, with graph reconstruction on input-form or policy changes. Single PD can use one continuous XPU graph. It remains opt-in because PP/TG results vary, including a lower Tiny TG mean. No library/kernel/default changes are included. See [policy](docs/kunlun-prune-mixed/PRUNE_INACTIVE_MIXED.md) and [Japanese evaluation](docs/kunlun-prune-mixed/REPORT.ja.md).

## Current recommended defaults

See [DEFAULTS.md](DEFAULTS.md). Common Q4, new SDNN Q6, Q4/F32 GET_ROWS and pure-token graph specialization are enabled when unset. Large PP remains disabled. Explicit 0 preserves rollback. This table supersedes dated opt-in reports.

The [production defaults and Tensor Split prerequisites](ggml/src/ggml-kunlun/TENSOR-DEFAULTS.md) describe the two-PD paths and environment overrides.

### Llama 7B Tensor Split prefill graph construction

With two Kunlun devices, full offload, and a Llama 7B Q4_0 inference graph,
prefill batches of at least 32 tokens defer placement probes until the graph
metadata is complete. The probes share the existing Meta probe cache only
within that graph build. Explicitly set `GGML_KUNLUN_TP_PP_BUILD_PROBE_BATCH=0`
to restore immediate placement checks. Disabling
`GGML_KUNLUN_TP_GRAPH_PROBE_CACHE` also restores the immediate path.
Embedding, NextN, backend sampling, LoRA, and evaluation callbacks keep the
immediate path. This uses the existing system Kunlun library ABI.

### Two-PD Q6_K output

Kunlun 0.7.15 enables paired execution for the validated Tensor Split
Llama 7B Q4_0 single-row 4096 x 32000 raw Q6_K logits. It keeps the
terminal PD0 output contract. PD1 sends its 16000 logits to PD0 after
both devices complete. Set `GGML_KUNLUN_TP_OUTPUT_PARALLEL=0` to use
the single-PD output path. Other shapes and source layouts keep the
existing dispatch.
