# Kunlun K200 backend (experimental)

This backend uses the independent KunlunK200 C++ library in the Git submodule `vendor/KunlunK200`, through `KunlunK200::kunlunk200`. That library owns device allocation, transfers, P2P, compiled kernels and math operations. This repository owns GGML tensor adaptation, KV-cache policy and split scheduling. Model weights and F16 KV caches reside in XPU HBM. Attention and unsupported operations still run on CPU.

## Build

Initialize the submodule first. Its own README describes the external XRE 4.33, KL1 XDNN and XTDK 2.7.98.2 dependencies. The default CMake build builds the library from that submodule. There are no SDK headers or device kernel sources in the adapter itself.

```sh
git submodule update --init --recursive
export XRE=/path/to/xpu-4.33.0
export XDNN=/path/to/xdnn
export XTDK_DIR=/path/to/xtdk-llvm15-ubuntu1604_x86_64
cmake -S . -B build -DGGML_KUNLUN=ON \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=ON
cmake --build build -j4 --target llama-app llama-bench test-backend-ops
cmake --build build -j4 --target kunlun-gemv-bench kunlun-llm-smoke kunlun-backend-contract
```

The submodule URL points to the library fork `UnagiDojyou/kunlunK200_lib`. Publish the library commit before publishing a llama.cpp commit that references it. For an existing installation, `-DKUNLUN_USE_SYSTEM_LIBRARY=ON -DCMAKE_PREFIX_PATH=/path/to/install` is also supported; that path does not need the submodule or XTDK during the adapter build.

## Execution modes

- Single PD: `-dev KUNLUN0 -sm none -ngl 99`
- Layer split: `-dev KUNLUN0,KUNLUN1 -sm layer -ts 1,1 -ngl 99`
- Weight row split: `-dev KUNLUN0,KUNLUN1 -sm row -ts 1,1 -ngl 99`

Row split distributes each matrix's output rows between the two PDs, sends activations with P2P, and gathers outputs with P2P. The main PD must be PD0. This is weight tensor sharding through llama.cpp's row mode; the newer `-sm tensor` graph/KV sharding mode is not implemented.

MUL_MAT supports contiguous 2D weight matrices in F32, F16, or Q4_0, F32 activations, K divisible by 32, and K up to 16384. Multiple activation columns are supported. ADD, MUL, RMS_NORM, SILU and zero-bias SCALE support selected contiguous F32 shapes with row width up to 16384; wider operations fall back to CPU. SET_ROWS supports F32/F16 sources into F16 caches, including strided sources and I32/I64 indices. Cache updates currently use CPU staging and conversion; transposed V updates are merged into bulk transfers. Use `-nkvo` only when CPU KV storage is desired. Unsupported attention and other operations run on CPU. This is hybrid inference.

## Precision profiles

With `GGML_KUNLUN_REQUANT_I8=0` or unset, the SIMD kernel reads standard Q4_0 blocks directly and uses F32 activations. This is the original, slower path.

With `GGML_KUNLUN_REQUANT_I8=1`, each Q4_0 matrix is requantized to rowwise INT8 once at load time. Original Q4 bytes remain in host memory for tensor readback. XPU weight storage grows to about 1.8 times the original size. Activations are represented as two INT8 components, and two CDNN GEMVs plus a residual sum reduce input quantization error. This is additional quantization, so logits can differ. The inputs are staged on CPU for this conversion; the matrix calculations run on XPU CDNN engines. This path is distinct from the vendor CDNN INT4 layout.

To use the two-pass INT8 profile with XPU KV storage:

```sh
export GGML_KUNLUN_REQUANT_I8=1
k200-thermal-guard --name inference --seconds 300 -- \
  build/bin/llama completion -m /path/to/model-q4_0.gguf \
  -ngl 99 -dev KUNLUN0 -sm none -c 256 -b 8 -ub 8 -n 32 \
  --temp 0 --seed 1 --simple-io -no-cnv -p 'The capital of France is'
```

## Validation status

The original standalone GEMV reproduced RD_OVER_IBUF on a reset device. Reserving an aligned trailing instruction-fetch region fixed the reproduced case on both PDs. Custom SIMD kernels use the CL launch API. The loader also caps individual buffers at 1 GiB so the 7B packed weights do not require an allocation across KL1 HBM regions. In-place RMS_NORM uses a stable input copy.

All 116 supported SET_ROWS and elementwise cases passed CPU comparison. Direct F32/F16/Q4 GEMV passed on both PDs. Packed weights passed partial-write, byte-readback and view checks. K=N=4096 GEMV measured 21.47 ms on the original path and 0.358 ms on the two-pass CDNN path; quantization error is evaluated separately from the original strict tolerance.

TinyLlama and Llama 2 7B were checked in single-PD, layer-split and row-split modes with XPU KV. Fixed-prefix CPU comparisons check full-vocabulary logits and KL divergence; this is a small quality sample, not a full perplexity evaluation. Llama 2 7B decoding improved from about 0.12 to about 5 token/s. An unforced 32-token CLI run completed at 5.26 token/s. Peak observed temperature was 56 C. Detailed results are in the workspace `llm.md`.

Use the external temperature guard and stop at 80 C. A kernel error is also a stop condition. Do not continue launching against a PD in state E. Firmware and PLL changes are not needed.

After all live XPU clients have exited, `sudo /usr/local/xpu-4.33.0/tools/soft_reset 0` resets the whole card, including both PDs. Check both states and start fresh application processes. This is distinct from `tools/kunlun1/soft_rst`; driver reload is not required after a successful driver-managed reset. The separate driver fork provides the guarded reset helper.

`GGML_KUNLUN_BASELINE=1` selects small DMA tiles and scalar nibble conversion for performance comparison. The default uses larger tiles and a local conversion table. The backend prints per-PD matmul and P2P counters when freed.

## Reproduce checks

`examples/kunlun` contains GGML buffer/GEMV and LLM integration checks. With `k200-thermal-guard` in PATH:

```bash
LLAMA_BIN_DIR="$PWD/build/bin" MODEL=/path/to/model-q4_0.gguf \
  KUNLUN_LOG_DIR=/path/outside/repos/results \
  bash examples/kunlun/verify.sh
```

This checks operation placement at the width limit, compares supported ops to the CPU backend, and checks exact and requantized GEMV, including partial writes, split batches and buffer clearing with unequal or single-PD weight splits. It creates a CPU token prefix and compares full-vocabulary logits in single-PD, layer and row modes with XPU KV caches. It does not download models. The faster profile has separate error tolerances and a KL bound. These short prompts do not replace a corpus-level quality evaluation.

## Batched row output copies

For batch > 1, row-split results use device scatter and one contiguous P2P output transfer per matrix. Scratch storage is reused. `GGML_KUNLUN_ROW_COPY=legacy` selects the previous per-column D2D/P2P path for comparisons. The batch=1 path is unchanged.

## Row split device ownership

`-dev KUNLUN0,KUNLUN1 -sm row -ts 1,1` splits every supported layer matrix across both physical PDs, including layers whose output belongs to PD1. The loader still assigns activation/KV ownership by layer; this does not change Layer Split or implement tensor-graph parallelism. The historical `-dev KUNLUN0 -sm row -ts 1,1` path remains supported. `GGML_KUNLUN_ROW_SPLIT=legacy` restores the former PD0-only split-buffer selection for comparisons. Split fractions describe physical PD0/PD1 weight rows.

## Device KV updates

F16 SET_ROWS updates with supported aligned strides and non-overlapping inputs use the indexed device kernel. The adapter checks every index; duplicates, aliasing and unsupported layouts retain the CPU path. A proven contiguous V-cache update batches token runs per feature. This removes the full-cache D2H/read-modify/H2D cycle for updates, while CPU Attention still transfers its inputs. `GGML_KUNLUN_SET_ROWS=legacy` restores the reference path. `GGML_KUNLUN_CPU_KV=1` keeps the CPU cache comparison mode.

## RMS normalization

The library selects a two-stage FP32 parallel reduction for small batches. `KUNLUN_RMS_NORM=legacy` permits numerical and performance comparisons with the previous reduction. Supported in-place tensors retain their input until normalization completes.
