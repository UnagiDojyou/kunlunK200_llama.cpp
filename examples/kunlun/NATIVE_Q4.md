# Native Q4_0 on K200

This backend requires KunlunK200 0.3. The existing submodule and installed-package build modes are supported. Default inference behavior is unchanged.

## Selection

Set GGML_KUNLUN_NATIVE_Q4=1 before loading a model. Native storage is selected for contiguous, non-view Q4_0 matrices accepted by native_q4_layout_supported: K divisible by 32, 32 <= K <= 16384, N > 0, K*N <= 90,000,000. This is the conservative validated envelope, not a claim that larger shapes are physically impossible.

| Native setting | REQUANT_I8 | Eligible Q4_0 weights | Other Q4_0 weights |
|---|---|---|---|
| unset / 0 | unset / 0 | Existing direct Q4 | Existing direct Q4 |
| unset / 0 | 1 | Existing CDNN rowwise INT8 | Existing policy |
| 1 | unset / 0 | Native Q4 | Existing direct Q4 |
| 1 | 1 | Native Q4 | Existing CDNN if eligible, otherwise existing fallback |

Only the literal native value 1 enables it. Keep settings fixed for the model lifetime. Native overrides REQUANT_I8 for eligible tensors at allocation time; it never replaces F32/F16 weights. KUNLUN_GEMV and GGML_KUNLUN_BASELINE continue to affect their existing legacy paths. A 7B output matrix can exceed the conservative native capacity, so native+REQUANT_I8=1 is a mixed profile. Check the logged path counts; do not describe every matrix as native merely because the option is set.

The backend still accepts contiguous F32 activations and F32 outputs for MUL_MAT. The independent library additionally supports F16 input. F16 input tensors, arbitrary strides, higher-dimensional multiplication, and weight views retain the existing CPU placement rules. F16 KV cache updates are unchanged.

## Dispatch and lifetime

The library owns layout conversion, activation quantization, SD/CL math, kernel loading and scratch. GGML only keeps the original Q4 bytes for readback/views/CPU fallback and delegates each device partition to upload_native_q4 on upload or weight modification. There is no per-GEMV repacking or weight/activation CPU readback in the native path. Each partition has one compressed resident weight copy; row padding may increase storage for irregular partitions. The host shadow is additional CPU memory and is not an HBM saving.

Single PD and Layer Split use the corresponding PD workspace. Row Split keeps the existing P2P transfer and output gather. Each actual partition is dispatched independently: batch=1 and rows divisible by 256 use N64; rows divisible by 128 use N32; other row counts, batches, or SD alignment use the library's resident-layout CL fallback. CL uses the original Q4 values and is usually more accurate than the activation approximation, but can be much slower. No CPU activation quantization or fallback INT8 weight cache is created for a native tensor. Thus batched prompt processing can regress substantially; choose batch/ubatch=1 when evaluating the native GEMV path, and report that setting.

Runtime errors retain fail-fast behavior. Both PD scratch lifetimes, L3 sharing and final waits are controlled by the library. Existing thermal guards, persistent stop latches and safe recovery remain required. No driver, firmware, clock, reset, or OS policy is changed.

## Build

~~~bash
source /home/unagidojyou/kunlun-k200/env.sh
cmake -S repos/llama.cpp -B build/native-integration-llama -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_KUNLUN=ON -DKUNLUN_USE_SYSTEM_LIBRARY=OFF \
  -DLLAMA_BUILD_EXAMPLES=ON -DLLAMA_BUILD_TESTS=ON -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_APP=ON
cmake --build build/native-integration-llama -j4 --target llama-app llama-bench kunlun-llm-smoke kunlun-gemv-bench kunlun-backend-contract
# Installed package: install the library to a separate prefix, then configure
# -DKUNLUN_USE_SYSTEM_LIBRARY=ON -DCMAKE_PREFIX_PATH=/path/to/prefix
~~~

The application target requires the existing SERVER/APP configuration; disabling SERVER while building llama-app can leave its cli/server implementation link dependencies missing.

## Run each arithmetic profile

Use the same model, prompt, clocks, context, batch, ubatch and KV settings for comparisons. The examples use batch=ubatch=1 to measure the B1 native path. Default K/V storage remains F16 on XPU where supported.

~~~bash
source /home/unagidojyou/kunlun-k200/env.sh
export KUNLUN_LOG_DIR=$K200_WORKSPACE/results/native-q4-local
mkdir -p "$KUNLUN_LOG_DIR"
MODEL=$K200_WORKSPACE/models/llama2-7b-q4_0.gguf
BIN=$K200_WORKSPACE/build/native-integration-llama/bin
# Original direct Q4
GGML_KUNLUN_NATIVE_Q4=0 GGML_KUNLUN_REQUANT_I8=0 k200-thermal-guard --name direct --seconds 900 -- "$BIN/llama" completion -m "$MODEL" -ngl 99 -dev KUNLUN0 -sm none -c 256 -b 1 -ub 1 -n 32 --log-verbosity 3 -fa off --no-warmup --temp 0 --seed 1 --simple-io -no-cnv -p 'The capital of France is'
# Existing CDNN
GGML_KUNLUN_NATIVE_Q4=0 GGML_KUNLUN_REQUANT_I8=1 k200-thermal-guard --name cdnn --seconds 900 -- "$BIN/llama" completion -m "$MODEL" -ngl 99 -dev KUNLUN0 -sm none -c 256 -b 1 -ub 1 -n 32 --log-verbosity 3 -fa off --no-warmup --temp 0 --seed 1 --simple-io -no-cnv -p 'The capital of France is'
# Native, with existing CDNN for tensors outside the native storage envelope
GGML_KUNLUN_NATIVE_Q4=1 GGML_KUNLUN_REQUANT_I8=1 k200-thermal-guard --name native --seconds 900 -- "$BIN/llama" completion -m "$MODEL" -ngl 99 -dev KUNLUN0 -sm none -c 256 -b 1 -ub 1 -n 32 --log-verbosity 3 -fa off --no-warmup --temp 0 --seed 1 --simple-io -no-cnv -p 'The capital of France is'
# CPU
k200-thermal-guard --name cpu --seconds 900 -- "$BIN/llama" completion -m "$MODEL" -ngl 0 -c 256 -b 1 -ub 1 -n 32 --log-verbosity 3 -fa off --no-warmup --temp 0 --seed 1 --simple-io -no-cnv -p 'The capital of France is'
~~~

For Layer Split replace -dev KUNLUN0 -sm none with -dev KUNLUN0,KUNLUN1 -sm layer -ts 1,1. For Row Split use -sm row -ts 1,1. Set REQUANT_I8=0 with native enabled to remove CDNN weight requantization entirely; out-of-envelope Q4 uses the existing direct kernel and may be much slower. For sampled 32-token checks use --temp 0.8 --ignore-eos -n 32. Native path logs and process totals are written to stderr even when the CLI suppresses GGML info. --log-verbosity 3 enables the other model/setup logs. Counts include native_sd/native_cl/cdnn/direct_q4 per PD; count across contexts rather than summing shared L3 byte figures.

kunlun-gemv-bench retains CPU verification, chunked upload, clear and view checks; native checks use a stricter NMSE gate and a native-specific label. kunlun-llm-smoke supports fixed teacher tokens, logits dumps, K200_BATCH/K200_UBATCH/K200_CTX, K200_WARMUP and K200_REPEATS. Its optional existing API profiler integration separates model/context/pp/tg/warmup phases and reports CPU user/system time. Phase times include their waits; do not add inclusive wait time again. Whole-model CPU/KV operations still generate transfers even when the native GEMV itself has none.

## Validation snapshot (2026-10-19)

Both available Q4_0 models passed CPU teacher-logit comparisons at 8 positions, with 8/8 matching greedy choices in Single PD, Layer Split and Row Split. They also passed 32 sampled tokens at temperature 0.8/seed 1 in all three layouts. Existing CDNN generation, direct Q4, REQUANT_I8=0 native, PD1-only generation, and F16 XPU KV cache were verified. Each model has a Q6_K output tensor (GGML type 14); it remains on the existing CPU path, as do unsupported attention/other operators. Q4_K/Q6_K arithmetic was not added.

Below: context 256, batch=ubatch=1, prompt The capital of France is (6 tokens), fixed CPU teacher tokens, FA off, 6 CPU threads, F16 KV, 900 MHz. Timings are medians of 3 iterations after one warmup; the existing API profiler was enabled for both XPU profiles. These settings are not a claim of optimal CPU/CDNN throughput. Native is slower than CDNN and the measured 6-thread CPU baseline; it preserves compressed weight memory and reduces the measured Q4 arithmetic/CPU-logit differences. Defaults remain unchanged.

| Model/profile/layout | PP token/s | TG token/s | Logit NMSE vs CPU |
|---|---:|---:|---:|
| tinyllama-native-none | 20.13 | 20.00 | 2.334e-4 |
| tinyllama-cdnn-none | 28.83 | 28.87 | 4.057e-4 |
| tinyllama-native-layer | 20.03 | 20.51 | 2.334e-4 |
| tinyllama-cdnn-layer | 24.02 | 24.88 | 4.057e-4 |
| tinyllama-native-row | 12.87 | 12.92 | 2.334e-4 |
| tinyllama-cdnn-row | 23.18 | 22.19 | 4.057e-4 |
| llama2-7b-native-none | 5.18 | 5.21 | 1.237e-4 |
| llama2-7b-cdnn-none | 6.36 | 6.35 | 2.612e-4 |
| llama2-7b-native-layer | 5.25 | 5.24 | 1.237e-4 |
| llama2-7b-cdnn-layer | 6.38 | 6.28 | 2.612e-4 |
| llama2-7b-native-row | 3.90 | 3.89 | 1.237e-4 |
| llama2-7b-cdnn-row | 6.14 | 6.10 | 2.612e-4 |

Single-PD API HBM peak: TinyLlama native 525.76 MiB vs CDNN 995.55 MiB; Llama 2 7B native 3603.32 MiB vs CDNN 6374.57 MiB. Native adds 1.375/5.375 MiB L3 respectively. CPU shadows are additional host memory. The CPU logit reference is an optimized GGML CPU implementation, distinct from the exact decoded-weight FP64 matrix reference; short teacher checks do not establish corpus perplexity or long-context quality.

Batched prompt caveat: TinyLlama batch=ubatch=8 processed the 6-token prompt in 11.272 s with native CL fallback versus 0.095 s for CDNN. This regression is retained in the evidence and is why the native setting is explicit and B1-focused. N32 improves common N128 row partitions, but arbitrary row tails and batched CL remain optimization targets. Larger validated native envelopes, dual16 weight routing, shared activation quantization, and asynchronous per-PD scheduling remain future work.

Llama 2 7B batch=ubatch=8 processed the same 6-token prompt in 68.088 s with native CL fallback versus 0.261 s with CDNN (single measurements, 2 teacher positions). Native/CL batching is a major limitation. Compared at the same two teacher positions, B8 versus B1 native logit NMSE was 7.46e-6 for TinyLlama and 9.35e-6 for 7B, with unchanged greedy choices; this also includes graph batching differences.

Transfer profiling labels inter-iteration KV clearing as its own clear phase, separate from steady pp/tg phases. Whole-model readback remains large because attention and Q6_K output/other unsupported operations use CPU. Do not interpret zero native-GEMV API copies as zero whole-model copies.
