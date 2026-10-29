# Kunlun K200 fork

Repository: [UnagiDojyou/kunlunK200_llama.cpp](https://github.com/UnagiDojyou/kunlunK200_llama.cpp).

This fork adds the experimental K200 GGML backend. The independent userspace device library is the Git submodule [`vendor/KunlunK200`](vendor/KunlunK200). Driver source belongs to the separate `kunlun_k200` fork.

Initialize with `git submodule update --init --recursive`, set external `XRE`, `XDNN` and `XTDK_DIR`, then configure `-DGGML_KUNLUN=ON`. CMake builds the submodule automatically and links the backend to `KunlunK200::kunlunk200`. See [the backend guide](ggml/src/ggml-kunlun/README.md) for build commands, precision profiles, XPU KV cache, single-PD and layer/row split execution, and hardware tests.

The submodule URL points to `UnagiDojyou/kunlunK200_lib`. Publish its referenced library commit before the parent llama.cpp commit. SDKs, models, build products and test logs belong outside both source trees.

## Q4 Activation sharing (opt-in, library 0.7.7)

Set GGML_KUNLUN_COMMON_Q4=1 and GGML_KUNLUN_Q4_ACT_SHARE=1 before loading the model to share graph-local, per-PD activation snapshots across QKV and FFN gate/up. The default remains disabled. See [conditions, fallback, memory and diagnostics](docs/backend/kunlun-q4-activation-sharing.md). Attention remains the validated 0.7.6 implementation.

## Q4 MAC/DMA stream GEMV (0.7.8)

KUNLUN_Q4_MAC_DMA=1 opts into the library stream GEMV for validated common-Q4 shapes. The default is disabled. Activation sharing is independently controlled by GGML_KUNLUN_Q4_ACT_SHARE. See [configuration and diagnostic counters](docs/backend/kunlun-q4-mac-dma.md). Existing GEMM, Attention, SwiGLU, SET_ROWS and fallback paths remain unchanged.
