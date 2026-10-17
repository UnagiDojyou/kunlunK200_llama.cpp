# Kunlun K200 fork

Repository: [UnagiDojyou/kunlunK200_llama.cpp](https://github.com/UnagiDojyou/kunlunK200_llama.cpp).

This fork adds the experimental K200 GGML backend. The independent userspace device library is the Git submodule [`vendor/KunlunK200`](vendor/KunlunK200). Driver source belongs to the separate `kunlun_k200` fork.

Initialize with `git submodule update --init --recursive`, set external `XRE`, `XDNN` and `XTDK_DIR`, then configure `-DGGML_KUNLUN=ON`. CMake builds the submodule automatically and links the backend to `KunlunK200::kunlunk200`. See [the backend guide](ggml/src/ggml-kunlun/README.md) for build commands, precision profiles, XPU KV cache, single-PD and layer/row split execution, and hardware tests.

The submodule URL points to `UnagiDojyou/kunlunK200_lib`. Publish its referenced library commit before the parent llama.cpp commit. SDKs, models, build products and test logs belong outside both source trees.
