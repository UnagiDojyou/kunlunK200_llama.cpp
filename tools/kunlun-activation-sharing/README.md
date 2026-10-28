# Activation-sharing evaluation helpers

These offline helpers read the separate result directory; they do not initialize the XPU or change inference dispatch.

```sh
python3 test_analysis.py
python3 test_measurement_cache.py
python3 analyze.py /path/to/results/activation-share-20261029
```

The analyzer preserves individual llama-bench samples, mean/median/sample standard deviation, prompt-prefixed TG trials, per-phase quantization counters, transfer API timings, management events and visible SDK allocation peaks. llama-bench stdout and backend stderr can interleave inside the JSON array: each complete benchmark object is decoded independently. Normal throughput and diagnostic observations stay separate by result name/series.

CPU fallback calls use worker 0 only. Its elapsed time is not the sum of all workers or a critical-path CPU estimate. Quantization, API, launch and wait spans overlap; do not add them together. Standard TG128 starts with empty KV, while the controlled smoke harness records 31 decode steps after each prefix. The latter excludes greedy selection/detokenization from process timings.

measurement_cache.py binds cached test results to command/flags, executable and host-library SHA256, and small input-fixture SHA256. Model size/mtime are included in the cache identity; full model and kernel-image SHA256 must additionally be fixed and recorded in the build manifest. Changing a shared library must invalidate old test results even when the executable path and bytes are unchanged.

Raw logs, CSV/JSON observations, models, SDKs and compiled observers belong in the external result directory and must not be committed here. See [backend configuration and ownership](../../docs/backend/kunlun-q4-activation-sharing.md).
