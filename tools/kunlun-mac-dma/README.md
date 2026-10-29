# Q4 MAC/DMA measurement tools

These tools are offline or diagnostic-only. Normal inference and llama-bench do not load them. The optimized kernel is in vendor/KunlunK200; enable it with KUNLUN_Q4_MAC_DMA=1 and GGML_KUNLUN_COMMON_Q4=1. Activation sharing remains separately controlled by GGML_KUNLUN_Q4_ACT_SHARE.

Run analyze.py RESULTS_DIRECTORY on the completed guarded result directory. It keeps raw normal samples separate from diagnostic API, node, launch, copy and wait spans. ABBA samples are grouped by model, split, ubatch and flags. Mean reciprocal throughput is computed per sample. Trial samples within the same process are correlated; do not interpret them as independent process replications.

stage-analysis.py RESULTS_DIRECTORY reads the private probe logs and assembly. TICKS records are interleaved by exact kernel name and include one warmup record; do not align them by result-row index. The script discards that warmup and checks the measured sample count.

~~~sh
python3 -m unittest discover -s tools/kunlun-mac-dma -p 'test_*.py'
python3 tools/kunlun-mac-dma/analyze.py /path/to/results
python3 tools/kunlun-mac-dma/stage-analysis.py /path/to/results
~~~

The existing measurement_cache.py in tools/kunlun-activation-sharing also hashes every bin/entry in the selected kernel directory, including KUNLUN_KERNEL_DIR overrides. The host DSO hash alone cannot identify the device implementation.

## Existing-wait observer

Build against the same SDK, library headers and diagnostic GGML revision:

~~~sh
c++ -shared -fPIC -O2 -std=c++17 -DK200_DIAGNOSTIC_HOOKS \
 -I"$XRE/include" -I/path/to/kunlun/install/include \
 -Iggml/include -Iggml/src -Itools/kunlun-profiling \
 tools/kunlun-mac-dma/audit-stage.cpp -o /path/to/audit-stage.so -ldl -pthread
~~~

Only a diagnostic build should set LD_PRELOAD to this observer. It retains the existing API observer and labels existing completion waits by the preceding launch group on the same calling thread, PD and stream. It never creates an additional synchronization. Single-launch waits are distinguished from grouped waits, such as multiple activation-quantization kernels. kernel.enqueue_complete spans run from launch entry to the existing completion; kernel.existing_wait spans cover the wait itself. They overlap with API/node/backend synchronization spans and must not be added together.

Do not set K200_SERIAL_KERNELS during these measurements: that inherited diagnostic option inserts additional waits and is a different experiment. Compare observer overhead against the lightweight node observer and compare outputs bitwise. Use a separate normal executable without LD_PRELOAD or diagnostic hooks for acceptance throughput.

Device-clock probe intervals are controller issue, completion and barrier intervals, not physical engine busy cycles. Different owners run concurrently. Keep paired uninstrumented samples and never sum owner rows into kernel wall time.

Keep models, SDKs, images, raw traces and dumps outside Git. Preserve abort latches and use the existing thermal guard for hardware runs.
