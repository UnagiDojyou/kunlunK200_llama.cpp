# Kunlun profiling observers and offline analysis

These helpers accompany the opt-in `K200_DIAGNOSTIC_HOOKS` in the GGML sources.
They do not change any kernel, weight layout, dispatch default, or library ABI.
Use a separate diagnostic build. Normal throughput must use an uninstrumented
build without `LD_PRELOAD`, TRACE, PROFILE, or forced synchronization.

Build the GGML diagnostic targets with both C and CXX flags containing
`-DK200_DIAGNOSTIC_HOOKS`. This observer interface currently targets Linux with
a GCC-compatible compiler. The API is an optional weak callback.

Three independent preloads are available:

- `audit.cpp`: runtime transfers/launch/waits, named native kernels and API
  intervals, plus GGML hooks. The 2026-10-28 study measured substantial TG
  overhead, so use it for counts/bytes/stages rather than the main wall shares.
- `head-audit.cpp`: CPU/backend/harness records only. No XPU interposition.
- `node-audit.cpp`: all GGML timing hooks, without XPU interposition. Use the
  calibrated light observer for main exclusive wall time and Amdahl estimates.

For example, from this tools directory, with the real SDK and 0.7.6 install:

```sh
c++ -shared -fPIC -O2 -std=c++17 -DK200_DIAGNOSTIC_HOOKS \
  -I../../ggml/src -I../../ggml/include \
  -I"$XRE_ROOT/include" -I"$KUNLUN_INSTALL/include" \
  audit.cpp -o audit.so -ldl -pthread
c++ -shared -fPIC -O2 -std=c++17 head-audit.cpp -o head-audit.so -pthread
c++ -shared -fPIC -O2 -std=c++17 node-audit.cpp -o node-audit.so -pthread
```

Set `K200_ENABLE_PROFILE=1`, `K200_PROFILE_PHASE=pp` or `tg` around the relevant workload, collect to
`K200_TRACE_PATH`, and invoke the harness emit hook. The saved smoke harness
already handles those boundaries. The intrusive `K200_SERIAL_KERNELS=1` mode
adds stream completion waits and is a separate experiment.

Offline analysis takes the result directory as an optional final argument:

```sh
python analyze.py /path/to/results
python analyze-extra.py /path/to/results
python analyze-light.py /path/to/results
python compare-glu.py /path/to/results
python analyze-flash.py /path/to/results
python analyze-opportunities.py /path/to/results
python analyze-configurations.py /path/to/results
python summarize-findings.py /path/to/results
python inspect-row-apis.py /path/to/results
python report-normal.py /path/to/results
python report-light.py /path/to/results
python report-diagnostic.py /path/to/results
python report-main.py /path/to/results
python plots.py /path/to/results
python plots-light.py /path/to/results
python test_interval_index.py
```

Analysis uses the Python standard library. Plotting requires Matplotlib/NumPy.
Input logs/events are produced by the saved lab controller; filenames, schema,
PP/TG units, teacher inputs, and configuration metadata must match the report.
Never run heavy analysis/compression/downloads on the benchmark host during
normal timing. Preserve the existing 80 C guard, persistent latches, ownership
checks, and error recovery policy. These observers do not replace those guards.

Wall intervals are partitioned by union/subtraction. Worker wall times overlap;
CPU core-time sums are separate. API, wait and launch intervals are inclusive
and must not be added to the exclusive wall partition. Runtime bytes are
logical requests, not total physical PCIe/NoC or kernel-internal DMA traffic.
Device clocks and serialized probes do not represent normal pipeline timing.

The dated Japanese report and compact derived tables are in
`docs/backend/kunlun-profiling-20261028/`. Raw traces/logits, detailed derived
CSVs, model/SDK/binary files and machine-specific safety controllers live in the
separate result artifacts, not Git. See the report's reproduction instructions.
