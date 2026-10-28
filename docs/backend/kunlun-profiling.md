# Kunlun profiling hooks

These are observational hooks for a separate diagnostic build. No device wait
is added to graph/node/CPU timing. Production builds do not define the macro;
the scope arguments and CPU wrapper are then removed by preprocessing.

Build the diagnostic executable with both
`-DCMAKE_C_FLAGS=-DK200_DIAGNOSTIC_HOOKS` and
`-DCMAKE_CXX_FLAGS=-DK200_DIAGNOSTIC_HOOKS`. This opt-in currently requires
Linux and a GCC-compatible compiler. Load an audit shared object defining
`k200_diag_event` to collect records. Without that weak callback, timing is
inactive. Normal performance measurements must use the uninstrumented build.

The callback receives category, name, tensor name, worker index/count,
monotonic begin/end nanoseconds, current-thread CPU nanoseconds and logical
bytes. CPU worker intervals overlap: sum their interval union for wall time,
and retain thread CPU sums separately. Backend/API/wait scopes are inclusive
and must not be added together. Tensor get/set bytes can be CPU-memory copies;
use actual XPU runtime copies to count H2D/D2H/P2P traffic.

The smoke harness scopes model processing, logits access and greedy/finite/
dump/detokenize work independently. Its existing per-node scheduler callback
adds synchronization and is reserved for separate layout/node diagnostics.

Keep the library, shader images, model, context, PD/split configuration,
thread count, batch, ubatch and all feature flags fixed. Calibrate the audit
overhead against normal runs with identical inputs and validate logits.
Device-clock or forced-kernel-wait probes are separate experiments and are
not normal throughput measurements. Preserve the existing thermal guard,
abort latches and device error handling.
