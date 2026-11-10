# Adaptive UT timing

`ut_timing_policy.h` owns policy `ut-adaptive-v2`.
`hip_adaptive_timing.h` applies it to the existing complete-op UT callback.
MatMulNBits, GQA decode/prefill and ragged W4 MatMul use this implementation.
Operators with other timing implementations are not qualified by this change.

Run the packaged tests with `hipdnn-kernel-ut.exe [--timing screen|standard|confirm]`,
or use `make test TIMING=<preset> HIP_SDK=<sdk>` at the unit-test root. Standalone
leaves read `HIPDNN_UT_TIMING=screen|standard|confirm` (default `standard`). This setting changes
sampling, not shapes, correctness checks, dispatch or config selection. Legacy
leaf `--iters` arguments no longer select the measured launch count; actual
counts are in `UT_TIMING` records.

The default `standard` mode accumulates at least 1,000 ms of valid GPU time per
measurement (2,000 ms soft) and is meant for routine and CI runs. The formal
`confirm` mode accumulates at least 10,000 ms; request it explicitly for priority
shapes, boundaries, suspect regressions and published performance evidence.
`screen` normally accumulates at least 100 ms for LUT config ranking.
Each uses five groups and at least two complete calls per group (ten total), so
calls of 100 ms or more cost the same ten calls in every mode.
The duration target is shared across the five groups, not multiplied by five.
Each group's result is total valid GPU time divided by actual complete calls;
the reported result is the median of the five group means. Every valid batch,
including slow ones, contributes. No per-iteration sorting or trimming occurs.

An excluded one-call pilot estimates the selected device/config/shape's cost.
Between batches the UT selects `ceil(remaining_group_ms / last_ms_per_call)`,
also satisfying the minimum remaining calls, capped by the device profile per event pair.
There is no total iteration cap. Every batch freezes its count, records exactly
one GPU start and stop event around back-to-back complete calls, waits for stop,
and consumes that duration before reusing the events. The original operand
rotation and stream gate remain; CPU references, allocations and untimed
rotation preparation are outside the events. Mandatory op wrapper kernels
remain inside. No host synchronization is added behind an unreleased gate.

The `normal` profile bounds a batch at 64 calls. Evidence-backed device profiles
may use 128 calls; selection requires the recorded architecture, exact runtime,
op entry, the qualified shapes and cache mode. The measured gfx1170
driver scope uses a deterministic 500 ms screen (1,000 ms soft). An unlisted/new device, runtime or entry automatically
uses `normal`. Profiles live in `ut_timing_device_profiles.h`; measured timing,
not a fixed architecture multiplier, still chooses the actual iterations.

The per-profile cap is a conservative command-retention bound, not an architecture speed factor
or proof that every arbitrary callback is safe: one complete op can enqueue
multiple kernels. Qualify actual entry points and runtime builds on each target.
Do not extend an event interval to millions of commands to meet a duration goal.
No architecture-specific iteration table is needed: a faster device receives
more calls, a slower device fewer, based on its own pilot and subsequent batches.

20,000 ms (`confirm`) / 2,000 ms (`standard`) / 200 ms (`screen`) are soft cumulative thresholds. A very
large call can overshoot, and ten complete calls take precedence. Never truncate
a running call or report a projected iteration as completed. Invalid, zero,
negative or nonfinite durations get at most two retries shared across the entire cell (including the pilot),
reducing the batch count. HIP errors fail the measurement. Spread and soft-limit overshoot are
reported, not thermal/clock rejection gates. No fixed steady-clock percentage
is required by this timer.

Each measurement prints a JSON `UT_TIMING` line: policy/preset, runtime device
and architecture, HIP runtime/driver versions, operand-copy count/size, batch
cap, pilot, valid calls/batches/GPU time, wall time, invalid attempts, five group
means and spread. An operand-rotation record with one copy is not proof of cold
caches; retain the leaf's allocation-fallback warning. Comparisons additionally
bind source/build, shape, actual route/config, cache mode and device identity.
Do not compare different devices or call a `screen` result a formal confirmation.

Five hours is a LUT-refresh planning target and warning threshold, not a timer
abort. ETA must use measured wall throughput on the actual device, including
allocation, CPU reference, launch and build costs. Screen only affected scope,
deduplicate equivalent launches, confirm selected priority/boundary/suspect
rows, and import measured winners directly. Do not repeat every config in formal
mode or silently drop mandatory coverage to meet the estimate.

The policy can be tested without HIP: `make -C common test`. Device qualification
also requires compiling/running the adapter and actual target leaves; host tests
alone do not certify any GPU architecture.

`UT_TIMING_BINDING` ties a CPU-checked CSV shape/config to its exact timing ID
and returned median. Multiple implementations in a case retain separate IDs;
unbound comparison timings are not the leaf result. An evidence-binding token
supplied by a consumer binds immutable source, SDK/compiler and executable
identities to that invocation. Untimed batch scratch preparation (ragged queue
heads) is bounded and included in wall accounting, not in GPU group totals.
