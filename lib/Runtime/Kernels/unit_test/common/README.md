# Adaptive UT timing

`ut_timing_policy.h` owns policy `ut-adaptive-v3`.
`hip_adaptive_timing.h` applies it to the existing complete-op UT callback.
MatMulNBits, GQA decode/prefill and ragged W4 MatMul use this implementation.
Operators with other timing implementations are not qualified by this change.

Run the packaged tests with `hipdnn-kernel-ut.exe [--timing screen|standard|confirm]`,
or use `make test TIMING=<preset> HIP_SDK=<sdk>` at the unit-test root. Standalone
leaves read `HIPDNN_UT_TIMING=screen|standard|confirm` (default `standard`). This setting changes
sampling, not shapes, correctness checks, dispatch or config selection. Legacy
leaf `--iters` arguments no longer select the measured launch count; actual
counts are in `UT_TIMING` records.

Excluded warm-up batches first run until the preset's warm-up GPU time (at least
one complete call); they also estimate the call cost. Each measurement then uses
five groups. A first round fills every group until it
reaches its share of the call goal (once it also has its share of the GPU-time
floor) or its share of the GPU-time goal, whichever comes first, with at least
one complete call per group. The measurement ends there when the spread of the
five group means, (max - min) / median, is within the preset's limit. Otherwise
every group continues, keeping its first-round batches, until it holds its share
of the GPU-time goal and at least two complete calls (ten in total).

| Preset | Warm-up | Call goal | GPU-time floor | GPU-time goal (soft) | Spread limit |
|---|---:|---:|---:|---:|---:|
| `screen` | 5 ms | 100 | 1 ms | 100 ms (200 ms) | none |
| `standard` (default) | 10 ms | 1,000 | 10 ms | 1,000 ms (2,000 ms) | 3% |
| `confirm` | 25 ms | 10,000 | 100 ms | 10,000 ms (20,000 ms) | 1% |

Short calls therefore stop at the call goal, mid-size calls at the GPU-time
goal, and calls longer than a group's share of the goal after five calls when
those agree, or after ten when they do not. A noisy short call keeps the full
GPU-time goal. A measurement whose spread still exceeds the limit at the full goal
is kept and reported with `converged: false`; consumers decide whether to rerun
or confirm it. `standard` is meant for routine and CI runs. Request `confirm`
explicitly for priority shapes, boundaries, suspect regressions and published
performance evidence. `screen` is for LUT config ranking.
On gfx1151 (12 MatMulNBits/GQA leaves, 2026-11-12) `standard` read about 2%
faster than `confirm` for 0.05-1 ms calls, consistent with shorter
measurements running at higher boost clocks; calls below 0.05 ms agreed within
1%. Use `confirm` when the absolute number matters.
The goals are shared across the five groups, not multiplied by five.
Each group's result is total valid GPU time divided by actual complete calls;
the reported result is the median of the five group means. Every valid batch,
including slow ones, contributes. No per-iteration sorting or trimming occurs.

The excluded warm-up estimates the selected device/config/shape's cost.
Between batches the UT predicts, from the last batch's time per call, the calls
needed to reach the group's nearer remaining goal, also satisfying the minimum
remaining calls, capped by the device profile per event pair.
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
large call can overshoot, and the minimum complete calls take precedence. Never truncate
a running call or report a projected iteration as completed. A batch with an
invalid, zero, negative or nonfinite duration is discarded and repeated with half
the calls; it never enters a group. The cell fails only when its invalid batches,
pilot included, exceed max(2, 1% of the batches attempted so far)
(`invalid_retry_limit`, `invalid_batch_fraction`), so a sporadic driver timing
glitch in a long `confirm` measurement no longer aborts the leaf, while a timer
that is invalid for a large share of batches still fails. HIP errors fail the
measurement. Spread decides only whether the first round may end the
measurement; neither spread nor soft-limit overshoot rejects data or acts as a
thermal/clock gate. No fixed steady-clock percentage
is required by this timer.

Each measurement prints a JSON `UT_TIMING` line: policy/preset, runtime device
and architecture, HIP runtime/driver versions, operand-copy count/size, batch
cap, warm-up, call goal, GPU-time floor, spread limit (`null` when none), warm-up
(`pilot_*`), valid
calls/batches/GPU time, wall time, invalid attempts, five group means, spread and
whether the first round ended the measurement (`early_stop`) and whether the
final spread is within the limit (`converged`). An operand-rotation record with one copy is not proof of cold
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
