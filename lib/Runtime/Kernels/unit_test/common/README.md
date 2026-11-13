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


## Native LUT ranking (`ut-rank-v3`)

LUT refreshes rank candidates with the UT leaves themselves, so tables and
tests share one timer and its v3 presets. Decision loss is the chosen entry's
time divided by the best time in an independent sequential `confirm` of the
final three candidates plus the default. On gfx1151 (2026-11-13; 32 MatMulNBits
and 25 GQA shapes) it was p50/p95/max 1.000/1.025/1.061 for MatMulNBits and
1.000/1.014/1.017 for GQA. These figures replay the recorded measurements with
the v3 decision rule below, which was revised after those results, so they are
not an independent qualification. A single `screen` pass per candidate scored
p95/max 1.049/1.082 (MatMulNBits) and 1.027/1.066 (GQA) on an earlier sample.
Near-tied finalists can swap between runs at little cost. The largest loss is a
shape whose ranking stages both measured the default fastest while the
sequential reference found a config 5.7% faster. Group interleaving still
leaves repeatable schedule differences for a few candidates. ADLX on this APU
exposes no manual clock, power or preset tuning, so clocks cannot be pinned.

Build the eight MatMulNBits or four GQA leaves with their existing Makefiles:
`make build RANK=1 BUILD_DIR=out_rank HIP_SDK=<sdk>`. This standalone variant
uses the unchanged production kernel TU and an empty LUT. It exposes the
existing configuration descriptors to the UT; ordinary builds keep their usual
behavior. MatMulNBits candidates enter `hip_matmul_nbits`, including casts,
zero-point conversion, repacking, dequantization and GEMM. GQA uses the existing
configured APIs; INT8 prefill times both KV dequantization launches and attention
in each complete call. It excludes KV append and rotary work outside that entry.

Supply `HIPDNN_UT_RANK=<csv>` with the exact header:

```csv
leaf,shape,priority,config
matmul_nbits/Xfp16_Wu4_Yfp16,M16_N2048_K2560_gs128,0,default
matmul_nbits/Xfp16_Wu4_Yfp16,M16_N2048_K2560_gs128,0,id=12
```

Repeated leaf/shape rows form one candidate set. `default` is mandatory and is
captured from the actual no-LUT final selection, then placed at index zero.
Final launches are deduplicated before correctness checks or timing. Config IDs
come from `HIPDNN_UT_RANK_CATALOG=1`, not from a separate sweep driver. Supported
shape strings are the native leaves' strings: MatMulNBits `M_N_K_gs` with an
optional `_noz`; GQA `B_H_G_D_len` for decode or `B_H_G_D_sq` for prefill. FP16
GQA also accepts its native window/sink/smooth fields. FP16 prefill requires
`_past<length>`, including `_past0`; INT8 prefill has no past suffix. Ranking
currently excludes padding cases. M-blocked MatMulNBits has no `tile_m` in the
existing LUT schema and therefore participates as a default singleton.

Every candidate must pass the owning leaf's CPU reference first. Screen all
candidates with the unchanged v3 preset. Keep at most four candidates within 5%
of the fastest, plus default. Standard and confirm use the same batch runner
as an ordinary UT measurement. Group interleaving measures contiguous batches
for each candidate within common balanced groups, rotating each group's first
candidate. Event batches retain the current profile's call cap. If a candidate
needs full-duration continuation, all continue and retain valid earlier batches.
The v3 warmup, call/time goals, floors, spread and invalid retry limits are unchanged.

Sort every candidate by the median of its five group ratios to the same default.
Pairwise ratios are diagnostics and never serve as the sorting comparator.
Confirm once for priority, a nondefault entry boundary (score 0.97..0.99 or
range crossing 0.98), winner/runner-up range crossing 1, or either competitor
being nonconverged. A default singleton requires no confirm.

The order between the winner and the runner-up is unresolved when their
range crosses 1 or their pairwise median contradicts the scalar order; the
record then reports `decision_status: winner_unresolved`. That status is
diagnostic only. Insert the ranked winner whenever it is nondefault and its
median ratio to the default is below 0.98; otherwise keep the default.
`selected` and `insert_nondefault` state the resulting LUT choice. One slow
group among five cannot veto an otherwise clear win, and a near-tied runner-up
costs little. Nonconvergence prompts confirm. Ratio ranges describe five
groups and are not confidence intervals. `ut-rank-v2` records, which fell back
to the default whenever the order was unresolved, are still validated under
their original rule.

A measurement whose invalid event durations exceed the timer allowance is
repeated once. A nondefault screen candidate that fails again is dropped and
listed in that phase's `excluded`; the default must measure.

`UT_RANK` includes candidate CPU results and final geometry; screen, standard
and optional confirm groups and ratios versus default/best; winner/runner-up;
entry decision, tie/obvious/converged flags, confirm trigger, per-phase GPU/pilot
and wall costs, and optional frequency. Existing `UT_TIMING` and
`UT_TIMING_BINDING` records remain available for full environment/build binding.

`HIPDNN_UT_RANK_REFERENCE=1` measures supplied candidates independently and
sequentially with confirm. Its winner uses absolute median times because its
groups do not share a drift history. The tooling freezes a MatMulNBits default
with `default;id=<original ID>`; it does not let a second runtime tune redefine
the reference default. Its index-matched group ratios are diagnostic because
sequential groups do not share a drift history; the decision uses absolute
median times. This mode validates ranking, not routine LUT refresh.

On Windows, `ADLX_INCLUDE=<GPUOpen ADLX v1.5 Include directory>` enables optional
System32 ADLX loading. Measurements run without SDK headers or driver support.
Clock, temperature and power queries occur after completed event intervals,
at most every 250 ms, excluding warmup. Frequency includes average/minimum/peak,
reference MHz/basis (SDK range, or observed peak when unavailable) and a below-75% diagnostic; unavailable fields are null and
`frequency_status` explains why. Short measurements may obtain only one sample;
driver telemetry cadence and the diagnostic do not establish clock causality.
During ranking, telemetry covers the whole phase and is shared by that phase's
candidate UT_TIMING records; it is not a per-candidate clock measurement.
`HIPDNN_UT_ADLX=0` disables observation for an overhead comparison. No clocks or
driver settings are changed.

Ranking leaves may reuse a previously compiled adapter for the same op with
`RANK_KERNEL_OBJ=<object>`. The caller must prove matching source/include
closure, compiler/SDK, ISA and kernel flags. Other builds compile their own TU.
GPU lease wrappers belong around each native process; CPU candidate generation,
JSON/FB writing, analysis and file transfer do not retain GPU ownership.
Catalog enumeration is a CPU metadata phase and does not require a GPU lease.

### Existing JSON table inputs and reports

Each operator's `update_lut.py rank-seed --from-lut <table.json> --out <csv>`
converts table points to default + old geometry candidates. JSON is the only
input format. Optional leaf/dtype/phase/key filters and seeded sampling run
before ranking. Unsupported, filtered and unsampled keys appear in the
skipped JSON; seed metadata binds selected keys to leaf/shape/config.
`--with-catalog --catalog <log>` appends all legal configs only when requested.

`rank-report --rank-log <log> --from-lut <table.json> --out <csv> [--md <md>]`
validates native evidence and exports point identities, old/default/winner/
selected configs, paired group ratios, changes, retests, exclusions and GPU/
wall costs. Ratios involving an old candidate no longer in the final phase
use the latest shared phase, explicitly named, or are unavailable. The native
winner/default decision always comes from the final phase. Headers, weights,
fallbacks, config indices, point order and unrelated fields survive rank-build.

`make test` includes `test_ut_lut_from_table.py` (host-only, Python and C++
compiler): every representable point in both gfx1151 JSONs is seeded, backed by
synthetic validated old-config-winner evidence, and imported with full parsed
JSON equality. It also checks filtering/sampling/skips, report decisions,
geometry resolution and default deduplication.

The FP16 W4 MatMulNBits and FP16 KV GQA decode/prefill leaves support an
optional launcher handshake, `HIPDNN_UT_GPU_ADMISSION=<private-prefix>`.
After CPU input/reference preparation, case N writes `<prefix>.N.request`
and waits for `.grant` before device allocation/work. After device cleanup
and synchronization it writes `.done`, then waits for `.released` before
CPU preparation of the next case. The launcher owns host admission and its
execution lease, waits through the actual worker exit, and reconciles a failed
worker before releasing. Without this variable, ordinary leaf execution is
unchanged. This handshake runs outside the timer and adds no sampling controls.
