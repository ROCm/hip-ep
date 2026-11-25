<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# perf-harness: the hip-ep performance toolkit (Windows)

Find the next thing worth optimising, size it before writing kernel code, and
prove the change afterwards. Everything is Python (`hipep_perf`) except the
trace shim, which only exists in profiling builds. Design and contracts:
[docs/design/perf-toolkit.md](../../docs/design/perf-toolkit.md).

```powershell
cd tools\perf-harness
pip install -r requirements.txt
python -m hipep_perf --help
```

## Setup

| variable | meaning |
|---|---|
| `HIPEP_BIN` | **required.** Directory with `model_benchmark.exe` and the EP DLLs under test |
| `HIPEP_MODEL` | **required.** Model directory (holds `genai_config.json`) |
| `HIPEP_OUT` | Output root (default `%TEMP%\hipep-perf`) |
| `HIPEP_PY` | Python for the VLM driver (default: `python` on `PATH`) |
| `HIPEP_PATH_EXTRA` | Extra `PATH` entries the EP needs, `;`-separated. Include TheRock's `bin` when `HIPEP_BIN` lacks `hipblaslt.dll`: without it the JIT cannot resolve `hipblasLt*` and the model fails to load |
| `HIPEP_VLM_BENCH`, `HIPEP_IMAGE`, `HIPEP_PROMPT_FILE` | `--driver vlm` only |
| `RGP_DIR` | Radeon Developer Tool Suite folder, if `RadeonDeveloperPanelCLI.exe` is not on `PATH` |
| `THEROCK_DIST` | TheRock SDK (`bin\hipcc.exe`, `bin\clang.exe`) for the calibration workload |
| `AMD_COMGR` | comgr DLL for readable kernel names in exports (default: `amd_comgr*.dll` in `HIPEP_BIN` or on `PATH`) |

Two builds are used:

- **Measurement build**: the normal build. Every number that decides anything
  comes from it.
- **Profiling build**: the same tree configured with
  `-DHIPDNN_EP_ENABLE_TRACE_SHIM=ON`, for traces, fenced captures and
  calibration. GPU kernels are identical (the option never reaches
  `custom_kernels_<arch>.dll`); compare the two DLLs' hashes to confirm.
  Never measure throughput with it.

## The loop

1. **Baseline** on the measurement build:
   `python -m hipep_perf bench tps --tag base --prompt-file D:\prompts\2k.txt`
   (`bench ttft` for prefill). Timed runs always start from CI's environment;
   `--set-env K=V` opts one run into anything else.
2. **Map** on the profiling build: `python -m hipep_perf trace run tps --tag map --out D:\t\map ...`
   gives time per subgraph, op and kernel (`trace_report.txt`), a Perfetto
   timeline (`trace.pftrace`, `trace_chrome.json`) and a validation block that
   must be clean (no dropped records, no command starting before its call).
3. **Split host from GPU** on the measurement build with
   `--set-env HIPDNN_EP_HOST_TIMELINE=<prefix>`, then
   `python -m hipep_perf analysis host_timeline <prefix>`. Its `dec_sync` caps
   any kernel-side gain.
4. **Capture** the op you care about: `python -m hipep_perf capture tps --fence-op gather --decode-step 8 --seqlen 128 --gen 3000 --reps 1 --counters --tag dec128`.
   The fence alone opens the panel's window at the op; `--sqtt-marker` is only
   for `--inproc-sqtt` (it hangs panel captures).
5. **Size and rank**: `analysis calibrate_sqtt`, `analysis decode_model`,
   `analysis headroom` (prefill) over the capture's `_dispatches.csv`.
6. **Record the prediction** before building anything:
   `python -m hipep_perf ledger predict --id gqa-fuse --metric tps --predicted-pct -2.5 --source decode_model`.
7. **Change one thing**, behind a default-off environment flag.
8. **Prove it** with an interleaved A/B on the measurement build, then record
   the result:
   ```powershell
   '{ "off": { "env": { "HIPDNN_EP_GQA_FUSE_APPEND": "0" } },
      "on":  { "env": { "HIPDNN_EP_GQA_FUSE_APPEND": "1" } } }' | Set-Content arms.json
   python -m hipep_perf ab run --manifest arms.json --metric tps --rounds 6 --reps 3 --prompt-file D:\prompts\2k.txt
   python -m hipep_perf ab run --manifest arms.json --metric tps --rounds 3 --reps 3 --start-round 7 --reverse --skip-prime --prompt-file D:\prompts\2k.txt
   python -m hipep_perf ab summary $env:HIPEP_OUT\tps\tps_summary.csv --metric tps --baseline off --json ab.json
   python -m hipep_perf ledger measure --id gqa-fuse --summary-json ab.json --arm on
   ```
9. **Land** it with numeric and LIT coverage, and check `ledger report` now and
   then: a source whose predictions keep missing should stop being trusted.

## Ground truth before anything else

`python -m hipep_perf calib check --out D:\t\calib` (profiling build in
`HIPEP_BIN`) traces a workload whose costs are known by construction: spin
kernels of exact duration on the GPU's constant-rate wall clock, and read,
write and copy kernels of exact byte counts. It passes only if traced spin
durations are within max(2 us, 2%) of the truth for 95% of launches, every
kernel lands in the ROCTx phase that launched it, no command starts before its
HIP call, and nothing was dropped. Run it after changing the shim, the HIP
runtime or the driver.

`python -m hipep_perf calib rgp --out D:\t\calib_rgp --counters` captures the
same workload through the Radeon Developer Panel. It checks the RGP decoder's
spin durations against the truth, and SPM: the bytes it attributes to each
read, write and copy kernel must be within 5% of the bytes that kernel moves.
Run it after changing the decoder's SPM handling.

For the runner itself, an A/A run (two identical arms) must give an interval
that spans zero, and a deliberate slowdown must be detected
(`HIPDNN_EP_MATMUL_DP4A=0` is about +25% ms/token on Laguna-XS-2.1 decode).

Host stalls make individual decode tokens take two to three times as long, on
every arm, and swamp a mean-TPS comparison. The `vlm` driver records per-token
times, so its rows carry `p10_ms` and `slow_pct`; when `ab summary` flags
stalled rounds, compare `--metric tps_p10` (kernel speed) or drop and re-run
those rounds. `model_benchmark` prints no per-token times, so only `tps` and
`tps_p50` apply to it. Before each run the harness kills competing benchmark
processes and refuses to continue if one survives: a process stuck in GPU
driver teardown keeps its device memory until reboot.

## Commands

| command | does |
|---|---|
| `bench ttft\|tps` | one clean measurement appended to `<metric>_summary.csv`; `--perf-ops` for per-family event timing (throughput invalid) |
| `ab run`, `ab summary` | interleaved, order-reversible A/B over DLL or env arms; paired interval. `ab run` aborts at the first run that yields no result. `ab summary --metric tps_p50\|tps_p10` compares median or fast-decile ms/token and flags rounds whose stalled-token share (`slow_pct`, tokens over 1.5x the run's p10) exceeds `--max-slow-pct` |
| `trace run`, `trace cmd`, `trace report` | run under the shim (profiling build) and post-process |
| `trace compare` | a shim trace against a ktrace capture (counts, attribution) |
| `trace finalize-ktrace` | rocprofv3 export of a ktrace raw spool, as ktrace's C++ `finalize` writes it |
| `capture` | RGP capture positioned by the shim fence (`--fence-op`), or `--trigger` for RGP's blind trigger |
| `rgp decode\|verify\|parity` | decode a `.rgp`; check a capture is usable; compare against frozen reference outputs |
| `calib build\|check\|rgp` | build the ground-truth workload; score traced timings; score RGP and (`--counters`) SPM |
| `analysis <name>` | headroom, decode_model, calibrate_sqtt, host_timeline, roctx_attrib, ... (`--help` after the name) |
| `ledger predict\|measure\|report` | predicted against measured gains |

## Decoder parity

The RGP decoder replaced `tools/rgp_parser` only after producing byte-identical
outputs on a frozen corpus of real captures. To re-check after changing it:
put captures in `<corpus>\rgp\`, reference outputs in `<corpus>\ref\` (as
`<name>_dispatches.csv` etc.), and run
`python -m hipep_perf rgp parity <corpus> --out <dir>`. The only intended
differences are the timeline's process label and the capture path; parity
decodes SPM the reference decoder's way, because that way is wrong (see
"SPM" below) and is kept only so the comparison covers everything else. Unit
tests: `HIPEP_RGP_CORPUS=<corpus>\rgp KTRACE_SRC=<ktrace checkout> python -m
pytest tests` (each variable enables the tests that need it).

## SPM

SPM bandwidth comes from the EA block's request counter. Two things were
wrong in the decoder this replaced, and are fixed here: SPM samples were
pinned to the first dispatch although the SPM ring keeps the most recent ones,
and requests were counted as 64 bytes although calibration kernels of known
size show 32. Per-dispatch `mem_gbps` and `bound_class` from captures decoded
before this change are unreliable whenever SPM wrapped (a capture longer than
the SPM buffer holds, about 130 ms at the panel's default sampling on
gfx1151): re-decode them. On a 16K decode capture, 483 of 580
"memory-bound" dispatches were misclassified.

## Exported traces

Every trace directory gets the rocprofv3 file set (`trace_kernel_trace.csv`,
`trace_hip_api_trace.csv`, `trace_results.json`, ... and `--stats` tables) in
ktrace's conventions, so tools that read rocprofv3 or ktrace output read these
traces. The export passes ktrace's ROCm 7.2.1 validator and, given the same
ktrace raw spool, is byte-identical to ktrace's C++ `finalize`. Call arguments
are not traced, so stream ids, launch geometry and allocation sizes are 0, as
ktrace writes them without `KTRACE_API_ARGS`.

## Op map

SQTT has no op boundaries, so capture analysis maps kernel families to EP ops
with `hipep_perf/analysis/op_families.json`. It is derived from ROCTx
attribution on shim traces, not written by hand:
`python -m hipep_perf analysis roctx_attrib <trace dir>... --emit-map map.json`
writes each family's op and checks the map against those traces; pass it to
`calibrate_sqtt --op-map`. The checked-in map covers Laguna-XS-2.1 (decode and
16K prefill, 100% agreement); trace any other model and derive its own.

## Rules that are not optional

Each of these is here because ignoring it produced a confident wrong answer.

**Never measure throughput with the EP's own profiler.** `HIPDNN_EP_PERF=1`
costs about 4% on its own. `HIPDNN_EP_TRACE_FILE` and `HIPDNN_EP_PERF_OPS` turn
the same profiler on silently, and a trace path left in a shell once inflated a
16K VLM baseline from 14,610 ms to 17,545 ms. The runner clears all of them,
plus `HIPDNN_EP_ROCTX` and the fence variables, before every run.

**SQTT stretches gaps; host overhead comes from the host timeline.** Tracing
slows the host far more than the GPU, so idle time between kernels in a capture
is mostly the tracer. Only `host_timeline` sizes host time.

**RGP pegs clocks: a capture win is not a TTFT win.** Under capture the clocks
are pinned at peak, so redundant arithmetic is free. Lowering the GEMV-to-WMMA
cutoff to M=2 looked 12.5% faster in RGP and was 47 ms slower on TTFT.

**A floor only bounds anything while its resource binds.** A bandwidth floor
computed at DRAM prices charges for bytes served from cache. One fix built from
a 5.38x bandwidth gap regressed TTFT by 4%; the same capture's `bound_reason`
showed those kernels above 100% of the roofline. Rows marked
`[BW FLOOR DOES NOT BIND]` are traffic volume, not recoverable time.

**Rank by utilisation, not by share of runtime.** The largest share is often
the op closest to its limit. `headroom` ranks by distance from each
component's own floor.

**Fence position is part of the measurement.** Fencing on the first instance
of an op lands before the autotuner settles; use `--fence-skip` and check the
capture with `analysis inspect_capture`.

**On an MoE model, synthetic ids are a different workload.** Random token ids
route narrower than text and differently on every run. Use real prompts
(`--prompt-file`) for timed runs; `--seqlen` random ids only where the Run
arithmetic must be exact.

**Interleave, reverse, and discard warm-up.** Run-to-run spread is comparable
to effects worth shipping, and the absolute level drifts downward over a
session as caches warm. A delta is only meaningful within one invocation; never
add deltas from different invocations.

**One run at a time.** Concurrent GPU work invalidates measurements; the
harness lock refuses a second run, and every timed run stops competing
benchmark processes it owns. It does not stop other GPU workloads (an
inference server, a game): check for them yourself.

**A benchmark can hang in driver teardown.** `model_benchmark` has stayed in
the GPU driver after printing its results, where not even a forced kill
finishes it. The harness kills a target 60 s after its output ends and moves on
if that fails; such a process holds its DLLs until a reboot, so run DLL A/Bs
from a fresh copy of `HIPEP_BIN` until then.

**A standalone microbenchmark is a hypothesis.** Per-shape GEMV wins measured
outside the model predicted +4-8% decode throughput and delivered nothing.
Check against in-model family time (`bench tps --perf-ops`) before
extrapolating.

**Keep the process alive long enough to dump.** RGP streams the trace out of
the live process; a late fence with too little runtime after it produces no
file. Raise `--reps` or `--gen`, or lower `--op-count`.
