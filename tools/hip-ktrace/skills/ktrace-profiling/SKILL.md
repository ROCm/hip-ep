<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: ktrace-profiling
description: Profile a HIP/ROCm workload on Windows and optimise a specific kernel, using ktrace for the global timeline and RGP + rgpparse for per-instruction analysis. Use when asked to find why a model is slow, to rank kernels by cost, to capture RGP/SQTT, or to optimise a named kernel with real runtime data rather than static disassembly.
---

# Profiling a HIP workload with ktrace

Two phases, in order. Do not skip phase 1. An RGP capture taken before the
global ranking exists is a guess about what matters.

## Phase 1 — global timeline

Run the workload under ktrace to get HIP runtime, kernel, and memory timelines.

```bat
run.bat system -- <application.exe> [args]
```

Costs roughly 10% runtime (measured ~9% on decode throughput). That buys the
dispatch count and order for the entire network.

**ROCTx is required, not optional.** The Perfetto export alone cannot separate
model load, prefill, and decode — it is one undifferentiated run of kernels. Add
ROCTx ranges to the application so phases are delimited. With them the timeline
carries four layers — HIP API, kernel, memory, and phase — that cross-validate
each other. Without them a kernel ranking has no context: you cannot tell which
phase a hotspot belongs to, so you cannot tell whether it is worth optimising.

Link against `roctx64.lib`, keep `roctx64.dll` on the path, and mark phases:

```cpp
roctxRangePush("prefill");
// HIP work
roctxRangePop();
```

**Read `ktrace_report.txt` first.** Its kernel summary ranks kernels by total
GPU time. That ranking decides what to optimise. Open
`ktrace_perfetto.json` at https://ui.perfetto.dev when the timeline shape
matters — gaps, serialisation, transfer stalls.

A representative capture is one warm-up plus 3–5 measured iterations, repeated
across the inputs that matter.

## Bridge — make the dispatch order reproducible

Disable autotune before going further. With it on, dispatch count and order vary
between runs, so a dispatch index from one run does not identify the same kernel
in the next, and phase 2 targets the wrong work.

With autotune off, find the target kernel's position in the dispatch sequence
from `<prefix>_kernel_trace.csv` (use `<prefix>_kernel_trace_readable.csv` to
match a demangled name). That row's dispatch index is the capture window for
phase 2.

## Phase 2 — per-instruction capture

Capture RGP at that position with SPM counters and instruction timing, then
analyse.

```bat
run.bat kernel -a dispatch:<index>:16 ^
    --rgp-counter-collection ^
    --rgp-instruction-tracing ^
    --rgp-sqtt-buffer-size maximum ^
    -- <application.exe> [args]

rgpparse all <out>\trace_capture.rgp --outdir <out>\rgpparse --instruction-mode split
```

ktrace prints this `rgpparse` command after a successful capture; it does not
run it, because rgpparse ships and versions separately.

The payoff is `<outdir>/trace_capture_instruction_detail/event_*/<Kernel>_isa.txt`:
every machine instruction with hit count, cost %, latency per hit, per wave and
total, plus VALU/SALU/LDS/branch busy. For example a line showing 7.60% of cost
on `s_waitcnt lgkmcnt(0)` at 72 cycles per hit localises a stall to one
instruction waiting on a scalar load.

This is the point of phase 2. Static disassembly shows what the compiler emitted;
it never shows what each instruction actually cost at runtime. With per-instruction
clocks, SPM counters, and LDS behaviour, kernel changes can be evaluated against
measurement and iterated, rather than reasoned about from the ISA alone.

## Reading what rgpparse produced

| Output | Answers | Read it for |
| --- | --- | --- |
| kernel summary in `*_timeline.json` | Which kernel is slowest? Which candidate won? | Ranking variants against each other |
| `<Kernel>_isa.txt` (split mode) | Which *instruction* the time went to | Hit count, cost %, latency per hit/wave/total, VALU/SALU/LDS/branch busy |
| `derived` in `*_spm.json` | Bandwidth-bound or latency-bound? Cache behaviour? | `Memory unit busy` vs `Memory unit stalled`, L0/L1/L2 hit, LDS bank conflict |
| `occupancyBand` in `*_timeline.json` | Is parallelism limited by resources? | Why a bigger tile can be slower |
| `*_chrome.json` | How kernels, barriers and counters lay out in time | Open at https://ui.perfetto.dev |
| `*_asic.json`, `*_chunks.json` | Hardware parameters, container contents | Confirming clocks and gfx IP level |

**Read SPM and ISA together; either alone misleads.** SPM identifies the *kind*
of bottleneck, the ISA identifies *where* it is.

Two distinctions worth making explicitly:

- **Bandwidth-bound versus latency-bound** decides the whole strategy. High
  `Memory unit busy` with low `Memory unit stalled` means the memory system is
  waiting, not saturated: raise arithmetic intensity and occupancy rather than
  shrinking data. Measured here: 55% busy against 0.84% stalled, which is what
  justified register blocking.
- **A healthy counter does not mean a healthy access pattern.** `LDS Bank
  Conflict` at 0.006% ruled out conflicts, yet LDS was still the top cost. The
  ISA showed why: `ds_load_2addr_b32` at 25.6% of cost where `ds_load_b128`
  belonged, because a shared-tile row stride padded by one float lost 16-byte
  alignment and blocked vectorisation. Padding by four fixed it and made every
  tile shape 1.2x-1.7x faster. SPM alone would have cleared LDS of suspicion.

A worked trace of this loop, tuning a naive tiled GEMM to 4.19x through three
measured iterations, is in `src/workload.cpp`: `GemmKernel` and the
`GemmKernelTiled*` variants are all dispatched every round, so one capture
ranks them under identical clocks.

## Traps

| Trap | What happens | Do this |
| --- | --- | --- |
| Capture window too large | `There was an error trying to collect a trace` (`result=-2`) | The limit is on total SQTT data, not on a dispatch count, so it moves with the kernels in the window. Start narrow and widen. Budget ~1.8 MB per dispatch with instruction tracing, ~0.16 MB without. |
| Target exits too quickly | Same `result=-2`, driver cannot dump the trace | Give the target real work. A run finishing in well under a second fails even with a tiny window. |
| `--instruction-mode full` | One 270 MB JSON held in memory | Use `split`. Identical data, 0.7 MB index plus per-event files. |
| Shared-tile padding of one element | Odd row stride loses 16-byte alignment, so wide LDS loads degrade to narrow ones | Pad by four floats: same bank offset, alignment kept. |
| Judging a variant across separate runs | Clocks and thermal state differ between runs | Dispatch the variants in one rotation and let a single capture rank them. |
| Autotune left on | Dispatch index points at a different kernel next run | Disable it before phase 2. |
| Missing `--` | `run.bat` exits with usage | `--` before the target command is mandatory; everything after it is forwarded untouched. |

The dispatch *start* index is not a constraint — starting at 8 works as well as
at 1000, so any kernel position is reachable. Only the *count* is limited.

Narrow a large capture with `--instruction-kernel <text>`,
`--instruction-event(s)`, or `--instruction-max-records` rather than exporting
everything.

## Reference

`README.md` documents the modes, flags, output files, and measured capture
limits. `ktrace_test.bat` in the workspace root exercises every mode and is the
quickest way to confirm a build behaves.
