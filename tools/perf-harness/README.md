<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# perf-harness

Find the next thing worth optimising, size it before writing any kernel code,
and prove the change afterwards. Two workloads, which are different problems:

| | prefill / TTFT | decode / TPS |
|---|---|---|
| unit | one chunk, scaled to the prompt | one token |
| bind | compute can bind | never: M=1, so bytes/BW throughout |
| bench | `bench/bench_ttft.ps1` | `bench/bench_tps.ps1` |
| model | `analysis/prefill_model.py` | `analysis/decode_model.py` |
| rank | `analysis/headroom.py` | `analysis/headroom.py --decode` |

Read [decode](#decode-tps) if TPS is the target; the rest of this document is
written around prefill.

[`tools/rgp_parser`](../rgp_parser) turns one `.rgp` into CSVs. This turns a
question — *what should I work on next, and did it work?* — into an answer:
scripts to position and take the capture, to rank candidates by how far each is
from its own hardware floor, and to A/B the result on real TTFT.

## Setup

Everything resolves from environment variables, with discovery fallbacks:

| variable | meaning |
|---|---|
| `HIPEP_BIN` | **required.** Directory with `model_benchmark.exe` and the EP DLLs under test |
| `HIPEP_MODEL` | **required.** Model directory (the one holding `genai_config.json`) |
| `HIPEP_PY` | Python interpreter. Default: `python` on `PATH` |
| `HIPEP_OUT` | Output root. Default: `%TEMP%\hipep-perf` |
| `HIPEP_PATH_EXTRA` | Extra `PATH` entries the EP needs (ROCm SDK runtime dirs), `;`-separated |
| `RGP_DIR` | Radeon Developer Tool Suite folder. Only needed if `RadeonDeveloperPanelCLI.exe` is not on `PATH` |

```powershell
$env:HIPEP_BIN   = 'C:\work\gpu-test-package\bin'
$env:HIPEP_MODEL = 'C:\models\gpt-oss-20b-webgpu-int4-rtn-block-32'
pip install -r ..\rgp_parser\requirements.txt
```

The capture scripts need a build containing the RGP capture fence
(`rgp_capture_fence` in `lib/Runtime/op_profile.cpp`). It is inert unless
`RGP_FENCE` is set, so a normal build carries it at no cost.

## The workflow

### 1. Establish a clean baseline

```powershell
.\bench\bench_ttft.ps1 -Tag baseline -SeqLen 16384
```

Nothing later means anything without this number, because a capture cannot tell
you whether a change helped — see [RGP pegs clocks](#rgp-pegs-clocks-a-capture-win-is-not-a-ttft-win).

### 2. Capture the op you care about

A 16k prefill is tens of thousands of dispatches. Positioning a capture on a
specific one by delay or dispatch index does not survive run-to-run jitter, so
the runtime fence does it instead: it drains the GPU and idles immediately
before the op's *N*-th instance, and the script triggers RGP inside that window.

```powershell
# the 37th qmoe instance -- deep enough that the autotuner has settled
.\capture\rgp_capture.ps1 -Op qmoe -Skip 36 -Counters -Buf minimum -Reps 16 -Tag shallow
```

`-Counters` turns on SPM. Without it every dispatch classifies as
`compute-or-memory (undetermined)`, so take it unless you know you don't need
bandwidth. `-Skip` is the whole game — see
[fence position](#fence-position-is-part-of-the-measurement).

### 3. Check what you actually captured

```powershell
python .\analysis\inspect_capture.py $env:TEMP\hipep-perf\captures\shallow_dispatches.csv
```

`rgp_capture.ps1` already gates on chunk inventory via `verify_rgp.py`, but that
only proves the file is decodable, not that it holds steady-state work. This
does.

### 4. Attribute, model, and rank

```powershell
cd analysis
# where the time sits, and which region each int4 kernel served
python attrib_regions.py ...\shallow_dispatches.csv

# one chunk -> the whole prefill, using a shallow and a deep capture
python prefill_model.py ...\shallow_dispatches.csv ...\deep_dispatches.csv `
       --at-chunk 2.5 31.5 --measured-ttft-s 9.04

# the ranking that decides what to work on
python headroom.py ...\shallow_dispatches.csv ...\deep_dispatches.csv `
       --dense-ms 31.9 --lm-head-ms 18.8 --attention-s 1.75 --prefill-s 8.53
```

The three earlier scripts print the exact arguments the next one wants.

### 5. Change one thing, then prove it

```powershell
'{ "base": { "dll": "D:\\base\\custom_kernels_gfx1151.dll" },
   "cand": { "dll": "D:\\cand\\custom_kernels_gfx1151.dll" } }' | Set-Content arms.json

.\bench\ab_interleaved.ps1 -Manifest arms.json -Rounds 6 -Reps 4
.\bench\ab_interleaved.ps1 -Manifest arms.json -Rounds 3 -Reps 4 -StartRound 7 -Reverse -SkipPrime
python .\bench\ab_summary.py $env:TEMP\hipep-perf\ttft\ttft_summary.csv --baseline base
```

## Decode (TPS)

Decode is a memory-traffic budget, not a compute problem. At M=1 every GEMM is a
GEMV — one pass over the weights for a single row of activations — so arithmetic
intensity is ~2 FLOP/byte against a part that needs ~232 to break even. The FLOP
term never binds, and each component's floor is just its bytes over the roofline.

### 1. Know what the model runs

The capture positions itself on an op *by name*, so the names have to come from
the build under test. A trace gives them, along with the shapes each op runs:

```powershell
$env:HIPDNN_EP_TRACE_FILE = 'D:\out\trace.json'   # writes trace.<pid>_<tag>.json
.\bin\model_benchmark.exe -i $model -g 16 -r 1 -w 1 -b 1 -ml 0 -l 128 --use_random_tokens
python .\analysis\trace_ops.py D:\out\trace.p1234_abcd.json --run 20 --sequence 16
```

`HIPDNN_EP_TRACE_FILE` implies the profiler, so this is structure only — see
[never measure throughput with the EP's own profiler](#never-measure-throughput-with-the-eps-own-profiler).
The op that *starts* a step (an embedding `gather`, usually) is the one to fence
on: fencing on the MoE op instead misses the first layer's attention.

### 2. Position on a decode step, not a dispatch index

`-Skip` cannot name a phase. Every one of the layers requests the same op within
one step, so a skip selects a *layer*. The Run counter does the phase:

```powershell
# decode step 8 at a 128-token prompt. -Gen is large on purpose: see below.
.\capture\rgp_capture.ps1 -Op gather -DecodeStep 8 -SeqLen 128 `
    -Gen 3000 -Reps 1 -Buf minimum -OpCount 1500 -Counters -Dwell 120 -Tag dec128
```

`-DecodeStep` converts to a Run index and prints the arithmetic, because a long
prompt is fed in 512-token chunks: decode step 1 at 16K is Run 32, not Run 1. It
refuses to run with `-PromptFile`, where the token count is unknown — use
`--use_random_tokens` for captures and real prompts for the timed runs.

**`-Gen` is the whole game for decode captures.** RGP streams the trace out of
the *live* process, and a decode step's worth of dispatches is a ~80 MB dump
that takes far longer to drain than the few hundred milliseconds of work left
after a late fence. A first attempt at `-Gen 128 -Reps 2` stalled at 0.8 of
82.5 MB and produced **no file at all**. `-Gen 3000` buys ~60 s of live process;
lowering `-OpCount` shrinks what has to move.

### 3. Attribute and rank

```powershell
python .\analysis\decode_model.py ...\dec128_dispatches.csv `
       --preset qwen3-30b-a3b --kv-len 144 --measured-ms 14.6

# how the ranking moves with context -- the only term that changes is the KV read
python .\analysis\headroom.py --decode --preset qwen3-30b-a3b `
       --at 144:14.6:...\dec128_dispatches.csv `
       --at 16512:16.4:...\dec16k_dispatches.csv
```

`decode_model.py` normalises by how many decode steps the window actually holds
(`topk_routing` count over `layers`), because a capture rarely lands on exactly
one. It also accepts `--trace` as a source: exact op attribution, but the
profiler inflates the total, so pass `--measured-ms` and it rescales the shares
onto the real number and says it did.

### Quantisation is per-tensor, and it dominates the floor

An export can leave the `lm_head` and the MoE router in fp16 while quantising
everything else, and `decode_bytes()` in `perfcommon.py` has `lm_head_fp16` /
`router_fp16` for exactly that. It is not a detail: on qwen3-30b-a3b the fp16
`lm_head` reads **622 MB per token**, four times what an int4 assumption
predicts and ~30% of the whole step's traffic. Check
`quantization_config` in `model_config.json`, and confirm it against the trace —
a quantised matmul lands on `matmul_nbits`, an fp16 one on `matmul`.

## Rules that are not optional

Each of these is here because ignoring it produced a confident wrong answer.

### Never measure throughput with the EP's own profiler

`HIPDNN_EP_PERF=1` costs about 4% on its own, which is larger than most changes
worth shipping. SQTT is hardware thread tracing and perturbs far less, so it
carries the timing. The harness clears `HIPDNN_EP_PERF`, `HIPDNN_EP_DEBUG` and
`HIPDNN_EP_TRACE_FILE` before every run rather than trusting the shell to be
clean.

`HIPDNN_EP_TRACE_FILE` is the dangerous one. `hipdnn_ep_perf_enabled()` is true
for it as well as for `HIPDNN_EP_PERF`, so it enables the profiler while
printing none of PERF's console output, and it survives in the shell after the
capture that wanted it. A 16K VLM baseline was reported as 17,587 ms for a
whole round of analysis on that basis; the same binaries measure 14,455-14,610
ms once the variable is gone, and re-setting it reproduces 17,545 ms on demand.
Cross-check any suspicious baseline by re-running it in a fresh shell.

### Rank by utilisation, not by share of runtime

A percentage of runtime says where time *goes*, not where it is *wasted*. The op
with the largest share is frequently the one closest to the hardware limit, and
ranking that way sends you to work on it. `headroom.py` instead computes, for
each component, the floor its own work implies — `max(bytes/BW, FLOP/peak)` —
and ranks by the gap.

On gpt-oss-20b this inverted the answer. The largest share was MoE experts at
51.6% of prefill; they were already running at reasonable efficiency. The real
finds were a tier of small-M expert blocks at **18% of floor**, and an `lm_head`
computing logits for all 512 rows of every chunk when only the last row is ever
read — 0.60 s of work that should not run at all, invisible to a share ranking
because it was only 7% of runtime.

### Use published ceilings, not the best rate you happened to see

Taking "the best rate observed" as the ceiling is circular: if the whole stack
misses a ceiling, that ceiling never appears in the data. Derive it and check
the derivation against a part with published numbers. The gfx1151 figure here is
40 CU × 2.9 GHz × 512 FLOP/clk = 59.4 TFLOP/s; the same model gives 122.9
TFLOP/s for a 7900 XTX against AMD's published 122.8, which is what makes it
trustworthy. (`hipInfo` reports `multiProcessorCount=20` — those are WGPs, two
CUs each. Halving the CU count halves the ceiling and doubles every utilisation
number in the report.)

### Fence position is part of the measurement

Fencing on the first instance of an op lands before the autotuner has settled.
One capture taken that way was 91.6% GQA — 527 consecutive dispatches cycling
four configurations — and every timing in it was a tuning trial. It looked
entirely valid. Use `-Skip` to reach steady state, and run `inspect_capture.py`,
which looks for exactly this signature.

### RGP pegs clocks: a capture win is not a TTFT win

Under capture the clocks are pinned at peak, so redundant arithmetic is free.
Work that costs power — and therefore sustained clocks — in production shows up
as free in the trace.

This is not theoretical. Lowering the GEMV→WMMA cutoff to M=2 improved the
targeted blocks by 12.5% in RGP, with untouched control buckets steady at 3% and
0.4%, predicting roughly a 73 ms TTFT win. Measured on TTFT it was **47 ms
slower**: WMMA computes 16-row tiles regardless of M, and on a power-managed APU
that waste depresses clocks for everything else. An intermediate cutoff of M=8
kept the read-efficiency win without the waste and landed at −211 ms.

So: captures rank candidates and explain mechanisms. Only TTFT decides.

### Interleave, reverse, and discard the warm-up

Run-to-run spread is comparable to the effects worth shipping, so
block-sequential runs let drift land entirely on one arm. `ab_interleaved.ps1`
alternates arms and pairs by round; `ab_summary.py` reports the interval over
the paired differences. Always run one block with `-Reverse`: if the sign flips,
you measured position and not the change. Discard rounds taken while the machine
is shedding heat from a build — judge that from the absolute level against a
known baseline, never by dropping rounds that disagree.

### Give each arm its own autotune cache

The on-disk WMMA tuner cache holds a single build timestamp and is discarded
when it doesn't match. Arms sharing one `TEMP` therefore make every DLL swap a
cold-tune run, and you measure tuning instead of the kernel.
`ab_interleaved.ps1` gives each arm its own `TEMP`.

### Keep the process alive long enough to dump

RGP streams the trace out of the *live* process. A late-positioned fence can
leave too little runtime for a large dump, which stalls part-written and
produces no file at all. That is what `-Reps` is for; it is not extra
measurement.

## Files

| | |
|---|---|
| `common.ps1` | Environment resolution shared by the PowerShell scripts |
| `capture/rgp_capture.ps1` | Take one capture — fence-positioned or auto-triggered, ± SPM — and decode it |
| `capture/verify_rgp.py` | Fail a capture missing `SqttData`/`SpmCounterData` before anything reads it |
| `bench/bench_ttft.ps1` | One clean TTFT run, appended to CSV |
| `bench/bench_tps.ps1` | One clean decode-throughput run, CI-identical args |
| `bench/ab_interleaved.ps1` | Interleaved, order-reversed A/B across DLL variants |
| `bench/ab_summary.py` | Paired statistics over the rounds |
| `analysis/perfcommon.py` | Model/device constants and dispatch-stream segmentation |
| `analysis/inspect_capture.py` | What is in this capture, and is it steady state |
| `analysis/attrib_regions.py` | Split kernel time between the MoE loop and dense projections |
| `analysis/expert_blocks.py` | Per-M-bucket expert cost vs floor; which kernel each bucket used |
| `analysis/prefill_model.py` | Two capture depths → whole-prefill composition |
| `analysis/trace_ops.py` | Operator inventory and per-Run structure from an EP chrome trace |
| `analysis/decode_model.py` | One decode step → per-component traffic vs its memory floor |
| `analysis/headroom.py` | Rank candidates by recoverable seconds (`--decode` for per-token) |

The analysis scripts default to gpt-oss-20b shapes and gfx1151 ceilings; both
are overridable (`--hidden`, `--vocab`, `--layers`, `--bw-gbs`, `--peak-tflops`).
The structural segmentation — regions, layers, expert token counts — is
recovered from the dispatch stream itself and carries no model assumptions.
