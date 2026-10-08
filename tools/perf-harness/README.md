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
| `HIPEP_VLM_BENCH` | `-Driver vlm` only: onnxruntime-genai's `vlm_benchmark.py` |
| `HIPEP_IMAGE` | `-Driver vlm` only: the benchmark image |
| `HIPEP_PROMPT_FILE` | Default `-PromptFile` (required on `-Driver vlm`) |
| `RGP_DIR` | Radeon Developer Tool Suite folder. Only needed if `RadeonDeveloperPanelCLI.exe` is not on `PATH` |

Every timed script runs the EP with **CI's environment**: it clears
`HIPDNN_EP_AUTOTUNE` and `HIPDNN_EP_MATMUL_CUSTOM_WMMA` (CI sets neither; the
harness used to force both on, so its captures showed kernels the measured
build does not run) and the instrumentation variables below. `-SetEnv 'K=V'`
opts one run into anything else and restores the previous value afterwards.

`-Driver vlm` runs `vlm_benchmark.py` through `bench/vlm_driver.py`, which does
what CI does: reads the prompt file and passes it in-process (a 16K prompt is
past Windows' command-line limit, and `vlm_benchmark.py` has no
`--prompt_file`), after loading `onnxruntime.dll` from `HIPEP_BIN`.

```powershell
$env:HIPEP_BIN   = 'C:\work\gpu-test-package\bin'
$env:HIPEP_MODEL = 'C:\models\gpt-oss-20b-webgpu-int4-rtn-block-32'
pip install -r ..\rgp_parser\requirements.txt
```

The capture scripts need a build containing the RGP capture fence
(`rgp_capture_fence` in `lib/Runtime/op_profile.cpp`). It is inert unless
`RGP_FENCE` is set, so a normal build carries it at no cost.

### Instrumentation switches in the build

All are latched on first read and cost one cached check when unset.

| variable | what it does | throughput valid? |
|---|---|---|
| `HIPDNN_EP_HOST_TIMELINE=<prefix>` | Host timestamps per EP Compute (entry, after marshal, after `inference_compute`, exit) to `<prefix>.ep.<pid>.csv`, and per `hipdnn_ep_stream_sync` (entry, exit) to `<prefix>.sync.<pid>.csv`. One steady_clock axis, batched appends, **no sync added** | yes: +0.09 ms/token, 95% CI [−0.07, +0.25], on gemma4 26B 2K (7 interleaved rounds, one block reversed) |
| `HIPDNN_EP_PERF_OPS=<op>[,<op>]` | The op profiler, limited to the named `OP_PROFILE` families, each call bracketed by its own start/end event pair. Implies `HIPDNN_EP_PERF` | **no** — only the family GPU ms |
| `HIPDNN_EP_PERF=1` / `HIPDNN_EP_TRACE_FILE` | The full op profiler / chrome trace | **no** |
| `RGP_FENCE*` | The capture fence (`rgp_capture.ps1` sets these) | n/a |

## The workflow

### 1. Establish a clean baseline

```powershell
.\bench\bench_ttft.ps1 -Tag baseline -SeqLen 16384

# MoE models: use real text, because random ids route differently -- see
# https://github.com/ROCm/hip-ep/blob/main/tools/perf-harness/README.md#on-an-moe-model-synthetic-ids-are-a-different-workload
.\bench\bench_ttft.ps1 -Tag baseline -PromptFile D:\prompts\16k.txt
```

`-SeqLen` and `-PromptFile` are mutually exclusive here: the file fixes the
length, and the CSV records the count the benchmark reports rather than the one
requested.

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

`-Metric tps` runs the same design on decode instead, through `bench_tps.ps1`
and the `ms_per_token` column. Arms can also be environment variables rather
than DLLs, which is what a change landed behind a default-off flag wants — one
binary, so the arms cannot differ in anything but the flag:

```powershell
'{ "off": { "env": { "HIPDNN_EP_GQA_FUSE_APPEND": "0" } },
   "on":  { "env": { "HIPDNN_EP_GQA_FUSE_APPEND": "1" } } }' | Set-Content arms.json

.\bench\ab_interleaved.ps1 -Manifest arms.json -Metric tps -SeqLen 2048 -Rounds 6 -Reps 3
python .\bench\ab_summary.py $env:HIPEP_OUT\tps\tps_summary.csv --metric tps --baseline off
```

Env keys are unioned across arms and every key is written for every arm, absent
ones removed: `Env:` is process-wide, so without that arm A's variables survive
into arm B and silently make it a second copy of arm A.

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

**Check that chunking assumption against the model in front of you.** It is the
`-PrefillChunk` default, not a law, and a model whose generator does not chunk
feeds the whole prompt as one Run: on a gpt-oss-120b proxy, 128, 2048 and 16384
token prompts each arrived as a single Run with `sq` equal to the whole prompt,
so decode step 1 was Run 1 at every length and `-DecodeStep` would have aimed 31
Runs past the target at 16K. `trace_ops.py` prints the Run table with the per-Run
`gqa` shape, which settles it in one run. For a prefill capture, positioning with
`-AfterInferences` on a Run the trace actually shows avoids the question.

Where prefill is a single Run, note also that `-w 0` in this script makes Run 0
the cold one: prefills land at Runs 0, `-Gen`, `2*-Gen`, and only the second
onward are past the autotuner.

**`-Gen` is the whole game for decode captures.** RGP streams the trace out of
the *live* process, and a decode step's worth of dispatches is a ~80 MB dump
that takes far longer to drain than the few hundred milliseconds of work left
after a late fence. A first attempt at `-Gen 128 -Reps 2` stalled at 0.8 of
82.5 MB and produced **no file at all**. `-Gen 3000` buys ~60 s of live process;
lowering `-OpCount` shrinks what has to move.

### 3. Split the step on the host first

A capture cannot tell you how much of a step is host overhead — see
[SQTT stretches gaps](#sqtt-stretches-gaps-host-overhead-comes-from-the-host-timeline).
The host timeline can, on the real build at full speed:

```powershell
# -SetEnv, not the shell: every timed script clears the instrumentation variables first.
.\bench\bench_tps.ps1 -Driver vlm -Tag tl -PromptFile D:\prompts\2k.txt `
    -SetEnv "HIPDNN_EP_HOST_TIMELINE=$env:HIPEP_OUT\tl\g26_2k"
python .\analysis\host_timeline.py $env:HIPEP_OUT\tl\g26_2k --json $env:HIPEP_OUT\tl\g26_2k.json
```

Ops in the EP file are compiled-kernel instance addresses (one per session's
fused graph), not names. It takes the decoder to be the one with the most total
time, takes one step as consecutive decoder calls, and splits the median step:

| component | meaning |
|---|---|
| `dec_marshal` | EP entry to `inference_compute` (input binding, output allocation) |
| `dec_launch` | host time inside `inference_compute` before the sync is reached |
| `dec_sync` | blocked in `hipStreamSynchronize` — GPU work the host waited for |
| `dec_post` / `dec_tail` | after the sync / after compute to EP exit |
| `other_ep`, `outside_ep` | other EP ops (embedding, vision) and everything outside the EP (sampling, genai) |

`dec_sync` is the only part a faster kernel can shrink, so it caps every
kernel-side gain below.

### 4. Time the families that matter, in-model

```powershell
.\bench\bench_tps.ps1 -Driver vlm -Tag fam -PromptFile D:\prompts\2k.txt `
    -PerfOps matmul_nbits,gqa
```

One process per op (their event pairs would otherwise perturb each other),
logged to `tps_fam.perfops_<op>.log`, then summarised by `calibrate_sqtt.py`:
per-step GPU ms and calls per step, from event pairs around each call. The run
appends nothing to the TPS CSV, because its TPS is not valid.

### 5. Calibrate the capture against it

Take the capture on the same build and config, then:

```powershell
python .\analysis\calibrate_sqtt.py ...\dec_dispatches.csv `
    --perf-log ...\tps_fam.perfops_matmul_nbits.log `
    --perf-log ...\tps_fam.perfops_gqa.log `
    --map gqa+=gemm --preset gemma4-26b-a4b --json calib.json
```

For each op, `factor = SQTT family ms / event ms` over the op's kernel families
(`OP_KERNEL_FAMILIES` in `perfcommon.py`). Both sides over-state the
uninstrumented time — SQTT stretches kernels, event pairs add marker packets
and intra-op gaps — so `decode_model.py` divides by `max(1, factor)` and keeps
the smaller. On gemma4 26B 2K: `matmul_nbits` 0.80 (events read ~10 µs per call
high), `gqa` 1.10, `qmoe` 1.15. It refuses to report a factor unless
SQTT dispatches per call come out a near-integer: a fractional count means the
family map is wrong, the capture window is off, or the build differs. `gemm` is
not in the default map because it is shared; on gemma4 decode it is `gqa`'s
decomposed attention (proved by `-PerfOps gqa` matching SQTT only with it), and
`--map gqa+=gemm` says so. On gpt-oss `gqa` has its own flash-decode kernels and
`gemm` is the fp16 `lm_head` and router `MatMul`s, so it is `--map matmul=gemm`
(1.92 dispatches per call: each router is a split-K GEMM plus its `PostGSU`
reduction; the router bias is a separate `Add`).
Unmapped families are listed so none is silently dropped.

A failed check prints the op's per-family counts. Some are genuine: on gemma4
12b, hipBLASLt picks a split-K GEMM for some layers' attention and adds a
`...PostGSU...` reduction kernel after it, so `gqa` averages 5.17 dispatches
per call (`gemm` 2.17). Once you have found the extra kernel in the capture,
`--accept gqa` records the override in the JSON instead of hiding it.

The markers' own GPU cost can be subtracted rather than bounded. Give each
`-PerfOps` run its own `-SetEnv HIPDNN_EP_HOST_TIMELINE=...`, add one
`-PerfOps none` run (PERF mode on, no op selected) as the base, and pass
`--base-timeline base.json --op-timeline matmul_nbits=mnb.json ...`. Because
the decoder is GPU-bound, the rise in its launch + sync wait over the base is
what the op's start/end pairs add; that is taken off the op's event time
before the factor is formed, and both numbers go into the JSON.

### 6. Budget and rank

```powershell
python .\analysis\decode_model.py ...\dec_dispatches.csv --preset gemma4-26b-a4b `
       --kv-len 2300 --calib calib.json --host-timeline $env:HIPEP_OUT\tl\g26_2k.json

# how the ranking moves with context -- the only term that changes is the KV read
python .\analysis\headroom.py --decode --preset gemma4-26b-a4b --calib calib.json `
       --at 2300:21.5:...\dec2k_dispatches.csv   --host-timeline ...\g26_2k.json `
       --at 16400:24.0:...\dec16k_dispatches.csv --host-timeline ...\g26_16k.json
```

With **both** `--calib` and `--host-timeline`, component times are the capture's
kernel time with calibrated families divided by their factor, the step period
comes from the host timeline, and the output gives recoverable ms and an upper
bound on tok/s (recovery capped at `dec_sync`). It also checks a budget: the
decoder's kernels must fit inside its measured compute (`dec_launch +
dec_sync`). Calibrated kernels more than 5% over it mean some family is still
inflated, so the output drops back to shares, `UNVALIDATED`, and says by how
much. Measured on gemma4 at 2K, same build, env and prompt for all three runs:

| model | decoder compute | calibrated kernels | result |
|---|---|---|---|
| 26B-A4B | 20.53 ms | 21.02 ms (+2.4%) | validated |
| 12b | 44.28 ms | 52.84 ms (+19.3%) | shares only: `matmul_nbits` reads 41.8 ms by events *and* by SQTT, so both are inflated there and nothing here can say by how much |
| gpt-oss-120b-proxy-L12 (`chunk_size` 512) | 10.86 ms | 10.56 ms (−2.8%) | validated; 16K 11.81 vs 11.45 ms. Two captures per context agree within 0.05 ms per component |

The floor defaults to the 256 GB/s datasheet peak, which no kernel reaches. A
grid-stride 16-byte read kernel on gfx1151 (Radeon 8060S) streams 235 GB/s
mean, 240 best, flat from 256 MB to 2 GB; pass `--bw-gbs 235` to rank against
what is reachable. At 256 GB/s gpt-oss's fp16 `lm_head` looks 0.55 ms short of
its floor; at 235 it is at 97%, and only fewer bytes (quantising it) can move it.

Two decode-floor terms were wrong before this and moved the 26B ranking:

- **FFN kernels were filed as `attn_proj`.** SQTT has no shapes, and the FFN,
  q/k/v/o and an int4 router all run `matmul_nbits` kernels once per layer.
  They are now split by position (one GEMV before `gelu` and two after it are
  the FFN; a GEMV followed directly by `softmax_row` is the router) into
  `dense_mlp` and `router`. `dense_mlp`'s floor is the shared MLP beside the
  experts (`dense_inter`), or a dense preset's whole FFN. On 26B `attn_proj`
  went from 5.98 ms at 38% of floor, ranked first, to 3.70 ms at 62%.
- **`kv_cache` charged every layer the full context.** Sliding layers read at
  most the window and global layers have their own head geometry, as
  `headroom.py`'s prefill floor already had it. The 26B 2K KV floor fell from
  578 MB to 258 MB.

Without both, it prints **shares only**, marked `UNVALIDATED (SQTT-relative)`
and naming the missing input. It used to rescale the capture's split onto the
measured ms/token; that charges SQTT's inflated gaps to the kernels, and it
produced the wrong ranking — see [lessons](#sqtt-stretches-gaps-host-overhead-comes-from-the-host-timeline).

A step is bounded by queue switches in the dispatch stream (a VLM decode step
runs on more than one queue), checked against the layer-marker count, and
kernels off the main queue land in `other_queue` rather than vanishing. Every
ranked gain is an estimate; `bench/ab_interleaved.ps1 -Metric tps` decides.

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
carries the kernel timing (but not the gaps — next section). The harness clears
`HIPDNN_EP_PERF`, `HIPDNN_EP_PERF_OPS`, `HIPDNN_EP_DEBUG`,
`HIPDNN_EP_TRACE_FILE` and `HIPDNN_EP_HOST_TIMELINE` before every run rather
than trusting the shell to be clean.

`HIPDNN_EP_PERF_OPS` is the profiler too. Its event pair around every selected
call adds marker overhead and captures intra-op gaps, so an event-timed family
can read *above* its SQTT time (gemma4 26B decode: `matmul_nbits` factor 0.80).
Event times also moved ~3% between processes of the same build (11.2 vs 11.57
ms per step), so neither number is exact; the budget check in
`decode_model.py` is what says whether they are close enough.

`HIPDNN_EP_TRACE_FILE` is the dangerous one. `hipdnn_ep_perf_enabled()` is true
for it as well as for `HIPDNN_EP_PERF`, so it enables the profiler while
printing none of PERF's console output, and it survives in the shell after the
capture that wanted it. A 16K VLM baseline was reported as 17,587 ms for a
whole round of analysis on that basis; the same binaries measure 14,455-14,610
ms once the variable is gone, and re-setting it reproduces 17,545 ms on demand.
Cross-check any suspicious baseline by re-running it in a fresh shell.

### SQTT stretches gaps: host overhead comes from the host timeline

Tracing slows the host far more than the GPU, so the idle between kernels in a
capture is mostly the tracer. On gemma4 26B 2K decode, a CI-env capture spans
27.9 ms per step (21.4 ms measured, x1.31) with 6.1 ms of it idle; the host
timeline on the same build puts 0.84 ms outside the decoder session. Rescaling that capture onto the
measured ms/token spread the phantom idle across the kernels in proportion,
which ranked host overhead and the wrong kernels first. Only
`host_timeline.py` sizes host time, and `decode_model.py` refuses absolute
numbers without it.

### A VLM decode step spans more than one queue

The genai VLM path runs the embedding session on its own queue between decoder
steps. Looking at the decoder queue alone, that time is a hole, and
`decode_model.py` used to count it as GPU idle — more "overhead" to recover
that was really another session's kernels. `decode_step_windows` now bounds
steps where the stream switches queues, checks each against the layer-marker
count (±10%: gemma4 26B runs 29 `topk_routing` in 30 layers), and puts the
other queue's kernels in `other_queue`.

### A standalone microbenchmark is a hypothesis, not a number

Timing each gemma4 GEMV shape outside the model and swapping in the per-shape
best predicted +4-8% decode throughput. The interleaved in-model A/B, with the
model confirmed to run the new configs, measured 20.73 -> 20.85 ms/token (26B)
and 44.20 -> 44.26 (12b): nothing. The standalone times had never been checked
to add up to the model's in-model GEMV time. Check that with `-PerfOps` before
extrapolating, and let `ab_interleaved.ps1` decide.

### Capture the environment you measure

The harness used to set `HIPDNN_EP_AUTOTUNE=1` and
`HIPDNN_EP_MATMUL_CUSTOM_WMMA=1` for every capture. CI sets neither, so
captures showed kernels and configs the benchmarked build never ran. Captures
now inherit the CI environment; `-SetEnv` is the explicit opt-in.

### On an MoE model, synthetic ids are a different workload

`--use_random_tokens` is the default on the text path because it gives an exact
length for free. On a mixture-of-experts model it also changes what you are
measuring, because routing is a function of the embeddings: pass `-PromptFile`
and use real text.

Measured on a gpt-oss-120b 4-layer proxy at a matched 187 tokens, active experts
out of 128 by layer:

| input | layer 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| real prompt | 102 | 101 | 86 | 82 |
| random ids, run 1 | 89 | 72 | 61 | 52 |
| random ids, run 2 | 82 | 69 | 64 | 51 |
| random ids, run 3 | 90 | 72 | 58 | 58 |

Two separate problems. Random ids route *narrower* than text and the gap widens
with depth (−13% at layer 0, −37% at layer 3), and expert breadth is what sets
the per-expert M buckets that `expert_blocks.py` and `headroom.py` rank. They are
also unseeded, so consecutive runs are not the same workload and a capture cannot
be composed with a baseline taken separately.

This is the mild version of the failure. Constant or zero ids — what
`onnxruntime_perf_test -I` supplies — embed every token identically and collapse
routing to top-k outright, giving 4 experts each seeing every token instead of
~100 each seeing a few. That still prints a plausible tokens-per-second number.
Check the count before believing anything:

```
HIPDNN_EP_DEBUG=1   ->   [REAL] wrap_qmoe: 102/128 experts active
```

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

The `lm_head` row is labelled "as executed, all rows" because that assumption is
not universal, and where it does not hold the row inverts into nonsense rather
than failing: a gpt-oss-120b proxy emits a `gather` immediately before the
`lm_head` and runs it at `m=1`, so the measured 7.0 ms sits against a 319.5 ms
all-rows floor and the row reports 4564% utilisation and *negative* recoverable
time. Read that as the optimisation already being present, not as a candidate.
`trace_ops.py --sequence` shows the `m=` on the `n=vocab` matmul directly.

### A floor only bounds anything while its resource is the binding one

`headroom.py` and `expert_blocks.py` score against `max(bytes/BW, FLOP/peak)`. The
bandwidth half of that is only a bound while bandwidth actually binds, and the
parser already measures whether it does. When the achieved rate is *above* the
roofline the traffic is being served from cache, and a floor computed at DRAM
prices is charging for bytes that never reach DRAM.

This is not a subtlety; it produced the worst wrong answer this harness has given.
On a gpt-oss-120b proxy at a 128-token prefill, `headroom.py` ranked small-M MoE
experts first at **54% of prefill**, from a 5.38x bandwidth-floor gap. The same
capture's `bound_reason` column said those kernels were running at **113-219% of
the roofline**. The implied fix was built, verified bitwise correct, and **regressed
TTFT by 4%**: the re-reads it eliminated had been cache hits all along.

Both scripts now read `mem_gbps` back from the dispatch rows and mark any bucket
whose bandwidth floor does not bind, `[BW FLOOR DOES NOT BIND]` in the ranking and
`binds? NO` in the per-bucket table. Treat those rows as traffic volume, not as
recoverable time, and size them from the FLOP floor or from a measured control at
a shape with nothing to re-read instead. The guard is silent where the floor is a
compute floor, which is the usual case at long prompts.

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

### One run at a time, and the harness now enforces it

Every timed script kills competing `model_benchmark` processes before it runs, so
two concurrent invocations do not just contend for the GPU: each aborts the
other's measurement mid-flight, and both append to the same summary CSV. The
result is silent and looks like data.

A triple-launch of one interleaved A/B produced arms with `n=1` and a `nan` in the
paired difference, on a machine whose 16K TTFT read 2018-2586 ms against a known
1110 ms baseline. Each number is plausible on its own.

The kill is scoped: `model_benchmark` and the RGP tools by name, and for
Python drivers only process trees the harness itself started (recorded in
`drivers.txt` under `$HIPEP_OUT` with their start time, so a recycled pid is
never touched). It no longer kills every `python.exe` on the machine.

`Enter-HarnessLock` in `common.ps1` now refuses to start a second run rather than
corrupting both, and clears a lock whose owning pid is gone. It is held by the
outermost script, so `ab_interleaved.ps1` driving `bench_ttft.ps1` stays one
logical run. If you are certain a lock is stale, delete `harness.lock` under
`$HIPEP_OUT`.

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
| `bench/bench_tps.ps1` | One clean decode-throughput run, CI-identical args; `-PerfOps` for in-model family GPU ms |
| `bench/vlm_driver.py` | Run `vlm_benchmark.py` against `HIPEP_BIN` with a prompt file, as CI does |
| `bench/ab_interleaved.ps1` | Interleaved, order-reversed A/B across DLL variants |
| `bench/ab_summary.py` | Paired statistics over the rounds |
| `analysis/perfcommon.py` | Model/device constants and dispatch-stream segmentation |
| `analysis/inspect_capture.py` | What is in this capture, and is it steady state |
| `analysis/attrib_regions.py` | Split kernel time between the MoE loop and dense projections |
| `analysis/expert_blocks.py` | Per-M-bucket expert cost vs floor; which kernel each bucket used |
| `analysis/prefill_model.py` | Two capture depths → whole-prefill composition |
| `analysis/trace_ops.py` | Operator inventory and per-Run structure from an EP chrome trace |
| `analysis/host_timeline.py` | Split a decode step on the host from `HIPDNN_EP_HOST_TIMELINE` |
| `analysis/calibrate_sqtt.py` | Per-family SQTT/event factor, with a dispatches-per-call check |
| `analysis/decode_model.py` | One decode step → per-component traffic vs its memory floor |
| `analysis/headroom.py` | Rank candidates by recoverable seconds (`--decode` for per-token) |

The analysis scripts default to gpt-oss-20b shapes and gfx1151 ceilings; both
are overridable (`--hidden`, `--vocab`, `--layers`, `--bw-gbs`, `--peak-tflops`).
The structural segmentation — regions, layers, expert token counts — is
recovered from the dispatch stream itself and carries no model assumptions.
