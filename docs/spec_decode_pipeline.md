<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# EAGLE-3 / MTP enablement pipeline

Brings up a speculative-decoding stack on the hip-ep EP and measures it, in the
order you want when enabling a new target or a new draft head:

```
preflight -> export -> sanity -> bench -> summarize
```

`scripts/run_spec_decode_pipeline.py` drives it.

---

## TL;DR

```bash
# what would run, without running it
python scripts/run_spec_decode_pipeline.py --preset qwen36-27b --dry-run

# check the stack is complete and every arm loads
python scripts/run_spec_decode_pipeline.py --preset qwen36-27b \
    --harness ../tools/eagle3 --models-root .. --stages preflight,sanity

# the full bake-off
python scripts/run_spec_decode_pipeline.py --preset qwen36-27b \
    --harness ../tools/eagle3 --models-root .. --json results/phase_t_p1.json
```

---

## The harness is out of tree

This script is an **orchestrator, not an implementation**. The speculative
decoding itself — the target and draft runners, the Spec-Bench question set and
subsetting, the exporters — lives in `tools/eagle3` in the speculative-decoding
workspace, not in this repository. Point at it with `--harness` or
`$SPEC_HARNESS`, and at the directory the preset's `models/...` paths resolve
against with `--models-root` or `$SPEC_MODELS_ROOT`.

Nothing is reimplemented here. What this adds is the ordering, the
preconditions, and presets that reproduce the recorded bake-offs.

---

## Why the stages are split

A full bake-off is hours, and most of that wall clock is not inference. hip-ep
compiles one autotune key per novel tensor shape, so every distinct prompt
length pays a first-encounter compile; in the Phase S and Phase T runs warming
was 82–84% of total wall time. Discovering a mistyped head directory or a target
graph without the right aux taps at hour three is exactly the failure this
pipeline exists to prevent.

| Stage | Cost | Does |
|---|---|---|
| `preflight` | seconds | Checks every file the run will open, and reports all of them rather than stopping at the first |
| `export` | minutes | Builds the MTP head if it is not already beside the target |
| `sanity` | minutes | One instance per subtask, 16 tokens, every arm — does it all load and decode? |
| `bench` | hours | The real bake-off, writing JSON after every question |
| `summarize` | seconds | Re-prints the tables from that JSON |

Default is everything except `export`. `summarize` is separate from `bench` so a
finished run can be re-read without touching the GPU, and an interrupted one can
still be summarized from whatever it wrote.

Only MTP is exported. An EAGLE-3 head is the output of a training and
quantization pipeline with model-specific knobs (aux taps, rope mode, vocab
pruning), so exporting one is not something this orchestrator can honestly
default; if a head directory is missing, `preflight` says so.

---

## Presets

```bash
python scripts/run_spec_decode_pipeline.py --preset <name> --list-presets
```

| Preset | Target | Arms |
|---|---|---|
| `qwen35-9b` | `amd_hipep_qwen35_9b/text_hipep_spec.onnx` | `none`, `mtp:1:mp2`, EAGLE-3 `blr2_plain` / `incumbent` / `blr2_sharegpt` at `cap2` |
| `qwen36-27b` | `qwen3.6-27b-int4-block-128/model_hipep_eagle3.onnx` | `none`, `mtp:1:mp6`, EAGLE-3 `incumbent` / `prism_full` at `cap2` |
| `qwen36-27b-specdrift` | `…/model_hipep_eagle3_sd.onnx` | `none`, `mtp:1:mp6`, EAGLE-3 `specdrift` at `mp3` |

The per-arm policies are not defaults anyone would guess — MTP wants `mp2` on
the 9B and `mp6` on the 27B, EAGLE-3 wants `cap2` except specdrift, which wants
`mp3`. They came out of the break-even curves, and they are baked in so a re-run
means the same thing as the recorded results.

Specdrift is a **separate preset rather than a flag** because it taps layers
L4/32/60 while the other two heads tap L2/32/61. That needs a different target
graph, so it cannot share a process with them. MTP is carried in both so the two
processes can be read against a common arm — in Phase T they agreed to 0.2%.

### Arm syntax

Passed through to the harness with `--arms`:

```
none                          baseline, no drafter
mtp:<k>[:policy]              MTP head, chain length k
eagle3[@<tag>]:<k>[:policy]   EAGLE-3 head <tag> from the preset
```

where policy is `mp<N>` (max pending) or `cap<N>` (width cap), default `mp8`.
Every arm runs back to back inside one question against one loaded target, so
prompt difficulty is removed from the contrast rather than averaged over.

---

## Options

| Option | Default | Meaning |
|---|---|---|
| `--preset` | required | see above |
| `--harness` | `$SPEC_HARNESS` | directory holding `qwen36_spec_bench.py` |
| `--models-root` | `$SPEC_MODELS_ROOT` | root the preset's `models/...` resolve against |
| `--stages` | all but `export` | subset of the five stages |
| `--json` | — | results file; required by `bench` and `summarize` |
| `--arms` | per preset | override the arm list |
| `--per-task` | per preset | instances per subtask (six subtasks, 80 available each) |
| `--max-new` | per preset | tokens generated per prompt |
| `--ep` | `hipgpu` | `cpu`, `hipgpu` or `amdgpu` |
| `--no-iobinding` | off | disable IO binding |
| `--dry-run` | off | print each stage's command and exit 0 |

Anything after the flags is passed straight through to `qwen36_spec_bench.py`,
so `--resume` and friends still work.

---

## What the results look like

Pooled decode throughput, reproducible with `--stages summarize` against the
recorded JSON:

**Qwen3.6-27B** (48 instances, baseline 10.17 tok/s)

| Arm | tok/s | Speedup | Depth-1 acceptance |
|---|---|---|---|
| `none` | 10.17 | 1.00x | — |
| `mtp:1:mp6` | 15.81 | 1.55x | 84% |
| `eagle3@incumbent:1:cap2` | 14.61 | 1.44x | 67% |
| `eagle3@prism_full:1:cap2` | 13.60 | 1.34x | 68% |

**Qwen3.5-9B** (120 instances, baseline 29.72 tok/s)

| Arm | tok/s | Speedup | Depth-1 acceptance |
|---|---|---|---|
| `none` | 29.72 | 1.00x | — |
| `mtp:1:mp2` | 35.04 | 1.18x | 80% |
| `eagle3@blr2_plain:1:cap2` | 38.01 | 1.28x | 67% |
| `eagle3@incumbent:1:cap2` | 37.23 | 1.25x | 67% |
| `eagle3@blr2_sharegpt:1:cap2` | 32.05 | 1.08x | 40% |

Two things worth reading off these. **Acceptance does not rank the arms**: MTP
accepts 84% on the 27B and 80% on the 9B, well ahead of every EAGLE-3 head, yet
loses to `blr2_plain` on the 9B — a cheaper draft step lowers the break-even
accept length, and EAGLE-3's pruned vocabulary buys more than its lower
acceptance costs. And **the 27B gets more out of speculation than the 9B on
every subtask**: its worst arm beats the 9B's best.

Speedups also vary by subtask far more than the pooled number suggests, so a
single-prompt measurement will mislead you. Run the whole benchmark.

---

## Runtime

Dominated by autotune warming, not tokens. Rough figures from the recorded runs:

| Configuration | Instances | Wall clock |
|---|---|---|
| 9B, 5 arms | 120 | ~4 h |
| 27B, 4 arms | 48 | ~9 h |
| Full Spec-Bench, 480 instances | 480 | ~19 h cold, ~17 h warm cache |

`--per-task` is the knob: it takes the first N instances of each of the six
subtasks deterministically, so a subset is comparable across runs.

---

## Caveats

**Not a correctness test.** The bench reports whether each arm's output was
token-identical to the no-drafter arm, but speculative decoding is only
greedy-equivalent when the drafter is verified exactly; treat divergences as
signal about the verify path, not as a pass/fail gate.

**Disk space matters.** Phase T recorded baseline spread ballooning from 0.5% to
79.5% when free space fell to 0.2%. Check before starting a nine-hour run.

**Intermittent MLIR codegen flake.** A run can abort at startup with
`IMAGE_REL_AMD64_ADDR32NB relocation requires an ordered section layout`. It
fails before producing tokens, so a restart does not bias timings.

---

## See also

- [`quick_start.md`](quick_start.md) — building the EP
- [`supported-operations.md`](supported-operations.md) — operator coverage
