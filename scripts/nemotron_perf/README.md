# Nemotron 3.5 Lightning performance checkpoint

This local development branch starts at main revision `04f1c0365`.
It is separate from the shape-inference PR stack.
CPU fallback is prohibited. Do not run llama.cpp for these checks.

## Scope

The checkpoint includes:

- Sparse expert fusion. Each token computes its selected experts.
- Grouped expert prefill and W4A8 expert decode tiles.
- Cooperative TopK and removal of inline constant scalar readbacks.
- FP16 recurrent-state boundaries with FP32 attention calculations.
- Compact Q/K heads and fixed-geometry linear-attention decode.
- SwiGLU fusion with the original FP32 rounding boundaries.
- A sequential FP32 router for the pinned matrix geometry.

The sparse rewrite still has an experimental gate. Inactive expert results
must be finite. Finite inputs alone do not prove this: intermediate values
can overflow. The tested logits agree with main, but this is not proof for
all inputs. Establish the compiler's numerical contract before enabling the
rewrite by default.

The checkpoint excludes the rejected split-router, HPG16 WMMA attention,
fixed-K dense, activation-sum reuse, and packed-wave experiments.
Their source and evidence remain under the ignored `.nemotron-perf/`
directory. They are not part of the qualified configuration.

Three small HIPSR compatibility edits permit the local LLVM 24 build.
They do not add HIPSR functionality or change the shape-inference stack.

## Local dependencies

Model:

`/scratch/fhanuman/RCOmEP/oga_models/Nemotron-3.5-Lightning-30B-A3B-fp16-int4-onnx`

The runtime uses ORT 1.31 and OGA 0.17 from the pinned main package.
It runs in `nemotron-main-runtime:20261020` because the package needs a
newer GLIBC than the host provides. The launcher mounts the existing zstd
fix read-only. It does not replace the host SDK or modify the source model.

These scripts are local investigation tools. Their default paths refer to
this workspace and its frozen runtime installations.

## GPU coordination and build

All GPU runs use this existing lock:

`/scratch/fhanuman/RCOmEP/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock`

Do not delete or replace it. The launcher acquires it, waits five seconds,
checks GPU availability, and releases it after the process exits. Builds
must also hold the lock because the CPU and GPU share the APU power budget.

Build without changing shared dependencies:

```sh
(
  exec {gpu_fd}< /scratch/fhanuman/RCOmEP/hrxWork/.cursor/artifacts/hrx-backend/gpu.lock
  flock --exclusive "$gpu_fd"
  bash scripts/nemotron_perf/build.sh
)
```

The build uses LLVM 24 and writes to `.nemotron-perf/build` and
`.nemotron-perf/install`. Frozen installations must never be overwritten.

## Run the qualified core

```sh
bash scripts/nemotron_perf/run.sh \
  --runtime-dir .nemotron-perf/runtime-checkpoint-core-v1 \
  --sparse-experts --la-f16-state --la-compact-heads --swiglu-f32 \
  --router-sequential-f32 --la-fixed-geometry \
  --no-matmul-autotune \
  --blas-tuning-override .nemotron-perf/control/router1278-head914.csv \
  --prompt "Explain why the sky is blue." --gen 128 --warmup 1 --reps 3
```

Use `--prompt-file PATH` to keep exact benchmark prompt text without adding
a chat template. `--gen` selects new tokens, not total sequence length.
Every run creates fresh staging and records model, compiler, kernel, and
configuration hashes. CPU fallback is disabled explicitly.

The BLAS override holds kernel choices constant for comparison. Solution
IDs are specific to the pinned library and GPU; do not reuse them elsewhere.

## Validation and measurement

The rebuilt core checkpoint matches all 6,291,456 main logits in the short, 2K, and 16K
captures. Each capture uses 16 steps, 131,072 logits per step, and the same
forced token history. This evidence does not certify all prompts or models.

The four compiler matcher suites, three lowering FileCheck tests, and five
GPU kernel suites also pass. The local checkpoint record is
`.nemotron-perf/checkpoint-source-20261021/core-validation.md`.

Two clean 2K pairs, in opposite orders, measured 56.12-56.16 TPS for v20
and 55.74 TPS for v18. The sampled device footprint was about 17.515 GiB.

Use `capture_logits.cpp` and `compare_logits.py` for numerical checks.
The `test_*fusion.py` and `test_linear_attention_heads.py` scripts test
compiler matchers. The selected `test_*.sh` scripts test GPU kernels.

Keep diagnostics separate from timings. `--perf`,
`--kernel-selection-log`, and `--ir-dump-tree --ir-dump-after-only` are
diagnostic options. Use `--output-dir` for a fresh evidence directory
without enabling diagnostics. Use `summarize_runs.py` to summarize clean
runs. Memory is sampled whole-device VRAM plus GTT, not exact model storage.

The published Vulkan reference uses different GGUF quantization and prompt
generation. It is not a controlled compiler A/B. The gap remains open.

## Retention

Keep frozen runtimes, raw logs, manifests, numerical comparisons, and
decisive IR snapshots locally. Do not commit model links, binaries, traces,
caches, or generated benchmark output. The historical investigation journal
and discarded probes stay local and outside the checkpoint commits.
