<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIPDNN EP Profiling Report Formatter

`format_perf_report.py` parses a `HIPDNN_EP_PERF=1` log produced by
`model_benchmark` driving the MorphiZen EP and renders a locked-down,
section-stable profiling report.

> **Scope: Linux Docker build.** The example invocations below use the
> Linux paths (`$ROOT/bin/`, `$ROOT/lib/libonnxruntime_morphizen_ep.so`)
> from the Docker build layout documented in
> [`docs/quick_start_linux.md`](../../docs/quick_start_linux.md).
> The script itself is pure Python 3 (stdlib only) and will run on any
> platform with Python ≥ 3.10, but the bench-side command shown under
> [Usage](#usage) is specific to the Linux build.

## Why

When `HIPDNN_EP_PERF=1` is set, the EP and runtime emit three independent
measurement streams that are interleaved with `model_benchmark`'s own
output:

| Stream | Emitted by | Cadence |
|---|---|---|
| `[PERF SUMMARY]` per-Compute aggregates (n, min/median/p99/max) | `MlirCustomOp::PerfCollector::dump_summary` (`backend-mlir-compiler/custom-op-mlir/src/MlirCustomOp.cpp`) | once at library unload |
| `[PERF] === ... ===` per-op GPU/CPU tables | `op_profile_resolve_and_print` (`lib/Runtime/op_profile.cpp`) | once per `Compute()` call |
| `model_benchmark` stats block (Prompt processing / Token generation / Token sampling / E2E) | `model_benchmark` | once per run |

Skim-reading the raw log to correlate these three views is painful. This
script does it for you: it locks the output into four sections with stable
header markers (`§ 1 HEADLINE`, `§ 2 STEADY-STATE DECODE BREAKDOWN`,
`§ 3 PER-OP GPU BREAKDOWN`, `§ 4 PER-CALL DISTRIBUTION`) and prints a
single one-line banner at the top and bottom of the report so external
tooling (CI checks, perf-regression bots) can grep for stable anchors.

See [`docs/design/per-op-profiling.md`](../../docs/design/per-op-profiling.md)
for the underlying instrumentation design.

> **Tool coverage.** The script needs the `model_benchmark` stats block
> (stream 3 above) to render any report — § 1 / § 2 / footer all depend
> on it, and the CLI exits with code 2 if it's absent. Logs from
> `onnxruntime_perf_test` or pytest contain streams 1 + 2 but no
> compatible § 3 — they're not supported today.

## Usage

The full `model_benchmark` invocation (including the `-ml` rules for
fixed-shape pipelines, how `$ROOT` is set up by the Docker build, and
the canonical bench env vars) is documented in
[`docs/quick_start_linux.md`](../../docs/quick_start_linux.md). The
bare-minimum two-step is:

```bash
# 1) Run model_benchmark with HIPDNN_EP_PERF=1, capturing stderr+stdout
#    into a single log file (all three [PERF*] streams go to stderr).
#    $ROOT and $WORKSPACE are set up by docs/quick_start_linux.md when
#    you `source` the in-container shell helpers; don't hardcode them.
HIPDNN_EP_PERF=1 $ROOT/bin/model_benchmark \
    -i /path/to/oga-model-dir \
    --ep_library MorphiZenEP $ROOT/lib/libonnxruntime_morphizen_ep.so \
    -l 128 -g 128 -ml -1 -r 3 -w 1 \
    > run.log 2>&1

# 2) Format the report:
python3 tools/perf-report/format_perf_report.py run.log
```

The script file is marked executable (`100755` in git), so the shebang
form also works:

```bash
./tools/perf-report/format_perf_report.py run.log
```

Flags:

| Flag | Effect |
|---|---|
| `--ascii` | Use ASCII tree-drawing characters instead of Unicode (for terminals or pipelines that don't render box-drawing glyphs cleanly) |
| `--no-banner` | Suppress the top/bottom banner lines (useful when embedding the report inside another report) |
| `--indent N` | Indent every output line by N spaces |

Exit codes:

| Code | Meaning |
|---|---|
| `0` | Report rendered. Any subset of the four sections may be present (each self-gates on its source data) |
| `1` | Log file not found |
| `2` | Log lacks the `model_benchmark` stats block (no `Batch size:` line). Caller should fall back to `tail -n 100 <log>` or similar |

## Output sketch

Real output produced by the script — units and headers are reproduced
verbatim from a live `-l 128 -g 128 -r 3 -w 1` run:

```
══════════════════════════════════════════════════════════════════════════
  HIPDNN EP profile  ·  <model-name>  ·  prompt=128 gen=128  ·  inferences=639
══════════════════════════════════════════════════════════════════════════

  § 1  HEADLINE  —  model_benchmark (batch=1, prompt=128, gen=128)
  ──────────────────────────────────────────────────────────────────────────
                          throughput     p50 latency        stddev   samples
    Prefill (TTFT)        770.66 t/s    166048.58 us    1888.95 us   3 * 128 token(s)
    Decode                 39.68 t/s     25066.96 us    1486.80 us   381 * 1 token(s)
    Sampling            14527.63 t/s        68.15 us       1.74 us   3 * 1 token(s)
    E2E generation                 —      3368.69 ms      12.82 ms   3
    Peak working set   1.52 GiB (1633189888 bytes)

  § 2  STEADY-STATE DECODE BREAKDOWN  —  1 decode token, median across run
  ──────────────────────────────────────────────────────────────────────────
                                                  latency    share   source
    OGA::GenerateNextToken                      25.067 ms   100.0 %   ◀ § 1 Decode p50
    ├─ OGA + ORT framework overhead              1.028 ms     4.1 %
    │  ├─ Token sampling                         0.069 ms     0.3 %   ◀ § 1 Sampling avg
    │  └─ Other (IoBinding rebind, etc.)         0.959 ms     3.8 %
    └─ EP MlirCustomOp::Compute()               24.039 ms    95.9 %   ◀ § 4 wall_ms
       ├─ ...
       └─ Trailing fence (hipStreamSync)         0.009 ms     0.0 %   ◀ § 4 fence_residual_ms

  § 3  PER-OP GPU BREAKDOWN  —  last Compute() = steady-state decode token
  ──────────────────────────────────────────────────────────────────────────
    ===============================================================
                                   calls  gpu (ms)  cpu (ms)  gpu %
     matmul_nbits                    225      20.0       0.6  86.9%
       m=1,n=14336,k=4096             64      10.0       0.2  43.3%
       ...
     TOTAL                                    23.0       1.8
    ===============================================================

  § 4  PER-CALL DISTRIBUTION  —  EP MlirCustomOp::Compute() over 639 invocations (all ms)
  ──────────────────────────────────────────────────────────────────────────
                              min     median        p99          max
    wall_ms                23.703     24.039     42.869     8008.406
    compute_cpu_ms         23.635     23.966     42.774     8003.431
    gpu_ms                 23.593     23.919     42.729     8003.326
    ...

══════════════════════════════════════════════════════════════════════════
  [OGA] prefill=770.66 tok/s  TTFT=166.09 ms  decode=39.68 tok/s  peak_WS=1.52 GiB (1633189888 bytes)
══════════════════════════════════════════════════════════════════════════
```

Each section is self-gating: if its source data is absent from the log
(e.g. `--mode none` runs without per-op tables, or older builds that
predate `[PERF SUMMARY]`), the section is silently omitted instead of
printing an empty header.

The bottom banner is a single greppable line — `^\[OGA\] prefill=` —
intended for CI / regression-bot consumption.

## Dependencies

Python ≥ 3.10, standard library only. No third-party packages.
