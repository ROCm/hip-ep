<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP EP Profiling Report Formatter

This directory holds three cooperating tools that turn a single
`HIPDNN_EP_PERF=1` bench run into a structured, section-stable profiling
report:

| File | Purpose | Portability |
|---|---|---|
| [`format_perf_report.py`](format_perf_report.py) | Parses a captured log and renders the four-section report. The value-add tool. | Pure Python 3, stdlib only — any platform with Python ≥ 3.10. |
| [`run_bench.sh`](run_bench.sh) | Bench driver. Runs `onnxruntime_perf_test` (default), `model_benchmark` (`--oga`), or `model_mm` (`--mm`, multimodal/VLM) against a model under `$WORKSPACE/oga_models/<dir>` and captures the log under `tools/perf-report/_perf_logs/`. The `--oga` path pipes the log through `format_perf_report.py`; `--mm` surfaces `model_mm`'s headline timing line inline and (under `--mode perf`) renders the per-op breakdown via `perf_multimodal_report.py`; the perftest path prints its own grep-based summary inline. | Linux + AMD GPU + the in-container build artefacts at `$WORKSPACE/install/{bin,lib}`. |
| [`docker_run_bench.sh`](docker_run_bench.sh) | Thin host-side wrapper that runs `run_bench.sh` (or any other repo-relative bench script via `REL_BENCH=...`) inside the `hip-ep-build` Docker image as a `docker run --rm` one-shot — no interactive shell needed. | Linux + Docker + an AMD GPU exposed via `/dev/kfd` and `/dev/dri/renderD*`. |

> **Scope: Linux Docker build.** The bench-side examples below use the
> Linux paths (`$ROOT/bin/`, `$ROOT/lib/libonnxruntime_morphizen_ep.so`)
> from the Docker build layout documented in
> [`docs/quick_start_linux.md`](../../docs/quick_start_linux.md).
> `format_perf_report.py` itself has no Linux assumptions and can be run
> against a log captured from any platform; `run_bench.sh` and
> `docker_run_bench.sh` are Linux / Docker specific.

> **Models are not downloaded.** `run_bench.sh` expects a model directory
> to already exist at `$WORKSPACE/oga_models/<dir>/` with `model.onnx`,
> `model.onnx.data`, `tokenizer.json`, `tokenizer_config.json`, and
> `genai_config_MorphiZenEP.json`. Stage one with e.g.
> `huggingface-cli download <repo> --local-dir $WORKSPACE/oga_models/<dir>`
> before invoking. `$WORKSPACE` defaults to the parent of the repo root,
> matching the `docker/run.sh` convention.

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

The three tools compose into three layers of increasing automation.
Pick the entry point that matches your situation.

### Layer 1: format only (any host with a captured log)

If you already have a `HIPDNN_EP_PERF=1` log produced anywhere, just
render it:

```bash
python3 tools/perf-report/format_perf_report.py run.log
```

`format_perf_report.py` is marked executable (`100755` in git), so the
shebang form also works:

```bash
./tools/perf-report/format_perf_report.py run.log
```

### Layer 2: run the bench inside the container shell

From inside the build container (`./docker/run.sh shell`), the bench
driver wires up env (`LD_LIBRARY_PATH`, `THEROCK_DIST`, optional
`HIPDNN_EP_PERF=1`) and captures the log under
`tools/perf-report/_perf_logs/`. On the `--oga` path it additionally
pipes the log through `format_perf_report.py` before exiting:

```bash
# OGA path -- model_benchmark, prompt=128 gen=128, three reps, one warmup,
# in --mode perf (sets HIPDNN_EP_PERF=1 so all four report sections render):
tools/perf-report/run_bench.sh --oga --prompt 128 --gen 128 --reps 3 --warmup 1 --mode perf

# perftest path -- onnxruntime_perf_test, decode-only dynamic, 30s window:
# (prints its own P50 / [PERF SUMMARY] / last-per-op-table grep summary;
#  does NOT auto-invoke format_perf_report.py because perftest output has
#  no model_benchmark stats block.)
tools/perf-report/run_bench.sh --time 30 --mode perf

# Different model (must already be staged at $WORKSPACE/oga_models/<dir>/):
tools/perf-report/run_bench.sh --oga --model Mistral-7B-Instruct-v0.3-dml-int4-awq-block-128 --mode perf
```

#### Multimodal / VLM models (`--mm`)

`--mm` drives `install/bin/model_mm` (the OGA C++ multimodal example) with an
image + prompt instead of `model_benchmark`. Unlike `--oga` there is no staged
dynamic dir — `model_mm` reads `genai_config.json` from the model dir directly,
so the dir's active config must already select MorphiZenEP (pass `--variant <v>`
to swap `genai_config_<v>.json` into place for the run; it is restored on exit):

```bash
# profiler OFF (true throughput) and ON (per-op breakdown), via the host wrapper:
tools/perf-report/docker_run_bench.sh --mm \
    --model <vlm-dir> --image eiffel.jpg --prompt "What is in this image?" \
    --mode none
tools/perf-report/docker_run_bench.sh --mm \
    --model <vlm-dir> --image eiffel.jpg --prompt "What is in this image?" \
    --mode perf
```

> **`model_mm` ignores `-g`/`--max_new_tokens`** (that flag is honored only by
> the `model_chat` example): generation runs until EOS, bounded only by
> `search.max_length` in `genai_config.json`. A full VLM generation under
> `--mode perf` dumps a per-op table after **every** token, which is slow and
> produces a large log. To cap the token count for a `--mode perf` run, point
> `--variant` at a config whose `search.max_length` is set to
> `(prompt_len + desired_new_tokens)`.

`run_bench.sh --help` prints the full flag list. Logs land at
`tools/perf-report/_perf_logs/<model>_<tool>_<...>_<ts>.log` — keep them
for postmortem, re-render any of them later with
`format_perf_report.py <log>`.

The raw `model_benchmark` invocation (which `run_bench.sh --oga` builds
internally — `-l`/`-g`/`-r`/`-w`/`-b`/`-v`, optional `-ml`, EP loaded
via CWD discovery from `$ROOT/lib`) is fully documented in the
[`docs/quick_start_linux.md`](../../docs/quick_start_linux.md) "Open a
container shell" section if you need to invoke it directly.

### Layer 3: run the bench from the host (no shell needed)

For a clean one-shot from any host terminal (including Cursor's embedded
terminal, where `docker exec -it` is unreliable), the wrapper spins a
`docker run --rm` container, exec's the bench inside it as your host
UID/GID with the GPU and bind mounts attached, and exits when it's done.
See the script header for the failure modes it works around (Cursor TTY
behaviour, the `docker/entrypoint.sh` `getent` crash, zombie-container
reaping).

```bash
# Equivalent to Layer 2's first example, but runs from the host:
tools/perf-report/docker_run_bench.sh --oga --prompt 128 --gen 128 --reps 3 --warmup 1 --mode perf
```

The wrapper is an unopinionated relay — it forwards all CLI args
verbatim to `$REL_BENCH` (default `tools/perf-report/run_bench.sh`).
Override `REL_BENCH` to invoke a different in-container bench:

```bash
REL_BENCH=oga/my_local_harness.sh \
    tools/perf-report/docker_run_bench.sh --foo --bar
```

Other env knobs (`WORKSPACE`, `IMAGE`) are documented in the script
header.

### Flags (`format_perf_report.py`)

| Flag | Effect |
|---|---|
| `--ascii` | Use ASCII tree-drawing characters instead of Unicode (for terminals or pipelines that don't render box-drawing glyphs cleanly) |
| `--no-banner` | Suppress the top/bottom banner lines (useful when embedding the report inside another report) |
| `--indent N` | Indent every output line by N spaces |

### Exit codes (`format_perf_report.py`)

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
  HIP EP profile  ·  <model-name>  ·  prompt=128 gen=128  ·  inferences=639
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

| Tool | Required by | Notes |
|---|---|---|
| Python ≥ 3.10 | `format_perf_report.py` | Standard library only; no third-party packages. |
| Bash + `git` + `python3` | `run_bench.sh` | All present in the `hip-ep-build` image. `git` is used only for the PR #212 reachability check, which silently no-ops if `git` is unavailable. |
| `$WORKSPACE/install/{bin,lib}` populated | `run_bench.sh` | `libonnxruntime_morphizen_ep.so` + `model_benchmark` (for `--oga`) + `onnxruntime_perf_test` (for the default path). Produced by `./docker/run.sh build` ([`docs/quick_start_linux.md`](../../docs/quick_start_linux.md)). |
| `$WORKSPACE/oga_models/<dir>/` staged | `run_bench.sh` | Model directory with `model.onnx`, `model.onnx.data`, `tokenizer.*`, `genai_config_MorphiZenEP.json`. `run_bench.sh` does NOT download — see the "Models are not downloaded" callout above. |
| Bash + `docker` CLI | `docker_run_bench.sh` | Tested with Docker Engine on Linux. Rootless Docker works as long as the engine can mount `/dev/kfd` and `/dev/dri/renderD*`. |
| AMDGPU driver + `hip-ep-build` image | `docker_run_bench.sh` | The image is the one built by `./docker/run.sh image` ([`docs/quick_start_linux.md`](../../docs/quick_start_linux.md)); image tag is overridable via `IMAGE`. |
