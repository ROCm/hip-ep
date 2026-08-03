<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: ort-ep-profiling
description: Profile ONNX models with onnxruntime_perf_test across HIP (hipgpu), DML, and CPU execution providers. Converts ORT profile JSON to op/shape tables and captures HIPDNN_EP_PERF breakdowns. Use when comparing EP performance, running ort profiling, onnxruntime_perf_test -p profile, or generating hipdnn_ep_perf tables. Run one EP at a time; never parallelize profiling runs.
---

# ORT Execution Provider Profiling

Compare per-op performance across **HIP** (hipgpu plugin EP), **DML**, and **CPU** using `onnxruntime_perf_test`. Execute commands yourself; do not only print instructions.

## Serial execution (all EPs)

**Never run profiling for more than one EP at a time** — not in parallel shell commands, not as concurrent background jobs, not via parallel tool calls. This applies to **HIP, DML, and CPU**.

When comparing multiple EPs:
1. Run **one EP at a time**, wait for it to finish completely.
2. Then run the next.

Parallel runs skew timings (GPU contention for HIP/DML; CPU/memory bandwidth contention for CPU) and can cause driver or EP conflicts.

## Required inputs

Ask the user for anything missing before running:

1. **onnxruntime_perf_test path** — full path to `onnxruntime_perf_test.exe` (Windows) or `onnxruntime_perf_test` (Linux).
2. **ONNX model path**
3. **Execution provider** — one of: `HIP`, `DML`, `CPU`
4. **hipgpu.dll path** (HIP only) — if not beside `onnxruntime_perf_test.exe`

Optional: working directory for profile output (default: directory containing the perf_test binary).

## Clean stale profile files (CPU/DML)

`-p profile` writes `profile_<timestamp>.json` into the **current working directory**. Old files make it easy to analyze the wrong run. **Always delete existing `profile_*.json` in `<CWD>` immediately before starting perf_test** (CPU and DML paths).

```powershell
python .cursor/skills/ort-ep-profiling/scripts/clean_profile_files.py "<CWD>"
```

`<CWD>` is usually the directory containing `onnxruntime_perf_test` (where you `Set-Location` before running). Use `--dry-run` to list matches without deleting.

After a successful run, the new trace is the `profile_*.json` file(s) created in that directory — with pre-run cleanup there should be exactly one.

## Base command flags

All runs use:

```text
-I -m times -r 2 -p profile <model>
```

- `-I` — sequential inputs
- `-m times` — measure by iteration count
- `-r 2` — two repetitions (warmup + profiled run)
- `-p profile` — emit Chrome-tracing JSON (`profile_<timestamp>.json`)

## Execution provider commands

### CPU

No `-e` or `--plugin_ep*` flags. Clean profile files first (see above).

```powershell
python .cursor/skills/ort-ep-profiling/scripts/clean_profile_files.py "<CWD>"
& "<PERF_TEST>" -I -m times -r 2 -p profile "<MODEL>"
```

### DML

Add `-e dml` and disable graph fusion for per-op visibility. Clean profile files first (see above).

```powershell
python .cursor/skills/ort-ep-profiling/scripts/clean_profile_files.py "<CWD>"
& "<PERF_TEST>" -I -m times -r 2 -p profile "<MODEL>" `
  -e dml `
  -C "ep.dml.disable_graph_fusion|1"
```

### HIP (hipgpu plugin EP)

Do **not** pass `-e`. Load the hipgpu plugin and enable EP perf printing:

```powershell
$env:HIPDNN_EP_PERF = "1"
& "<PERF_TEST>" -I -m times -r 2 -p profile "<MODEL>" `
  --plugin_ep_libs "hipgpu|<HIPGPU_DLL>" `
  --plugin_eps "hipgpu"
```

- Default `<HIPGPU_DLL>`: `hipgpu.dll` in the same directory as `onnxruntime_perf_test.exe`.
- If missing, ask the user for the full path.
- Ensure ROCm/TheRock runtime DLLs are on `PATH` when running HIP (see repo `docs/quick_start.md`).

## Post-run outputs

### HIP — extract the **last** `[PERF]` table from perf_test output (not profile JSON)

The hipgpu plugin EP does **not** produce useful per-op metrics in the ORT `profile_*.json` trace. With `$env:HIPDNN_EP_PERF=1`, hip-ep prints a full op/shape table **once per inference**. Because `-r 2` (plus compile/warmup runs) repeats inference, the log contains **multiple** tables — always report **only the last one** (final timed repetition, after warmup/compile).

Capture stdout+stderr (PowerShell `Tee-Object` writes UTF-16; the extractor handles that):

```powershell
& "<PERF_TEST>" ... 2>&1 | Tee-Object -FilePath "<LOG.txt>"
```

Extract the last table with the helper script (do not hand-pick blocks from earlier repetitions):

```powershell
python .cursor/skills/ort-ep-profiling/scripts/extract_hip_perf_table.py "<LOG.txt>"
python .cursor/skills/ort-ep-profiling/scripts/extract_hip_perf_table.py "<LOG.txt>" -o "<TABLE.txt>"
```

The script finds separator-delimited blocks that contain the `calls  gpu (ms)  cpu (ms)  gpu %` header and returns the final block only.

Report that block to the user. Ignore `profile_*.json` for HIP.

### CPU and DML — convert ORT profile JSON

After pre-run cleanup and a successful perf_test, locate the new trace in `<CWD>`:

```powershell
Get-ChildItem -Path "<CWD>" -Filter "profile_*.json"
```

There should be one file. If multiple exist (unexpected), take the newest by `LastWriteTime` and note the ambiguity in the report.

Convert to a summarized op/shape table in the same grouping as the hip-ep `[PERF]` table (op type → shape detail):

```powershell
python .cursor/skills/ort-ep-profiling/scripts/ort_profile_table.py "<PROFILE_JSON>" -o "<OUTPUT.tsv>"
```

Table columns: `Op`, `Detail`, `Calls`, `Time (ms)`.

The script also prints an **op-type summary** (`Op`, `Calls`, `Time (ms)`) sorted by total time descending — include this when reporting CPU/DML results.

To print to stdout, omit `-o`.

## Workflow checklist

```
Task Progress:
- [ ] Collect perf_test path, model path, EP choice
- [ ] If profiling multiple EPs: schedule runs sequentially (never parallel)
- [ ] Resolve hipgpu.dll (HIP only)
- [ ] CPU/DML: clean profile_*.json in CWD before running
- [ ] Set HIPDNN_EP_PERF=1 (HIP only)
- [ ] Run onnxruntime_perf_test with EP-specific flags
- [ ] HIP: capture perf_test output and extract last [PERF] table (ignore profile JSON)
- [ ] CPU/DML: locate new profile_*.json and run ort_profile_table.py
- [ ] Report both summary table and total inference time if available
```

## Report back

Summarize for the user:

1. EP profiled and command used
2. Profile artifact path (`profile_*.json` or perf log)
3. Per-op table — `[PERF]` block (HIP) or TSV from `ort_profile_table.py` (CPU/DML), including the op-type summary with **calls** and total time per op
4. Any errors (missing DLL, EP registration failure, model load failure)

## Notes

- **Never run more than one EP profiling session concurrently** (HIP, DML, or CPU); run sequentially and wait for each to exit.
- HIP profile JSON from `-p profile` is not useful for hip-ep op breakdown; always use the `[PERF]` stderr table.
- `-r 2` prints multiple `[PERF]` tables; use `extract_hip_perf_table.py` and never report an earlier repetition.
- Do not use `HIPDNN_EP_PERF=1` when measuring raw throughput; it adds profiling overhead. This skill intentionally enables it for HIP op breakdowns.
- For DML, graph fusion is disabled so fused ops appear as individual nodes in the trace (closer to HIP/CPU granularity).
- Always run `clean_profile_files.py` before CPU/DML perf_test; never convert a pre-existing `profile_*.json` without confirming it came from the current run.
- Quote paths that contain spaces.

## Scripts

- [scripts/ort_profile_shapes.py](scripts/ort_profile_shapes.py) — ONNX op → shape detail formatters (aligned with `lib/Runtime/real/` OP_PROFILE).
- [scripts/clean_profile_files.py](scripts/clean_profile_files.py) — delete stale `profile_*.json` before a CPU/DML run.
- [scripts/ort_profile_table.py](scripts/ort_profile_table.py) — converts ORT profile JSON to op/shape TSV (CPU/DML).
- [scripts/extract_hip_perf_table.py](scripts/extract_hip_perf_table.py) — extracts the **last** `[PERF]` table from a HIP perf_test log.
