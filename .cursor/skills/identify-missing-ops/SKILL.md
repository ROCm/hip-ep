<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: identify-missing-ops
description: >-
  Check ONNX model compatibility with hip-ep by running identify_missing_ops.py
  (full compile check via onnxruntime_perf_test, static ONNX scan, or MLIR/log
  re-parse). Use when the user asks to identify missing ops, check hip-ep
  compatibility, run /identify-missing-ops, or diagnose ONNX compile failures
  on hip-ep.
---

# identify-missing-ops

Run [scripts/identify_missing_ops.py](scripts/identify_missing_ops.py) to check ONNX model compatibility with hip-ep. Execute the commands yourself; do not only print instructions.

## Model input

Use the path(s) the user typed after this command (e.g. `/identify-missing-ops model.onnx`). If none was provided, ask for an ONNX file or directory.

Resolve paths relative to the workspace root. Quote paths that contain spaces or parentheses.

## Discover paths (do not hardcode)

1. **Script** — `.cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py` (relative to workspace).
2. **gpu-test-package** — locate a directory that contains `bin/onnxruntime_perf_test` or `bin/onnxruntime_perf_test.exe`:
   - `$env:PACKAGE_DIR` (Windows) / `$PACKAGE_DIR` (Unix), if set
   - Search workspace for `gpu-test-package*` siblings
   - If cwd is `*/bin`, use the parent directory
3. **Log dir** — use a subdirectory under the workspace (e.g. `output_test/`). Create it if missing.

If gpu-test-package is not found, use `--scan-onnx` for a static-only check and tell the user full compile needs the package.

## Run

**Full compile check** (preferred when gpu-test-package exists):

```powershell
python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py "<MODEL>" `
  --package-dir "<PACKAGE_DIR>" `
  --dump-mlir `
  --log-dir "<LOG_DIR>" `
  --show-next-step
```

Run from `gpu-test-package/bin` if that simplifies package resolution; otherwise pass `--package-dir` explicitly.

**Static scan only** (no GPU / no perf_test):

```powershell
python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py --scan-onnx "<MODEL>" --show-next-step
```

**Batch directory**:

```powershell
python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py "<MODEL_DIR>" --glob "*.onnx" `
  --package-dir "<PACKAGE_DIR>" --dump-mlir --log-dir "<LOG_DIR>" --show-next-step
```

Do not pass `--allow-cpu-fallback` unless the user explicitly requests it.

## Report back

Summarize for the user:

1. **hip-ep result** — ok / compile failed / unsupported ops
2. **Failure stage** and **Failure reason** (if any)
3. Blockers by category: compile, shape, variant, dtype, partial, unsupported
4. **Next step** column when `--show-next-step` was used
5. MLIR dump directory path (if `--dump-mlir` was used)

### Category meanings

| Section | Meaning |
|---------|---------|
| Compile blockers | No OnnxToHip converter |
| Shape blockers | Converter exists; shape limits block lowering (e.g. Conv with dynamic H/W) |
| Variant blockers | Unsupported attributes or rank |
| Dtype blockers | Unsupported element type |
| Lowered by hip-ep | Successfully converted — not a blocker |

**Conv under shape blockers** means Conv is implemented but this model uses dynamic spatial dims. Only dynamic batch (N) is supported in `ConvConversion.cpp`.

## Other modes

- Re-parse MLIR dumps: `--parse-mlir-dir "<DUMP_DIR>"`
- Re-parse log: `--parse-log "<LOG.log>"`
- Machine-readable: add `--json-out "<path.json>"`
- Op names only: `--quiet`
- Full diagnostics: `--verbose`

## Troubleshooting

| Symptom | Action |
|---------|--------|
| `onnxruntime_perf_test not found` | Locate or ask user for gpu-test-package; or fall back to `--scan-onnx` |
| `requires the 'onnx' package` | `pip install onnx` |
| Timeout on large model | Increase `--timeout` |
| Windows copy/path errors | Quote paths with `-LiteralPath` or double quotes |
