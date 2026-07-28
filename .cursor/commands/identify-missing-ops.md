<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Identify missing ops (hip-ep ONNX compatibility)

Follow the **identify-missing-ops** project skill (`.cursor/skills/identify-missing-ops/SKILL.md` in this workspace).

1. Use the path to the `.onnx` model (or directory) from my message; if missing, ask for it. Resolve paths relative to the workspace root; quote paths with spaces or parentheses.
2. **Run the checker yourself** (do not only print instructions):
   - Locate **gpu-test-package** (`bin/onnxruntime_perf_test` or `.exe`) via `$env:PACKAGE_DIR` / `$PACKAGE_DIR`, or search the workspace for `gpu-test-package*`.
   - **Full compile check** (preferred when package exists):
     ```powershell
     python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py "<MODEL>" `
       --package-dir "<PACKAGE_DIR>" `
       --dump-mlir `
       --log-dir "<LOG_DIR>" `
       --show-next-step
     ```
   - **Static scan only** (no gpu-test-package):
     ```powershell
     python .cursor/skills/identify-missing-ops/scripts/identify_missing_ops.py --scan-onnx "<MODEL>" --show-next-step
     ```
3. Do not pass `--allow-cpu-fallback` unless I explicitly request it.
4. Summarize: hip-ep result (ok / compile failed / unsupported ops), failure stage/reason, blockers by category (compile, shape, variant, dtype, partial, unsupported), next-step hints, and MLIR dump path if used.
