<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: onnx-model-to-onnx-mlir
description: >-
  Generates hip-compiler input MLIR from an ONNX model path via hip-onnx-runner,
  writing <stem>.mlir next to the .onnx file. Use for onnx-model-to-onnx-mlir, onnx to mlir, morphizen
  mlir dump, hip-compiler input, or /onnx-model-to-onnx-mlir.
---

# ONNX model to ONNX MLIR

Writes **`<model_stem>.mlir`** in the **same directory** as the source `.onnx` (hip-compiler input from MorphiZen). Scripts start `hip-onnx-runner`, wait until **`mlir_bytecode_dump.mlir`** size is stable, then **kill the runner** so post-dump hip-compiler / session init is skipped.

**Project location:** `.cursor/skills/onnx-model-to-onnx-mlir/` (repo-relative paths below).

**Slash command:** `/onnx-model-to-onnx-mlir` via `.cursor/commands/onnx-model-to-onnx-mlir.md`.

## Agent workflow

1. Resolve the ONNX path (absolute or from workspace).
2. Run **`scripts/onnx-model-to-onnx-mlir.sh`** or **`scripts/onnx-model-to-onnx-mlir.ps1`** with that path only (output dir defaults to the ONNX parent folder).
3. Success = **`<onnx_dir>/<stem>.mlir`** exists (renamed from `mlir_bytecode_dump.mlir`).
4. Rebuild runner if `--dump-compiler-mlir` is missing:
   `cmake --build $BUILD_DIR --config RelWithDebInfo --target hip-onnx-runner -j 8`

## Quick run

**Git Bash / gpu-bash:**

```bash
bash ".cursor/skills/onnx-model-to-onnx-mlir/scripts/onnx-model-to-onnx-mlir.sh" <path-to-model.onnx>
```

**PowerShell:**

```powershell
& ".cursor/skills/onnx-model-to-onnx-mlir/scripts/onnx-model-to-onnx-mlir.ps1" <path-to-model.onnx>
```

Optional second argument overrides the output directory (default: ONNX file directory).

## Paths (hip-ep)

| Item | Default |
|------|---------|
| `BUILD_DIR` | `$env:BUILD_DIR` or `$env:USERPROFILE\workspace\build\hip-ep` (PowerShell); `$BUILD_DIR` or `$HOME/workspace/build/hip-ep` (Git Bash) |
| `hip-onnx-runner` | `$BUILD_DIR/bin/RelWithDebInfo/hip-onnx-runner.exe` (or `Release/` per `--config`) |
| **Output MLIR** | `<dir-of-onnx>/<stem>.mlir` |

Run from the **hip-ep repo root** (Git Bash launched from an x64 Native Tools prompt; see [docs/quick_start.md](../../docs/quick_start.md)).

## Notes

- Runner flag `--dump-compiler-mlir` sets MorphiZen env before `hipgpu.dll` loads.
- MorphiZen may also write `morphizen.*.mlir` debug snapshots in the same directory during the run.
- Tune watch/kill: `ONNX_TO_MLIR_POLL_SEC` (default 2), `ONNX_TO_MLIR_STABLE_POLLS` (default 3), `ONNX_TO_MLIR_TIMEOUT_SEC` (default 7200).
- See [reference.md](reference.md) for pipeline details.
