<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# ONNX model to ONNX MLIR (hip-compiler input)

Follow the **onnx-model-to-onnx-mlir** project skill (`.cursor/skills/onnx-model-to-onnx-mlir/SKILL.md` in this workspace).

1. Use the path to the `.onnx` model from my message; if missing, ask for it.
2. **Run the dump script yourself** (do not only print env vars):
   - gpu-bash: `bash "$HOME/workspace/.cursor/skills/onnx-model-to-onnx-mlir/scripts/onnx-model-to-onnx-mlir.sh" <path-to-model.onnx>`
   - PowerShell: `& "$env:USERPROFILE\workspace\.cursor\skills\onnx-model-to-onnx-mlir\scripts\onnx-model-to-onnx-mlir.ps1" <path-to-model.onnx>`
3. **Output location:** same directory as the ONNX file, as `<model_stem>.mlir` (e.g. `model.onnx` → `model.mlir`).
4. Treat success as the `.mlir` file exists; the script stops the runner after a stable pre-compiler dump (no full session/compile wait).
5. Report full path, size, and note that the runner was stopped early (not a full inference session).
