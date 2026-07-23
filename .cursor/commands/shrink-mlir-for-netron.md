<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Shrink MLIR for Netron

Follow the **shrink-mlir-for-netron** project skill (`.cursor/skills/shrink-mlir-for-netron/SKILL.md`).

1. Use the path to the huge `.mlir` from my message; if missing, ask for it.
2. **Run the combined pipeline yourself**:
   - gpu-bash: `bash "$HOME/workspace/.cursor/skills/shrink-mlir-for-netron/scripts/shrink_mlir_for_netron.sh" <path.mlir>`
   - PowerShell: `& "$env:USERPROFILE\workspace\.cursor\skills\shrink-mlir-for-netron\scripts\shrink_mlir_for_netron.ps1" <path.mlir>`
3. **Output:** **`<stem>.netron.mlir`** only (e.g. `model.netron.mlir` — open this in Netron).
4. Report sizes/stats; output is **view-only**, not compilable MLIR. Constant stripping may break SSA (expected for Netron).
5. Optional: `-KeepConstants` / `--keep-constants` to keep `onnx.Constant` nodes in the graph.
