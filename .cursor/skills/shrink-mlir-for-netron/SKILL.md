<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
---
name: shrink-mlir-for-netron
description: >-
  Shrinks huge MorphiZen/ONNX MLIR for Netron (elide dense/quant payloads, Cast
  quoting, ui types, optional constant strip). Use when model.mlir is too large
  for Netron, shrink mlir for netron, or /shrink-mlir-for-netron.
disable-model-invocation: true
---

# Shrink MLIR for Netron

**View-only** pipeline beside the source `.mlir` (e.g. from `/onnx-model-to-onnx-mlir`):

| Output | Purpose |
|--------|---------|
| **`<stem>.netron.mlir`** (e.g. **`model.netron.mlir`**) | Shrunk + Netron fixes — open this in Netron |

Do not open the multi-GB original. No other files are written next to the input.

**Project skill:** `~/workspace/.cursor/skills/shrink-mlir-for-netron/`

**Slash command:** `/shrink-mlir-for-netron`

## Agent workflow

1. Path to huge `.mlir`.
2. **Run the combined script** (streaming; do not load file into chat):
   - PowerShell:
     `& "$env:USERPROFILE\workspace\.cursor\skills\shrink-mlir-for-netron\scripts\shrink_mlir_for_netron.ps1" <path>`
   - gpu-bash:
     `bash "$HOME/workspace/.cursor/skills/shrink-mlir-for-netron/scripts/shrink_mlir_for_netron.sh" <path>`
3. Report stats from the script; final path **`<stem>.netron.mlir`**.
4. Note: **not compilable MLIR**; stripping constants can leave **invalid SSA** (OK for Netron-only viewing).

**Keep weight nodes in the graph:** pass **`--keep-constants`** (bash) or **`-KeepConstants`** (PowerShell).

Requires **Python 3** on PATH.

## Related

- **onnx-model-to-onnx-mlir** → `model.mlir`
