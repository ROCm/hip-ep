<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# hip-add-context-arg Pass

**Date:** 2026-03-03
**Document Type:** Design
**Status:** Draft
**Related:** [../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md), [02-OnnxToHip.md](02-OnnxToHip.md), [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md)

**Input:** ONNX-MLIR module
**Output:** ONNX-MLIR module with `!hip.context` prepended to all functions

---

## Overview

`hip-add-context-arg` prepends `%ctx: !hip.context` as argument 0 to every
`func.func` in the module. `%ctx` is a threading argument — it carries GPU
runtime state (stream, library handles, constant pointers) through every HIP
operation without global variables. See
[RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) for the `RuntimeState`
internals.

This pass runs before `convert-onnx-to-hip` so that patterns in that pass can
reference `%ctx` when emitting `hip.get_constant` and HIP compute ops.

---

## Why a Threading Argument?

HIP operations need access to GPU runtime state: the HIP stream, MIOpen and
hipBLAS handles, and the constant pointer array. The alternatives are:

- **Global variables** — not thread-safe; multiple concurrent inference sessions
  would race.
- **Re-acquire from OS/driver per op** — high overhead on the hot inference path.
- **Threading argument** — zero overhead once passed; one state per session,
  fully isolated.

`%ctx` is the threading argument. It flows from the function entry point through
every HIP op as an explicit operand, making data flow visible in the IR and
enabling LLVM to optimize accessor calls to direct loads after IR merging with
the runtime bitcode. See [RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md)
for the zero-cost abstraction mechanism.

---

## Input / Output Format

**Before:**
```mlir
func.func @main_graph(%arg0: tensor<1x3x224x224xf32>)
                      -> tensor<1x64x224x224xf32> {
  ...
}
```

**After:**
```mlir
func.func @main_graph(%ctx: !hip.context,
                      %arg0: tensor<1x3x224x224xf32>)
                      -> tensor<1x64x224x224xf32> {
  ...
}
```

The function type is updated accordingly. All call sites are updated if present.

---

## Related Documents

- [02-OnnxToHip.md](02-OnnxToHip.md) — Downstream pass; uses `%ctx` in every
  emitted HIP op
- [../HIP-DIALECT-DESIGN.md](../HIP-DIALECT-DESIGN.md) — `!hip.context` type
  definition
- [../../RUNTIME-ARCHITECTURE.md](../../RUNTIME-ARCHITECTURE.md) — `RuntimeState`
  internals and zero-cost abstraction via IR merging
- [../LOWERING-PIPELINE.md](../LOWERING-PIPELINE.md) — Pipeline stage 1
