<!--
Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Compilation Options

**Date:** 2026-03-03
**Document Type:** Design
**Status:** Draft
**Related:** [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md), [MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md)

**Source:** [`proto/compilation_options.fbs`](../../proto/compilation_options.fbs)

---

## Overview

Compilation options are defined as a [FlatBuffers](https://flatbuffers.dev/) schema in
`proto/compilation_options.fbs` and passed to the compiler as a JSON string via the
`options_json` parameter of `udna_compile_with_fs` (see `include/udna-compiler/compiler_api.h`).

Callers may pass `NULL` for `options_json` to accept all defaults.

---

## Options

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `opt_level` | `int` | `2` | LLVM optimization level (0–3) |
| `output_mode` | `OutputMode` | `DLL` | Output format: `DLL` or `LLVM_IR` |
| `verbose` | `bool` | `false` | Enable verbose diagnostic output |
| `constants_file` | `string` | `"constants.bin"` | Path (relative to the `FileSystem` root) where constant weight data is written during compilation. See [CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md). |

### `output_mode` values

| Value | Meaning |
|-------|---------|
| `DLL` | Produce a shared library (`.dll` / `.so`) |
| `LLVM_IR` | Emit LLVM IR text (`.ll`) for inspection |

---

## Example

```json
{
  "opt_level": 0,
  "output_mode": "LLVM_IR",
  "verbose": true,
  "constants_file": "my_model/weights.bin"
}
```

---

## Related Documents

- **[CONSTANT-HANDLING-DESIGN.md](CONSTANT-HANDLING-DESIGN.md)** — How `constants_file` is used during constant extraction and runtime upload
- **[MLIR-COMPILATION-OVERVIEW.md](MLIR-COMPILATION-OVERVIEW.md)** — End-to-end compilation pipeline
- **[mlir/passes/02-OnnxToHip.md](mlir/passes/02-OnnxToHip.md)** — Pass that reads `constants_file` to write weight data
