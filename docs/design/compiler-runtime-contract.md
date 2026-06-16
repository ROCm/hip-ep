<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Compiler–Runtime Contract

**Date:** 2026-03-05
**Document Type:** Design
**Status:** Draft
**Related:** [constant-handling-design.md](constant-handling-design.md), [morphizen-ep-integration.md](morphizen-ep-integration.md)

---

## Table of Contents

- [Overview](#overview)
- [Schema: Single Source of Truth](#schema-single-source-of-truth)
- [Embedding Mechanism](#embedding-mechanism)
- [Current Fields](#current-fields)
- [Consumers](#consumers)
- [Design Principles](#design-principles)
- [Related Documents](#related-documents)

---

## Overview

`schemas/model_metadata.fbs` is the **single source of truth** for the contract
between the compiler and the runtime. Every piece of information the compiler
needs to communicate to the runtime at code-generation time is expressed as a
field in this schema.

```
  Compiler (GenerateInterface pass)
        │
        │  serializes model_metadata into:
        │    __metadata_blob  (binary, for runtime)
        │    __metadata_json  (text, for inspection)
        ▼
  ┌──────────────────────────────────────┐
  │           model.bc (bitcode)         │
  │  __metadata_blob       (internal)    │  ← baked into bitcode data
  │  __metadata_json       (internal)    │  ← human-readable copy
  │                                      │
  │  inference_init(out_state, fs)       │  ← public API
  │  inference_get_metadata_json()       │  ← public API
  └──────────────────────────────────────┘
        │
        │  LlvmIrJit parses bytes → llvm::orc::LLJIT codegen
        │  inference_init reads __metadata_blob directly
        │  (same LLVM module — hidden from public API)
        ▼
  Runtime (EP DLL hosts the JIT)
```

This contract is **format-independent**. The same metadata blob, the same
`inference_*` / `hipdnn_ep_runtime_begin_compute` symbols, and the same JSON
mirror are produced for both artifact formats
(`CompilationOptions.output_mode`):

- **LLVM IR** (default): the symbols are JIT-resolved by `LlvmIrJit` in-process.
- **Native**: the same symbols are exported from the per-model `.dll`/`.so` and
  resolved via `morphizen::Plugin` (`GetProcAddress` / `dlsym`). The runtime is
  merged into the module at producer time so `hipdnn_ep_runtime_begin_compute`
  is present in the native artifact too.

The EP selects the loader from `mlir_metadata::Metadata.artifact_format` before
opening the artifact. See [native-vs-ir-comparison.md](../native-vs-ir-comparison.md).

---

## Schema: Single Source of Truth

`schemas/model_metadata.fbs` defines the schema compiled by FlatBuffers into
`__metadata_blob`. The compiler writes the blob from MLIR module attributes;
the runtime reads it directly.

```
table HipModelMetaInfo {
  version:            int32 = 1;
  constants_filename: string;
  constants:          [ConstantInfo];
  input_count:        int64;
  output_count:       int64;
  inputs:             [TensorInfo];
  outputs:            [TensorInfo];
  // ...
}
```

See [Current Fields](#current-fields) for the purpose of each field.

---

## Embedding Mechanism

The `GenerateInterface` pass (`lib/Dialect/Transforms/GenerateInterface.cpp`)
builds the FlatBuffers blob from MLIR module attributes and emits two globals
into the LLVM module with **internal linkage** (not visible to JITted callers
outside `model.bc`):

| Global | Content | Linkage |
|--------|---------|---------|
| `__metadata_blob` | FlatBuffers binary bytes | Internal |
| `__metadata_json` | Human-readable JSON representation | Internal |

Both are compile-time constants baked directly into the bitcode's data section
and end up in JITted memory after ORC codegen. No file I/O is needed to access
them at runtime.

The generated `inference_init` accesses `__metadata_blob` directly — they are
in the same LLVM module and JITted together. This is entirely hidden from the
public interface.

### inference_get_metadata_json

`inference_get_metadata_json()` is a public export that returns a pointer to
`__metadata_json`. The caller must not free the pointer. Safe to call from
multiple threads.

```c
const char* inference_get_metadata_json(void);
```

The JSON mirrors the FlatBuffers schema. `constant_sizes` is included for
external consumers; `constant_offsets` is omitted (internal to the runtime).
Dynamic dimensions are encoded as `-1`.

```json
{
  "version": 1,
  "constants_filename": "constants.bin",
  "constant_sizes": [int64, ...],
  "input_count": int64,
  "output_count": int64,
  "inputs":  [ { "shape": [int64, ...], "element_size": int64 }, ... ],
  "outputs": [ { "shape": [int64, ...], "element_size": int64 }, ... ]
}
```

---

## Current Fields

| Field | Type | Purpose |
|-------|------|---------|
| `version` | `int32` | Schema version |
| `constants_filename` | `string` | Name of the constants file to open via `fs` |
| `constants[i].size` | `int64` | Byte size of constant `i` |
| `constants[i].offset` | `int64` | Byte offset of constant `i` in the constants file |
| `input_count` | `int64` | Number of model inputs |
| `output_count` | `int64` | Number of model outputs |
| `inputs[i].shape` | `[int64]` | Tensor dimensions of input `i`; `-1` = dynamic |
| `inputs[i].element_size` | `int64` | Bytes per element of input `i` |
| `outputs[i].shape` | `[int64]` | Tensor dimensions of output `i`; `-1` = dynamic |
| `outputs[i].element_size` | `int64` | Bytes per element of output `i` |

---

## Consumers

| Consumer | Fields used | How |
|----------|-------------|-----|
| Runtime (`inference_init`) | `constants_filename`, `constants[i].offset` | Reads `__metadata_blob`; opens constants file via `fs`; indexes into blob by offset |
| Any caller | all | Reads `__metadata_json` via `inference_get_metadata_json()` |

---

## Design Principles

**All compiler–runtime coupling goes through this schema.** If the compiler
needs to pass information to the runtime, it adds a field here — not through
a separate file, environment variable, or hardcoded constant.

---

## Related Documents

- [constant-handling-design.md](constant-handling-design.md) — how `constants_filename`, `sizes`, and `offsets` fields are produced and consumed
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — JIT loader contract; `inference_init` public interface
- [compilation-options.md](compilation-options.md) — `constants_file` compilation option
