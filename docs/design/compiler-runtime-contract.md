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
- [Generated-Code Runtime Inputs](#generated-code-runtime-inputs)
- [Consumers](#consumers)
- [Design Principles](#design-principles)
- [Related Documents](#related-documents)

---

## Overview

`schemas/model_metadata.fbs` is the **single source of truth for serialized
model metadata** shared with artifact loaders and external consumers. It does
not contain every compile-time value passed to runtime functions by generated
LLVM IR. Transient memory planning is the canonical generated-code-only
contract; see [Generated-Code Runtime Inputs](#generated-code-runtime-inputs).

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

## Generated-Code Runtime Inputs

Some compiler/runtime coupling is encoded directly in generated LLVM calls
rather than serialized in `model_metadata.fbs`. These contracts are internal to
the compiled model and are documented by the design that owns the lowering.

Pool planning is the canonical example:

- `hip-pool-allocs` writes `hipdnn.pool_size`,
  `hipdnn.buffer_count`, `hipdnn.buffer_offsets`, and optional
  multi-domain attributes on the MLIR module.
- `generate-interface` consumes the legacy attributes at compile time and emits
  `hipdnn_ep_pool_init` only when all three are present and
  `hipdnn.pool_size > 0`.
- Each `hip.get_pool` lowers directly to
  `hipdnn_ep_get_pool_base(state, domain_id, needed_size)`.
- None of the `hipdnn.pool_*` or `hipdnn.buffer_*` attributes are fields in the
  FlatBuffers schema or the JSON mirror.

The generated LLVM IR plus `RuntimeState` therefore carry pool behavior. See
[pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) for the
attribute, lowering, and grow-on-demand runtime contract.

GatherND has another generated-code-only contract. `wrap_gather_nd` receives
the data, indices, and output pointers plus their host-side i64 shape arrays,
ranks, `batch_dims`, and the data element type. It carries no indices element
width. Both the wrapper and custom kernel interpret `indices` as an
`int64_t *`, so ONNX conversion, HIP verification and reification, and
HIP-to-LLVM lowering all require i64 indices before emitting destination-shape
IR or a runtime call. Supporting i32 indices would require an explicit runtime
ABI extension rather than reusing this call.


---

## Consumers

| Consumer | Fields used | How |
|----------|-------------|-----|
| Runtime (`inference_init`) | `constants_filename`, `constants[i].offset` | Reads `__metadata_blob`; opens constants file via `fs`; indexes into blob by offset |
| Any caller | all | Reads `__metadata_json` via `inference_get_metadata_json()` |

---

## Design Principles

**Durable serialized metadata goes through this schema.** Add a schema field
when information must be inspected by the loader or external consumers, or
must remain available independently of generated code. Compile-time values
used only to form calls inside the generated model may remain an internal
lowering contract, but must be documented by the owning design and must not be
described as part of `__metadata_blob`.

---

## Related Documents

- [constant-handling-design.md](constant-handling-design.md) — how `constants_filename`, `sizes`, and `offsets` fields are produced and consumed
- [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) — generated-code-only pool attributes and runtime calls
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — JIT loader contract; `inference_init` public interface
- [compilation-options.md](compilation-options.md) — `constants_file` compilation option
