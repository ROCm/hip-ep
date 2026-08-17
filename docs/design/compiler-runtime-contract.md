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
  `hipdnn_ep_get_pool_base(state, site_id, domain_id, needed_size)`.
- `site_id` is the deterministic top-level function ordinal and `domain_id` is
  local to that function. The explicit pair prevents caller/outlined-helper
  pool collisions without relying on symbol-name hashes.
- Each `hip.get_host_scratch` lowers to
  `hipdnn_ep_get_host_scratch_base(state, site_id, needed_size)`. Distinct
  function sites own distinct host-mapped allocations; production outlined
  helpers are non-recursive and execute under the single-inference-per-state
  contract.
- None of the `hipdnn.pool_*` or `hipdnn.buffer_*` attributes are fields in the
  FlatBuffers schema or the JSON mirror.

The generated LLVM IR plus `RuntimeState` therefore carry pool behavior. See
[pool-allocs-memory-planning.md](pool-allocs-memory-planning.md) for the
attribute, lowering, grow-on-demand runtime contract, and stale-artifact
invalidation requirement. Because the generated call ABI changed, cached model
artifacts compiled against the old pool helper or two-argument host-scratch
helper must be deleted and rebuilt with the matching runtime bitcode.

Input staging is another generated-code-only contract. `generate-interface`
queries `hipdnn_ep_tensor_buffer_storage_words` and constructs every opaque
`TensorBuffer` before the first fallible prepare call. It commits a prepared
count after each success, so shared error cleanup releases exactly that prefix
through one runtime batch helper. The runtime, not generated LLVM, owns the
record layout and release loop.

Operation runtime calls use one status contract. After HIP-to-LLVM conversion,
every status-bearing call records its nonzero `i32` in the shared runtime error
flag. The generated interface reads and clears that flag after stream
synchronization and returns it through `inference_compute`. Conversion fails
closed if any `i32` call result is left unused; scalar data-return calls must
consume their value explicitly.

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
