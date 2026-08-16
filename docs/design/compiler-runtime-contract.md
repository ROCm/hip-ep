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
- [Generated-Artifact ABI Handshake](#generated-artifact-abi-handshake)
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
  │  inference_get_artifact_abi()        │  ← public pre-bind handshake
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
That EPContext metadata also carries an early copy of the tagged artifact ABI
token; the generated artifact export remains the final authority.

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

## Generated-Artifact ABI Handshake

`include/hip/artifact_abi.h` owns the single
`hipdnn::abi::kArtifactAbiVersion` constant and the tagged token derived from
it. `generate-interface` emits:

```c
uint64_t inference_get_artifact_abi(void);
```

The loader must validate this stable no-argument query before resolving or
calling `inference_init`, `inference_compute`, or any generated/runtime wrapper
entry point. Missing exports, a malformed tag, and unequal versions all fail
closed. LLVM bitcode is also inspected structurally before its model/runtime
modules are added to the JIT; native artifacts expose the same exported query.

EPContext metadata stores the same tagged token so stale contexts can be
rejected before the artifact is opened. This is an early consistency check, not
a second version authority: both copies are emitted from
`kArtifactAbiVersion`, and the artifact export is always validated after load.
Old contexts and artifacts without the handshake are rejected.

Bump `kArtifactAbiVersion` whenever:

- an `inference_*` signature changes;
- a generated call to a runtime wrapper changes name, signature, or semantics;
- generated code observes a changed `RuntimeState` or other runtime layout.

Do not bump it for additive inspection metadata or implementation-only changes
behind unchanged wrapper and layout contracts. Cache-key salting remains useful
for normal invalidation, but it is not the ABI safety mechanism.

---

## Current Fields

| Field | Type | Purpose |
|-------|------|---------|
| `version` | `int32` | FlatBuffer schema version; not the generated/runtime ABI version |
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
invalidation requirement. A generated-call ABI change requires an artifact ABI
version bump; the loader then rejects cached artifacts compiled against the old
pool or host-scratch signatures before inference setup.

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

Standalone Softmax generated calls carry both input and output row/column
extents plus a semantic `HIPDNN_EP_DATATYPE_*` value. The runtime validates
dynamic descriptor equality before copying or dispatching f16, bf16, or f32
storage through the typed custom kernel. This contract supersedes the former
element-byte-width argument and therefore requires artifact invalidation.

Payload-dependent Expand lowering passes a checked `shape_valid` argument to
`wrap_expand_checked`. The wrapper returns failure before dispatch when grouped
readback, broadcast compatibility, descriptor agreement, or element-count
validation fails. This argument is part of the versioned artifact ABI.

HIP MatMul and Gemm carry two contraction extents in generated calls. MatMul
calls `wrap_hipblasLtMatmul` with `A[-1]`, `B[-2]`, and one i1 proving
all right-aligned runtime batch axes and output extents agree with ONNX
broadcast. Gemm passes the transpose-aware K extent from each operand. The real
and mock wrappers validate these contracts before descriptor/cache creation or
dispatch, then use the equal K value in cache keys. Runtime-invalid dimensions
and checked output-size overflow set the shared error flag and skip BLAS work;
a known, valid nonempty output allocation is zero-filled on failure.

Artifacts built against the former 11-argument MatMul signature or the old
one-K Gemm signature must be recompiled.

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
