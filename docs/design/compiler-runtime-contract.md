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
  │              model.dll               │
  │  __metadata_blob       (internal)    │  ← baked into DLL data section
  │  __metadata_json       (internal)    │  ← human-readable copy
  │                                      │
  │  inference_init(out_state, fs)       │  ← public API
  │  inference_get_metadata_json()       │  ← public API
  └──────────────────────────────────────┘
        │
        │  inference_init reads __metadata_blob directly
        │  (same LLVM module — hidden from public API)
        ▼
  Runtime
```

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
into the LLVM module with **internal linkage** (not visible outside `model.dll`):

| Global | Content | Linkage |
|--------|---------|---------|
| `__metadata_blob` | FlatBuffers binary bytes | Internal |
| `__metadata_json` | Human-readable JSON representation | Internal |

Both are compile-time constants baked directly into `model.dll`'s data section.
No file I/O is needed to access them at runtime.

The generated `inference_init` accesses `__metadata_blob` directly — they are
in the same LLVM module and linked into the same DLL. This is entirely hidden
from the public interface.

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

## Dynamic Output Shapes (DimSpec, dyn_dim_slots)

Some ONNX ops produce outputs whose shapes are only known at runtime
(e.g. `NonZero`, `Range` with non-static `start`/`limit`/`delta`,
`ConstantOfShape` with a non-constant shape input). The compiler partitions
each output dim into one of three categories and dispatches accordingly.

### Provenance categories

| Category | Definition | Dispatch |
|----------|------------|----------|
| **A** (static) | Dim is a compile-time integer literal | Folded into module metadata as a positive value |
| **B** (host-resolvable) | Dim is derivable at the EP boundary from func-arg shape metadata or from scalar host-side values (e.g. `Range` with int64 scalar inputs) | EP pre-computes the dim before `inference_compute`; output OrtValue is allocated with the right shape; runtime wrapper receives the buffer and fills it |
| **C** (runtime-published) | Dim is only known after a kernel runs (e.g. `NonZero` count, `Range` whose inputs trace to another GPU-resident tensor) | Runtime wrapper allocates from the dyn pool, **publishes** the dim size to a slot, and **publishes** the buffer pointer; EP post-`inference_compute` reads the slot and constructs the OrtValue |

Categories B and C share the same lowering machinery — the wrapper just sees
different operand provenance. The category is implicit in the lowering choice
(`wrap_X` vs `wrap_X_dyn`), so the runtime never has to reason about it.

### DimSpec tree

Each per-op output dim carries a `DimSpec` attribute — a small algebraic tree
describing how to compute that dim. Leaves:

| Leaf | Encoding | Meaning |
|------|----------|---------|
| `Static(v)` | `[1, v]` | Compile-time constant `v` |
| `InputDim(arg, dim)` | `[2, arg, dim, …]` | Dim `dim` of func-arg `arg` |
| `InputValueI64(arg)` | `[3, arg, …]` | Scalar i64 value of func-arg `arg` |
| `RuntimeSlot(slot_id)` | `[4, slot_id, …]` | Runtime-published value in slot `slot_id` |

Internal nodes encode arithmetic (`Add`, `Sub`, `Mul`, `Div`, `Max`, `Min`,
`SelectGT`, …) and saturate-to-zero clamping at the **root** only.
`ComposeDimSpecs` (`lib/Dialect/Transforms/ComposeDimSpecs.cpp`) walks SSA
producers backward to compose per-op specs into a single per-module-output
spec, materialised as the `hipdnn.output_dim_specs` module attribute. The EP
host-side resolver (`backend-mlir-compiler/custom-op-mlir/src/DimSpecResolver.cpp`)
evaluates the tree at every `Compute()` for both pre-marshal (to size the
OrtValue) and post-marshal (to publish results) purposes.

### Per-op DimSpec builder registry

Not every HIP op pre-attaches an `output_dim_specs` ArrayAttr at conversion
time — that would require every conversion to be aware of the dynamic-shape
plan, including ones whose shape semantics are trivial (rank-preserving
permutations, NumPy-style broadcast, identity-style passthroughs). To keep
those conversions simple, `shape_interface::getResultDimSpec`
(`lib/Dialect/IR/HipShapeInterface.cpp`) consults a per-op **builder
registry** keyed on the op's full name (e.g. `"hip.transpose"`).

Resolution order in `getResultDimSpec(op, result_index, dim_index)`:

1. **Explicit attribute** — if the op carries `output_dim_specs`, return
   the serialised entry. Highest precedence so an op that knows its shape
   (e.g. a Category-C producer like `hip.nonzero` that publishes a
   `RuntimeSlot`) wins unconditionally.
2. **Registered builder** — if the op's name has a builder registered,
   invoke it. Builders are pure functions of `(op, result_index,
   dim_index)`; they typically walk operands via `resolveDimFromValue` to
   compose a DimSpec from upstream producers. Returning empty signals "I
   cannot resolve this here" and falls through to (3).
3. **Static-type fallback** — if the result MLIR type has a known dim
   size, return `Static(dim_size)`. Otherwise return empty (caller
   handles the un-resolvable case, e.g. by `notifyMatchFailure` and
   waiting for upstream ops to convert further).

Builder registration is performed once from `HipDialect::initialize` via
`populateBuiltinDimSpecBuilders` (idempotent — guarded by `std::call_once`).
This guarantees builders are available before any pattern (including
`ShapeToHip`, the canonical consumer) ever runs.

Built-in builders today:

| Builder | Op names | Behaviour |
|---|---|---|
| `buildTransposeDimSpec` | `hip.transpose` | Rank-preserving permutation: output dim `d` ← `resolveDimFromValue(operand_1, perm[d])`. Returns empty if `perm` is missing/malformed (defensive — should never happen post-verifier). |
| `buildBroadcastDimSpec` | `hip.add/sub/mul/div/min/equal/less/and/mod/where/miopen.add/not/neg/cos/sin/sign/silu/sigmoid/softplus/gelu/reciprocal/sqrt/cast` | NumPy-style broadcast: walks every data operand `[1, N-1)` (skipping ctx and the last DPS init), right-aligns ranks against the result, and returns the first operand whose dim at this position resolves to a non-empty *and* non-broadcast-1 DimSpec. A static dim of 1 is treated as "broadcast me" — kept as a fallback only. |

**Adding a new shape-behavior op** is a one-line registration call inside
`populateBuiltinDimSpecBuilders`. Prefer re-using an existing builder when
the new op shares its shape semantics — `buildBroadcastDimSpec` correctly
handles **all** pure elementwise ops regardless of arity (unary,
two-operand, three-operand `where`, …) since it iterates the data operand
range generically. Write a new builder only when the shape transformation
genuinely differs (e.g. reductions, gathers, concats — all currently rely
on the explicit-attribute path).

**Coverage gap intentionally left open**: ops whose output shape is a
*non-trivial* transformation of their inputs (`hip.gather`, `hip.slice`,
`hip.scatter_nd`, `hip.reduce_*`, `hip.cumsum`, `hip.expand`, `hip.tile`,
…) still rely on explicit `output_dim_specs` attached by their respective
conversions. The registry is the framework facility to remove that
requirement op-by-op when the explicit-attribute path proves insufficient.

LIT coverage: `test/lit/Conversion/onnx-to-hip/test_shape_dyn_via_passthrough.mlir`
exercises both `Shape(hip.transpose(?))` and `Shape(hip.<elementwise>(?))`
end-to-end through the lowered IR (asserting on the materialised
`element_dim_specs` ArrayAttr in `hip.shape`).

### Runtime slot mechanics

`RuntimeState->dyn_dim_slots` is an `int64_t[dyn_dim_slots_count]` (size
known at compile time from `hipdnn.dyn_dim_slots_count` module attr) plus a
parallel `void *[dyn_dim_slots_count]` for buffer pointers. Helpers:

```c
void hipdnn_ep_state_publish_dim   (RuntimeState*, int32_t slot, int64_t v);
int64_t hipdnn_ep_state_read_dim   (RuntimeState*, int32_t slot);
void hipdnn_ep_state_publish_buffer(RuntimeState*, int32_t slot, void* gpu_ptr);
void *hipdnn_ep_state_read_buffer  (RuntimeState*, int32_t slot);
void *hipdnn_ep_state_dyn_pool_alloc(RuntimeState*, int64_t bytes);
```

The `dyn_pool` is a growable contiguous GPU buffer managed by
`hipdnn_ep_state_dyn_pool_alloc` (bump-pointer within current segment;
`hipMalloc` a new segment on overflow). Reset across each `Compute()` via the
`begin_compute()` runtime hook (also invalidates the GQA seqlens_k cache).

### `hipdnn_ep_get_pool_base(state, required_size)` contract

The lowering of `hip.get_pool(%ctx, %size_index)` emits

```mlir
llvm.call @hipdnn_ep_get_pool_base(%state, %size) : (!llvm.ptr, i64) -> !llvm.ptr
```

This is a separate, **growable-pool** API from `dyn_pool_alloc` and is used
by the post-bufferize `hip-pool-allocs` pass to back ALL `memref.alloc`s
(including single-alloc functions) with a view into the pool. Contract:

- `required_size = 0` returns the current base (or `nullptr` if uninitialised).
- `required_size > state->pool_size` triggers `hipFree(old) + hipMalloc(new)`.
- The base pointer can change across calls — long-lived consumers must
  re-query on every entry into the function. The generated code already
  follows this contract (no SSA value of the base survives across
  `inference_compute` boundaries).
- Failure to allocate (rare) leaves `state->pool_base = nullptr` and
  `state->pool_size = 0`, returns `nullptr`.

The legacy 1-arg `hipdnn_ep_get_pool_base(state)` is **no longer supported**.
This signature is wired through `lib/Conversion/HipToLLVM/MemoryLowering.cpp::GetPoolOpLowering`
to match what the codegen produces.

### Pipeline ordering invariants (added)

Two passes were added to `buildHipToLLVMPipeline` to support the dyn-shape
codegen:

1. `createLowerAffinePass()` between `ExpandStridedMetadata` and
   `ConvertHipToLLVM` (lowers `affine.apply` that survives strided-metadata
   expansion of multi-dim collapse_shape).
2. `arith::createArithExpandOpsPass()` after `LowerAffinePass` and before
   `ConvertHipToLLVM` (expands `arith.ceildivsi` / `arith.floordivsi` /
   `arith.ceildivui` which `populateArithToLLVMConversionPatterns` does NOT
   handle natively — needed by `wrap_range`-style output-length math).

Both passes are silent in production when omitted: compilation fails, the EP
falls back to CPU, and any CPU-vs-CPU comparison passes. The only reliable
detection is checking wall-clock time and `HIPDNN_EP_DEBUG=1` output for
`[REAL] wrap_*` lines confirming GPU dispatch.

### Slot-buffer coalesce (Phases 3 + 4)

Two compile-time passes (see `docs/design/slot-buffer-coalesce.md` for the
full per-class taxonomy) cut dynamic-shape footprint and kernel-launch count
without changing the runtime ABI:

- `hip-identity-propagator-rebind` (PRE-bufferize, in the OnnxToHip tail)
  erases ops whose registered identity predicate
  (`shape_interface::isIdentityOp`) is true. Built-in predicates cover
  `hip.transpose` with `perm = [0..rank)`, `hip.cast` with matching dtype,
  `hip.expand` / `hip.tile` / `hip.slice` / `hip.reduce_*` with matching
  shape. After erasure, the pass remaps the erased op's published slot id
  (if any) to the upstream input's slot id across publishers, consumers,
  `RuntimeSlot` leaves, and module-level metadata.
- `hip-slot-lifetime-coalesce` (between `hip-annotate-input-dim-slots` and
  `convert-hip-to-llvm`) groups published slot ids by canonical DimSpec
  bytes (`DimSpec::canonicalize()` constant-folds + sorts commutative
  children) and first-fit-decreasing bin-packs by lifetime within each
  group. The smallest slot id in each bin becomes the representative; every
  other id in the bin is rewritten throughout the module. Surviving slot
  ids are renumbered to a contiguous `0..K-1` range and
  `hipdnn.dyn_dim_slots_count` is updated so `GenerateInterface` emits the
  reduced count into the FlatBuffers metadata.

Runtime ABI additions for Phase 2 publisher / Phase 3 reuse:

```c
void *hipdnn_ep_state_dyn_pool_alloc_for_slot(RuntimeState*, int64_t bytes, int32_t slot);
void *hipdnn_ep_state_publish_buffer_resize  (RuntimeState*, int32_t slot, int64_t bytes);
```

`dyn_pool_alloc_for_slot` is the publisher fast path: allocates a fresh
exact-size buffer and publishes it to the given slot.
`publish_buffer_resize` is the Phase 3 coalesced-slot helper: returns the
existing buffer if the prior allocation in the same `Compute()` already met
the byte requirement; otherwise allocates a fresh segment. Both functions
range-check `slot_id` against `state->dyn_dim_slots_count` and `LOG(FATAL)`
on out-of-range.

Output-lifetime invariant: slot ids referenced from module-level
`hipdnn.output_dim_specs` live past `inference_compute` return. The EP-side
resolver reads them after stream sync to populate output OrtValue shapes.
Two output-bound slots cannot share a dyn-pool buffer. The coalescer models
this by assigning `lastUseIdx = +∞` to every output-bound slot;
`VerifyDimSpecsPass` enforces the invariant.

Disabled paths for A/B measurement:
`--hip-identity-propagator-rebind=disable-identity-rebind=true` and
`--hip-slot-lifetime-coalesce=disable-coalesce=true`. Combine with
`--hip-print-pool-stats` (the diagnostic pass added in
`lib/Dialect/Transforms/PrintPoolStats.cpp`) for before/after footprint and
kernel-launch comparison.

---

## Related Documents

- [constant-handling-design.md](constant-handling-design.md) — how `constants_filename`, `sizes`, and `offsets` fields are produced and consumed
- [morphizen-ep-integration.md](morphizen-ep-integration.md) — DLL contracts; `inference_init` public interface
- [compilation-options.md](compilation-options.md) — `constants_file` compilation option
- [dynamic-shape-debug-surface.md](dynamic-shape-debug-surface.md) — env-var gates for tracing DimSpec resolution and slot publish/read; deferred-validator rationale
- [slot-buffer-coalesce.md](slot-buffer-coalesce.md) — full op-class taxonomy, per-class invariants, and pass design notes for `hip-identity-propagator-rebind` and `hip-slot-lifetime-coalesce`
