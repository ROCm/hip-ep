<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# Slot-Buffer Coalescing — Op-Class Taxonomy and Chain Trace

**Date:** 2026-05-25
**Document Type:** Design
**Status:** Phase 0 fast-trace deliverable
**Related:** [compiler-runtime-contract.md](compiler-runtime-contract.md), [dynamic-shape-debug-surface.md](dynamic-shape-debug-surface.md)

---

## Table of Contents

- [Overview](#overview)
- [Op-Class Taxonomy](#op-class-taxonomy)
- [Per-Model Chain Trace](#per-model-chain-trace)
- [Pattern Coverage Matrix](#pattern-coverage-matrix)
- [Phase-by-Phase Coverage Decisions](#phase-by-phase-coverage-decisions)
- [Implementation status (as of 2026-05-25)](#implementation-status-as-of-2026-05-25)
- [Per-class invariants](#per-class-invariants)

---

## Overview

This document is the Phase 0 deliverable of the **Slot-Buffer Coalescing** initiative
(see the plan file at `c:\Users\Administrator\.cursor\plans\slot-buffer-coalesce_17ec4e47.plan.md`).
Its single purpose is to enumerate every `Hip_DpsOp` in the dialect and every
dynamic-shape chain in the three Qwen3.5-VL-35B-A3B reference models so that
Phases 1–4 have a written coverage contract.

Scope is **fast-trace** (per user decision):

- **Full chain trace** for `embedding.onnx` (canonical regression artifact)
  and `text.onnx` (MoE-rich Cat-C surface).
- **Sampled chain trace** for `vision.onnx` (3-5 representative Cat-C chains
  in the main graph; vision is otherwise covered by SYNTHETIC tests since
  end-to-end vision is structurally blocked by 27× `Loop`).

Acceptance criterion: for every chain documented below, EITHER it maps to a
Phase 1/2/3/4 work-item OR it is listed in the [out-of-scope blockers](#out-of-scope-blockers)
section with rationale.

---

## Op-Class Taxonomy

### Six op classes

| Class | Meaning | Slot-machinery role |
| --- | --- | --- |
| **Publisher** | Cat-C op that reserves a `slot_id` and calls `publish_dim` / `publish_buffer` at runtime | Currently: `hip.nonzero`, `hip.range` Cat-C path, `hip.constant_of_shape` Cat-C path |
| **Single-input propagator** | Rank-preserving (`hip.transpose`) or rank-preserving elementwise (every unary op) | Inherits upstream slot via `findDpsWriter`; needs `output_slot_ids` to publish its own exact-size buffer (Phase 2) |
| **Broadcast propagator** | NumPy-style elementwise binary (`hip.add`, `hip.sub`, ... `hip.where`) | Same as single-input propagator |
| **Aggregating propagator** | Output dim is the SUM of operand dims (`Concat` — no converter today) | Needs `buildConcatDimSpec` returning `Add(InputDim(0,axis), ..., InputDim(N-1,axis))` |
| **Reducing propagator** | Drops or collapses dims (`hip.reduce_sum`, `hip.reduce_max`, `hip.reduce_prod`, `hip.cumsum`) | Needs `buildReduceDimSpec`; `keepdims=0` skips reduced dim, `keepdims=1` becomes `Static(1)` |
| **View propagator** | Zero-cost MLIR views (`tensor.expand_shape`, `tensor.collapse_shape`, `tensor.extract_slice`, `memref.subview`) produced by Reshape / Squeeze / Unsqueeze / Split lowerings | NO slot plumbing — inherits storage from upstream; verified pass-through |
| **Shape-data op** | Value-producing ops where elements encode dims (`hip.shape`, `hip.size`) | Needs `buildShapeAsValueResolver` so downstream Range / ConstantOfShape can lift the value into a leaf form |

### Hip_DpsOp inventory (from `include/hip/Dialect/IR/HipOps.td`)

| Op | Class | DimSpec builder | Phase 2 retarget | Phase 4 identity predicate |
| --- | --- | --- | --- | --- |
| `hip.conv` | static-shape | n/a (static) | n/a | n/a |
| `hip.matmul` | static-shape | n/a | n/a | n/a |
| `hip.matmul_nbits` | static-shape | n/a | n/a | n/a |
| `hip.qmoe` | static-shape | n/a | n/a | n/a |
| `hip.rms_norm` | static-shape | n/a | n/a | n/a |
| `hip.layer_norm` | static-shape | n/a | n/a | n/a |
| `hip.skip_rms_norm` | static-shape | n/a | n/a | n/a |
| `hip.rope` | static-shape | n/a | n/a | n/a |
| `hip.miopen.add` | broadcast propagator | `buildBroadcastDimSpec` (registered) | YES | n/a |
| `hip.mul` | broadcast propagator | `buildBroadcastDimSpec` | YES | n/a |
| `hip.add` | broadcast propagator | `buildBroadcastDimSpec` | YES | n/a |
| `hip.min` | broadcast propagator | `buildBroadcastDimSpec` | YES | n/a |
| `hip.miopen.softmax` | single-input propagator | broadcast-style (registered as cast/unary) | YES (rare dyn case) | n/a |
| `hip.transpose` | single-input propagator | `buildTransposeDimSpec` (registered) | YES | identity-perm (`perm == [0..rank)`) |
| `hip.gather` | propagator (Cat-C-when-data-or-indices-dyn) | **NEW: `buildGatherDimSpec`** | YES | n/a |
| `hip.range` | **publisher** (Cat-C) / Cat-B fast path | NEW for non-publisher case: `buildRangeDimSpec` returning `CeilDiv(Sub(limit,start), delta)` | YES (publisher already done) | n/a |
| `hip.silu` / `sigmoid` / `softplus` / `gelu` / `reciprocal` / `sqrt` / `cos` / `sin` / `neg` / `sign` / `mod` | unary propagator | `buildBroadcastDimSpec` (registered) | YES | n/a |
| `hip.div` / `equal` / `and` / `not` / `less` / `sub` / `where` | broadcast propagator | `buildBroadcastDimSpec` (registered) | YES | n/a |
| `hip.cast` | single-input propagator | `buildBroadcastDimSpec` (registered) | YES | identity-cast (input elem type == output elem type) |
| `hip.reduce_sum` | reducing propagator | **NEW: `buildReduceDimSpec`** | YES | reduce-noop (empty axes + `noop_with_empty_axes=1`) |
| `hip.reduce_max` | reducing propagator | **NEW: `buildReduceDimSpec`** | YES | reduce-noop |
| `hip.reduce_prod` | reducing propagator | **NEW: `buildReduceDimSpec`** | YES | reduce-noop |
| `hip.causal_conv_with_state` | static-shape | n/a | n/a | n/a |
| `hip.gqa` | static-shape | n/a | n/a | n/a |
| `hip.hipdnn_graph` | static-shape | n/a | n/a | n/a |
| `hip.multi_head_attention` | static-shape | n/a | n/a | n/a |
| `hip.linear_attention` | static-shape | n/a | n/a | n/a |
| `hip.gemm` | static-shape | n/a | n/a | n/a |
| `hip.cumsum` | reducing propagator (pass-through shape) | **NEW: `buildCumSumDimSpec`** (pass-through) | YES | n/a |
| `hip.pad` | propagator (output_dim = input_dim + start_pad + end_pad) | **NEW: `buildPadDimSpec`** | YES | n/a |
| `hip.tile` | propagator (output_dim = input_dim * repeats[i]) | **NEW: `buildTileDimSpec`** | YES | identity-tile (`repeats == 1`) |
| `hip.expand` | propagator (output_dim from shape operand) | **NEW: `buildExpandDimSpec`** | YES | identity-expand (target shape == input shape) |
| `hip.slice` | propagator (output_dim = ceildiv(end-start, step)) | **NEW: `buildSliceDimSpec`** | YES | identity-slice (starts=0, ends=full, steps=1) |
| `hip.gather_nd` | propagator | **NEW: `buildGatherNDDimSpec`** | YES | n/a |
| `hip.scatter_nd` | propagator (pass-through: output shape == data shape) | **NEW: `buildScatterNDDimSpec`** | YES | n/a |
| `hip.size` | shape-data op | **NEW: extend `resolveValueFromI64Tensor`** | n/a (value-producing) | n/a |
| `hip.constant_of_shape` | **publisher** (Cat-C) | already done | already done | n/a |
| `hip.nonzero` | **publisher** (Cat-C) | already done | already done | n/a |
| `hip.shape` | shape-data op | **NEW: extend `resolveValueFromI64Tensor`** for intermediate `x` | n/a | identity-shape (static-only inputs always fold; only dyn-rank case matters) |

**Coverage summary:** 12 builder gaps to fill in Phase 2; 6 identity predicates to register in Phase 4.

### View propagators (produced by ONNX-to-HIP lowering)

The following ONNX ops do NOT have `hip.*` equivalents; they lower to MLIR view ops:

| ONNX op | Lowering | Slot implications |
| --- | --- | --- |
| `onnx.Reshape` | `tensor.expand_shape` / `tensor.collapse_shape` | View pass-through; storage inherited |
| `onnx.Squeeze` | `tensor.collapse_shape` | View pass-through |
| `onnx.Unsqueeze` | `tensor.expand_shape` | View pass-through |
| `onnx.Split` | `tensor.extract_slice` per output | View pass-through; multi-result handled by view-per-result, not slot-grid |

Phase 2.5 verifies that `PromoteStridedHipOperands` materializes contiguous copies for hip-op consumers downstream of these views without breaking slot pointer propagation.

---

## Per-Model Chain Trace

### embedding.onnx (FULL TRACE)

Source: `test/numeric/tests/test_nonzero_qwen_embedding.py` documents the full graph
(extracted from `D:\Develop\m\models\Qwen3.5-35B-A3B_int4_rtn_128gs_cuda\embedding.onnx`).
Op counts:

- 1× `NonZero` (Cat-C publisher), 1× `Transpose` (single-input propagator),
  1× `Reshape` (view), 1× `Slice` (propagator), 1× `ScatterND` (propagator),
  1× `Shape`, 1× `Gather`-on-Shape, 1× `Unsqueeze` (view), 1× `Gather` (data),
  1× `Equal` (broadcast), 1× `Expand` (propagator).
- 0× `Loop`, 0× `Concat` (in the active chain), 0× `Reduce*`.

#### Chain #1 — primary Cat-C chain

```
Gather(embed_tokens.weight, input_ids)         -> [B, S, 4096] fp16   STATIC
Equal(input_ids, IMAGE_TOKEN_ID)               -> [B, S]   bool       STATIC
Unsqueeze                                      -> [B, S, 1]            STATIC (view)
Expand(predicate, Shape(Gather))               -> [B, S, 4096] bool    STATIC (Cat-A: shape from Gather)
NonZero(Expand)                                -> [3, N]               CAT-C PUBLISHER (slot S0)
Transpose(NonZero, perm=[1,0])                 -> [N, 3]               propagator (slot S0 in dim 0)
Reshape(image_features, [-1])                  -> [F]                  view (storage inherits)
Slice(Reshape, [0], [N], [0])                  -> [N]                  propagator (slot S0 in dim 0)
ScatterND(Gather, Transpose, Slice)            -> [B, S, 4096] fp16    STATIC output, dyn intermediate
```

Phase mapping:

| Op | Phase work item | Notes |
| --- | --- | --- |
| `NonZero` | Phase 1 elide DPS-init; Phase 2 schema upgrade | publisher today, exact-size buffer published, but UB DPS-init alloc is still reserved |
| `Transpose` | Phase 2 retarget wrapper + lowering | currently writes into its own UB DPS-init; Phase 3 coalesces with NonZero |
| `Slice` | Phase 2 add `buildSliceDimSpec` + retarget wrapper | Cat-C-downstream; needs builder + slot publish |
| `ScatterND` | Phase 2 add `buildScatterNDDimSpec` (pass-through) | static output but dyn inputs; consumer needs slot pointer rewiring (today works via `hipdnn.input_slot_buffers`) |
| `Shape(Gather)` for Expand | Strategy 3 of `getResultDimSpec` (static) | already works — `Shape(static-tensor)` folds at conversion in `ConstantOfShapeConversion.cpp` style |

Phase 3 expected coalescing outcome: 4 slot allocations (S0 from NonZero, plus the slot S0 propagates through Transpose / Slice / ScatterND) — but each writes its own buffer today. Phase 3 collapses Transpose's buffer to share storage with NonZero's (when lifetimes don't overlap, which they don't in this linear chain), and Slice's buffer with the same arena offset.

#### Chain #2 — degenerate path on N=0

Same chain, but `NonZero` publishes `N=0`. Verifies the empty-publish + zero-byte ScatterND path. Already tested today. No new Phase work required; Phase 3 coalescing must remain correct on N=0 (a zero-byte slot still occupies the same arena offset, just with zero published bytes).

### text.onnx (FULL TRACE — Cat-C chains)

Source: `D:\Develop\m\models\Qwen3.5-35B-A3B_int4_rtn_128gs_cuda\text.onnx`. 2761 nodes, 27 unique op types (all supported by the dialect).

#### Op surface

- **20× Range** producing data-dependent shapes that feed MoE expert dispatch chains (each Range's count depends on per-token routing decisions).
- **20× Reshape** consuming Range outputs in the QMoE pre-processing (view propagators).
- **40× Split** in per-expert routing (multi-result, lowers to `tensor.extract_slice` per result).
- **20× Concat** (NO converter today — see [out-of-scope](#out-of-scope-blockers); most static-fold at compile time, those that don't would need a `ConcatConversion.cpp`).
- **21× Shape** — most are static-folded; a small handful drive Reshape/Range.
- **MoE-internal**: per-expert chain runs through `hip.qmoe`, which today has its own bucketing and the Cat-C Range outputs are consumed inside the QMoE wrapper rather than re-exported.

#### Chain #1 — generic Range Cat-C → Reshape → QMoE

```
Range(start=Gather(seqlens), limit=expert_count_constant, delta=1)  -> [T] CAT-C (slot S_k)
Reshape(Range, [-1, expert_block_size])                              -> view (storage inherits)
QMoE(..., reshape_output, ...)                                       -> consumes slot S_k indirectly via descriptor sizes
```

Phase mapping:

- `Range` Cat-C: Phase 1 elide DPS-init for the 20 publishers.
- `Reshape`: view pass-through (Phase 2.5 verification only).
- `QMoE`: static-output op; consumes the slot via descriptor `sizes[0]` (`AnnotateInputDimSlotsPass` already handles this).

**Verification needed**: do the 20 Cat-C `Range` chains actually reach `wrap_qmoe`, or are they folded inside the per-expert routing logic such that the Cat-C path never fires? This needs a verification run with `HIPDNN_EP_DEBUG=1` on a text.onnx subgraph extract during Phase 1 testing — if the 20 publishers fold away (per-expert routing keeps its own count internally), the 20-publisher pool-size regression assertion shifts to a synthetic test.

#### Cat-B compounds (resolved EP-side, no slot work)

The text.onnx symbol space contains: `batch_size * sequence_length`, `sequence_length * num_attention_heads`, `sequence_length * num_key_value_heads`.

All three are Cat-B compounds rooted at func-args. The existing EP-side `DimSpecResolver` handles single-leaf `InputDim`; the compound `Mul(InputDim, InputDim)` needs the Phase 2.3 EP-side resolver extension to evaluate `Mul`. **The DimSpec algebra already supports `Mul` (lib/Dialect/IR/HipShapeInterface.cpp line 256)** — only the EP-side evaluator needs the extension. No slot machinery work.

### vision.onnx (SAMPLED TRACE — 5 representative Cat-C chains)

Source: `D:\Develop\m\models\Qwen3.5-35B-A3B_int4_rtn_128gs_cuda\vision.onnx`. 2077 nodes, 41 unique op types.

#### Out-of-scope items (vision-specific)

- **27× Loop** — see [out-of-scope blockers](#out-of-scope-blockers). Vision e2e is blocked.
- **81× CastLike** — already handled by `simplifyCastLike` in `lib/Conversion/OnnxToHip/SimplifyOnnx.cpp`. No slot work.

#### Cat-C publisher inventory (main graph)

- **6× Range** Cat-C
- **2× Tile** Cat-C (repeats fed by computed values)
- **2× Expand** Cat-C (shape operand fed by computed values)
- **1× ConstantOfShape** Cat-C (shape from intermediate)
- **Cat-C propagator hits in main graph**: 33× Reshape, 54× Slice, 5× Gather

#### Sampled chain #1 — Range → propagator chain

```
Range(0, num_patches, 1)            -> [num_patches]  CAT-C (slot S_v0)
Reshape(Range, [grid_h, grid_w])    -> view
Gather(patch_embed_table, Reshape)  -> [grid_h, grid_w, dim]  propagator
```

Phase mapping: `Range` Cat-C → Phase 1 elide; `Reshape` → view pass-through; `Gather` → Phase 2 `buildGatherDimSpec` + retarget wrapper.

#### Sampled chain #2 — Tile → consumer

```
Tile(positional_embed, repeats=[1, num_repeats, 1])  -> [1, num_repeats, dim]  CAT-C (slot S_v1)
Add(Tile, base_embed)                                -> propagator (broadcast)
```

Phase mapping: `Tile` Cat-C → Phase 2 `buildTileDimSpec` + slot publisher convention (similar to Range Cat-C; `repeats` operand resolves to a non-func-arg value, so this is Cat-C); `Add` → broadcast retarget.

#### Sampled chain #3 — Expand → consumer

```
Expand(small_tensor, shape=Concat(static_dims, dyn_dim))  -> Cat-C if shape operand is intermediate
ScatterND(..., Expand, ...)                                -> propagator
```

Phase mapping: `Expand` Cat-C → Phase 2 `buildExpandDimSpec` + retarget; `ScatterND` → Phase 2 `buildScatterNDDimSpec`.

#### Sampled chain #4 — Slice (Cat-C-downstream)

```
some_dyn_input  -> Slice(input, [0], [dyn_end], [0])  -> propagator (slot from dyn_end)
```

`dyn_end` comes from a Cat-C upstream (e.g. Shape(NonZero)). Phase 2 `buildSliceDimSpec` returning `CeilDiv(Sub(end, start), step)` over operand-value DimSpecs.

#### Sampled chain #5 — Cat-B compound symbol

```
ScatterND(..., update_with_shape ((144 * u6 * u7 * u8) // num_patches), ...)
```

This is the canonical FloorDiv compound. Resolved EP-side (no slot work) once Phase 2.3 adds FloorDiv/Mul to the EP-side resolver. The compound is rooted at func-args (u6, u7, u8, num_patches), so it's Cat-B.

---

## Out-of-scope blockers

The following vision.onnx structural items are surfaced here so the plan is honest about what it does NOT fix:

### Loop (27 occurrences in vision.onnx)

ONNX control flow. No `hip.loop` op exists in `include/hip/Dialect/IR/HipOps.td`,
no converter in `lib/Conversion/OnnxToHip/`, no LLVM lowering. End-to-end vision
compilation requires a separate `Loop` lowering work-item (likely unrolling or
scf.while-style structured control flow), which this plan does NOT undertake.
The slot machinery designed here remains correct in the presence of `Loop` once
that op gets a converter, because each loop body is its own region and slot
allocation can be scoped per region.

Each `Loop` body in vision.onnx contains: 3× Slice, 2× Gather, 1× MultiHeadAttention,
1× Concat-as-accumulator. The accumulator semantics will require careful slot
lifetime handling (the loop body's accumulator slot lives across iterations),
which is deferred.

### CastLike (81 occurrences in vision.onnx)

Already handled: `lib/Conversion/OnnxToHip/SimplifyOnnx.cpp` line 64
`simplifyCastLike` rewrites it to `onnx.Cast` + a dead type-donor function
argument before the dialect conversion runs. No slot work required.

### QMoE Cat-C Range fold (text.onnx)

The existing `hip.qmoe` runtime path already handles per-token expert dispatch;
whether the 20 Cat-C `Range` publishers in text.onnx actually surface as runtime
slots (vs being folded into QMoE's per-block routing logic) needs verification
during Phase 1 testing. If they fold away, the 20-publisher pool-size assertion
shifts to a synthetic test.

---

## Pattern Coverage Matrix

| Pattern | Models exercising it | Count | Phase coverage | Notes |
| --- | --- | --- | --- | --- |
| NonZero → propagator chain | embedding.onnx | 1 | Phase 1 + 3 | end-to-end proven today |
| Range Cat-C → Reshape → QMoE | text.onnx | 20 | Phase 1 + 2 | first multi-publisher case |
| Range Cat-C → general propagator | vision.onnx | 6 | Phase 1 + 2 + 4 | synthetic test coverage |
| Tile Cat-C → consumer | vision.onnx | 2 | Phase 2 (`buildTileDimSpec`) | synthetic |
| Expand Cat-C → consumer | vision.onnx | 2 | Phase 2 (`buildExpandDimSpec`) | synthetic |
| ConstantOfShape Cat-C → consumer | vision.onnx, embedding (indirect) | 1+ | Phase 2 (schema upgrade) | already partially proven |
| Slice Cat-C-downstream | embedding, text, vision | many | Phase 2 (`buildSliceDimSpec`) | needs retarget |
| Gather Cat-C-downstream | embedding, text, vision | 6+ | Phase 2 (`buildGatherDimSpec`) | needs retarget |
| ScatterND Cat-C-downstream | embedding | 1 | Phase 2 (`buildScatterNDDimSpec`) | pass-through, easy |
| Cat-B compound `bs * seq`, etc. | text.onnx | many | Phase 2.3 EP-side resolver (Mul) | no in-DLL slot work |
| Cat-B compound `((144 * u6 * u7 * u8) // np)` | vision.onnx | many | Phase 2.3 EP-side resolver (FloorDiv + Mul) | synthetic test |
| Reshape view (storage inherits) | all models | hundreds | Phase 2.5 verification only | no retarget |
| Loop | vision.onnx | 27 | **OUT OF SCOPE** | separate work-item |
| CastLike | vision.onnx | 81 | already handled | no action |
| Concat aggregating | text.onnx (20), vision.onnx (65) | 85 total | converter doesn't exist | landing Concat converter unblocks Phase 2 retargeting via `buildConcatDimSpec` |

---

## Phase-by-Phase Coverage Decisions

### Phase 1 (elide publisher DPS-init)

| Publisher | embedding | text | vision (main) |
| --- | --- | --- | --- |
| `NonZero` | 1× | 0 | 0 |
| `Range` Cat-C | 0 | 20× (subject to QMoE fold verification) | 6× |
| `ConstantOfShape` Cat-C | 0 (Cat-A only) | depends on chain | 1× |
| `Tile` Cat-C (new publisher) | 0 | 0 | 2× |
| `Expand` Cat-C (new publisher) | 0 | 0 | 2× |

**Phase 1 work**: attach `hipdnn.elide_dps_init = true` in the converters of the
THREE existing Cat-C publishers (NonZero, Range Cat-C, ConstantOfShape Cat-C).
`Tile` and `Expand` Cat-C are NOT publishers today — they're propagators whose
operands trace to Cat-C upstream. Phase 2 handles those.

### Phase 2 (retarget propagators + builders)

12 new DimSpec builders required (see [Hip_DpsOp inventory](#hip_dpsop-inventory-from-includehipdialectirhipopstd)).
Schema upgrade: replace scalar `slot_id` + ad-hoc `slot_ids` with uniform
`hipdnn.output_slot_ids` per-result-per-dim array (backward compatible).

Compound DimSpec evaluator covers all seven `DimSpecKind` arithmetic kinds:
`Add`, `Sub`, `Mul`, `FloorDiv`, `CeilDiv`, `Min`, `Max` (the enum already defines
all seven in `lib/Dialect/IR/HipShapeInterface.cpp` lines 252–264).

### Phase 3 (SlotLifetimeCoalesce)

Output-bound lifetime model classifies each slot as **intermediate-only** or
**output-bound** (consumed by a `hipdnn.output_index`-tagged result).
Output-bound slots cannot coalesce with overlapping intermediate slots.

Bin-packing template: greedy first-fit-decreasing on lifetime overlap (same
template as `lib/Dialect/Transforms/PoolAllocs.cpp::packDynamicAllocs` line 200).

### Phase 4 (identity-rebind)

Six identity predicates required:

| Op | Identity condition |
| --- | --- |
| `hip.transpose` | `perm == [0..rank)` |
| `hip.cast` | input elem type == output elem type |
| `hip.expand` | target shape (resolved per-dim) == input shape |
| `hip.slice` | starts=0, ends=full, steps=1 on every axis |
| `tensor.expand_shape` / `collapse_shape` | always identity from slot perspective (view) |
| `hip.reduce_*` | empty axes set AND `noop_with_empty_axes=1` |

---

## Verification gate before Phase 1 begins

1. ✅ Op-class table complete (every `Hip_DpsOp` classified).
2. ✅ Per-model chain trace complete (embedding full, text full, vision sampled).
3. ✅ Out-of-scope items explicitly listed.
4. ✅ Phase coverage decisions tied to model evidence.

No silent gaps remain. Phase 1 is unblocked.

---

## Implementation status (as of 2026-05-25)

All four phases of the generalized A → B → C → D plan have landed. Per-phase
implementation summary:

### Phase 1 — `ElideSlotPublisherAllocsPass`

- File: [`lib/Dialect/Transforms/ElideSlotPublisherAllocs.cpp`](../../lib/Dialect/Transforms/ElideSlotPublisherAllocs.cpp).
- Cat-C converters attach `hipdnn.elide_dps_init` UnitAttr.
- Pipeline position: post-bufferize, pre-`hip-pool-allocs`.
- Effect: shrinks the DPS-init `memref.alloc` of every Cat-C publisher to a
  0-byte placeholder so `hip-pool-allocs` reserves no static-pool bytes for it.
- Regression test: [`test/lit/e2e/test_elide_slot_publisher_pool_size.mlir`](../../test/lit/e2e/test_elide_slot_publisher_pool_size.mlir)
  asserts `hipdnn.pool_size = 0` for a NonZero-only graph.

### Phase 2 — Translucent propagator slot retargeting + compound consumer

- Schema: `hipdnn.output_slot_ids` per-result-per-dim `DenseI32ArrayAttr`
  array form (legacy `slot_id` scalar + `slot_ids` array still recognised by
  the annotator and coalescer).
- DimSpec builders added for `range`, `tile`, `expand`, `gather`, `slice`,
  `pad`, `reduce_sum`, `reduce_max`, `reduce_prod`, `cumsum`, `scatter_nd`,
  `gather_nd`, `concat` (registered in `populateBuiltinDimSpecBuilders`).
- `ReservePropagatorSlotsPass` reserves fresh slot ids for translucent
  propagator output dims whose DimSpec contains a `RuntimeSlot` leaf.
- Compound DimSpec evaluator: `hipdnn_ep_state_eval_dim_spec` runtime helper
  + `GenerateInterface`-emitted LLVM IR for `Add / Sub / Mul / FloorDiv /
  CeilDiv / Min / Max` over `RuntimeSlot / InputDim / InputValueI64 / Static`
  leaves.
- Runtime ABI: `dyn_pool_alloc_for_slot` (publisher fresh-alloc),
  `publish_buffer_resize` (Phase 3 reuse).

### Phase 3 — `SlotLifetimeCoalescePass`

- File: [`lib/Dialect/Transforms/SlotLifetimeCoalesce.cpp`](../../lib/Dialect/Transforms/SlotLifetimeCoalesce.cpp).
- Algorithm: group slot ids by canonical DimSpec bytes
  (`DimSpec::canonicalize()` constant-folds + sorts commutative children),
  first-fit-decreasing bin-pack within each group on lifetime, pick smallest
  slot id per bin as the representative, rewrite every slot reference
  (publisher attrs, consumer attrs, `RuntimeSlot(N)` leaves inside
  `output_dim_specs`, module-level `hipdnn.output_dim_specs`).
- Output-bound model: slots referenced from module-level
  `hipdnn.output_dim_specs` get `lastUseIdx = +inf`; two output-bound slots
  cannot share storage (verifier-enforced in `VerifyDimSpecsPass`).
- Disabled with `--hip-slot-lifetime-coalesce=disable-coalesce=true` for A/B
  footprint testing.
- Regression test: [`test/lit/e2e/test_slot_lifetime_coalesce_footprint.mlir`](../../test/lit/e2e/test_slot_lifetime_coalesce_footprint.mlir)
  asserts a 2-publisher × 2-propagator chain shrinks to 2 dyn slots end-to-end.
- LIT positive + negative coverage: [`test/lit/Transforms/slot-lifetime-coalesce.mlir`](../../test/lit/Transforms/slot-lifetime-coalesce.mlir).

### Phase 4 — `IdentityPropagatorRebindPass`

- File: [`lib/Dialect/Transforms/IdentityPropagatorRebind.cpp`](../../lib/Dialect/Transforms/IdentityPropagatorRebind.cpp).
- Identity predicate registry in
  [`lib/Dialect/IR/HipShapeInterface.cpp`](../../lib/Dialect/IR/HipShapeInterface.cpp);
  built-in predicates for `hip.transpose` (identity perm), `hip.cast` (same
  dtype), `hip.expand` (same shape), `hip.slice` (full-range), `hip.tile`
  (all-ones repeats), `hip.reduce_*` (`noop_with_empty_axes=1` + shape match).
- Pipeline position: PRE-bufferize, in the OnnxToHip tail (the pass relies on
  `op->getResult(0)` being a real SSA value, which is the tensor-IR form).
- Effect: erases identity ops, RAUW their result with the upstream input,
  remaps every slot id reference from the eliminated op's slot to the input's
  slot.
- Disabled with `--hip-identity-propagator-rebind=disable-identity-rebind=true`
  for A/B kernel-launch testing.
- Regression test: [`test/lit/e2e/test_identity_transpose_elision.mlir`](../../test/lit/e2e/test_identity_transpose_elision.mlir)
  asserts a two-identity-Transpose graph emits ZERO `wrap_transpose` calls.
- LIT per-class coverage: [`test/lit/Transforms/identity-propagator-rebind.mlir`](../../test/lit/Transforms/identity-propagator-rebind.mlir).

### Diagnostic — `--hip-print-pool-stats`

- File: [`lib/Dialect/Transforms/PrintPoolStats.cpp`](../../lib/Dialect/Transforms/PrintPoolStats.cpp).
- Analysis-only pass that prints `[pool-stats]` blocks to stderr with
  `static_pool_bytes`, `dyn_dim_slots_count`, `next_dyn_slot_id`,
  `output_dim_specs.count`, `identity_propagator_predicate_hits`,
  `slot_publisher_count`. Use as the before/after pivot per phase.
- Test: [`test/lit/Transforms/print-pool-stats.mlir`](../../test/lit/Transforms/print-pool-stats.mlir).

---

## Per-class invariants

These are the invariants the slot-buffer-coalesce machinery relies on. Any
new op added to a class MUST satisfy them or coalesce / identity-rebind will
produce wrong code that compiles cleanly and runs without crashing.

### Publisher invariants (Cat-C: `hip.nonzero`, `hip.range` dyn, `hip.constant_of_shape` dyn)

1. **Exact-size allocation**. The wrapper allocates a buffer sized to the
   actually-produced element count, not the upper-bound. Use
   `hipdnn_ep_state_dyn_pool_alloc_for_slot(state, bytes, slot)` so the
   buffer pointer is published atomically with the dim.
2. **`slot_id` attribute** must be a non-negative i32 IntegerAttr on the op.
   The compiler-side `ReservePropagatorSlotsPass` skips ops that already
   carry a `slot_id` (i.e. publishers are reserved at conversion time, not
   later).
3. **`hipdnn.elide_dps_init = true`** must be set so
   `ElideSlotPublisherAllocsPass` skips the upper-bound DPS-init alloc.
   Forgetting this attribute does not affect correctness but doubles
   memory (the elided alloc still claims an arena slot).
4. **No reads from the DPS-init buffer** — publishers MUST overwrite, never
   read-modify-write, the DPS init memref. Verified at runtime via the null
   buffer pattern (when `N=0`, the publisher publishes a null pointer and
   downstream peek-helpers return null safely).
5. **Output-lifetime contract**: if the publisher's slot is referenced
   from module-level `hipdnn.output_dim_specs`, the buffer MUST survive past
   `inference_compute` return. `SlotLifetimeCoalescePass` enforces this by
   assigning `lastUseIdx = +∞` to output-bound slots.

### Single-input / broadcast propagator invariants (`hip.transpose`, `hip.add`, `hip.cast`, …)

1. **Rank-preserving** (or rank-broadcasting for binary ops). Inherits the
   upstream `slot_id` via `findDpsWriter` walking through the DPS chain.
2. **`output_dim_specs` array** must be a faithful symbolic re-expression
   of the inputs' DimSpecs (`buildBroadcastDimSpec` / `buildTransposeDimSpec`
   from `populateBuiltinDimSpecBuilders`). Any new op in this class MUST
   register a builder or be covered by a generic broadcast registration.
3. **Identity predicate** (Phase 4): if the op carries an attribute that
   makes it a runtime no-op (e.g. `perm = [0..rank)` for transpose, same
   `to_dtype` for cast), it can be erased by
   `IdentityPropagatorRebindPass`. Predicates MUST be compile-time only —
   they can read MLIR attributes and result types but never runtime values.

### Aggregating propagator invariants (`hip.concat` — future)

1. **Output dim = SUM of operand dims along the axis**. DimSpec builder
   must produce `Add(InputDim(0, axis), …, InputDim(N-1, axis))` so the
   coalescer's canonicalization sees structurally equivalent aggregates as
   merge candidates.
2. **Slot allocation must size for the aggregated count**. Compound
   DimSpec evaluator (`hipdnn_ep_state_eval_dim_spec`) handles this on the
   consumer side.

### Reducing propagator invariants (`hip.reduce_sum`, `hip.reduce_max`, `hip.reduce_prod`, `hip.cumsum`)

1. **`axes` attribute drives the builder**. `keepdims=0` removes the dim;
   `keepdims=1` replaces it with `Static(1)`.
2. **`noop_with_empty_axes=1` + matching shape** is the identity predicate:
   `IdentityPropagatorRebindPass` will erase such an op.
3. **CumSum is the rare exception** — it is rank-preserving but treated as
   reducing for builder dispatch because its output may be shorter than the
   input when `exclusive=1`. The builder for cumsum must inspect the
   attribute.

### View propagator invariants (`tensor.expand_shape`, `tensor.collapse_shape`, `tensor.extract_slice`, `memref.subview`)

1. **Zero kernel launches, zero slot publishes**. These are MLIR view
   ops; storage is inherited from the source memref.
2. **DimSpec propagation via `resolveDimFromValue`**. The
   `findDpsWriter` walker treats views as transparent — the upstream
   producer's slot id is inherited without any new slot reservation.
3. **`PromoteStridedHipOperands` materializes contiguous copies** when a
   downstream `hip.*` op requires contiguous storage. This may introduce a
   new alloc, but the alloc is static-shape (the view ranks/strides are
   compile-time) and goes through `hip-pool-allocs` like any other.

### Shape-data op invariants (`hip.shape`, `hip.size`)

1. **Output is a fixed-shape `i64` tensor encoding the input's dims**.
   Consumers read individual elements via `resolveValueFromI64Tensor`,
   which knows how to lift each element back into a DimSpec leaf
   (`InputValueI64` for input elements, lifted Static for indexed shape
   elements).
2. **Identity predicate is rarely hit** — static-shape inputs already fold
   to a constant during ONNX-to-HIP conversion (`ShapeToConstant` fast
   path); only dynamic-rank inputs survive to the HIP dialect.

### Slot-id renumber invariants (enforced by `SlotLifetimeCoalescePass` + `IdentityPropagatorRebindPass`)

Every slot id rewrite MUST update **all** of:

1. Publisher `slot_id` IntegerAttr and `slot_ids` DenseI32ArrayAttr.
2. Consumer `hipdnn.input_dim_slots` (per-operand `[dim_idx, slot_id]` pairs).
3. Consumer `hipdnn.input_slot_buffers` (per-operand i32 slot id, -1 = use
   descriptor pointer).
4. Op-level `hipdnn.output_slot_ids` (per-result-per-dim
   `DenseI32ArrayAttr` array form).
5. `RuntimeSlot(N)` leaves inside `output_dim_specs` ArrayAttr (kind=3,
   slot id at field index 5).
6. Module-level `hipdnn.output_dim_specs` (same RuntimeSlot leaf
   rewrite).
7. Module-level `hipdnn.output_slot_ids` (per-EP-output i32 slot id).
8. Module-level `hipdnn.dyn_dim_slots_count` (final count after renumber).

Skipping any of these leaves stale references that point at a defunct slot
id. The EP-side resolver reads the module-level metadata after
`stream_sync` to populate output OrtValue shapes — stale references there
manifest as random output shapes (often zero), which downstream consumers
silently accept and produce numerically wrong outputs that pass
shape-only conformance.
