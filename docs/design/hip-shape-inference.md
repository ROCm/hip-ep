<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP dialect shape inference

**Date:** 2026-07-24
**Document Type:** Design
**Status:** Implemented
**Related:** [unranked-tensor-handling.md](unranked-tensor-handling.md), [pool-allocs-memory-planning.md](pool-allocs-memory-planning.md), [output-allocator-design.md](output-allocator-design.md), [compiler-runtime-contract.md](compiler-runtime-contract.md), [pipeline_pass_menu.md](../pipeline_pass_menu.md)

## Purpose

HIP destination-passing-style (DPS) operations expose shape information through standard MLIR interfaces. The design has four layers:

1. **Construction-time result typing.** `InferTypeOpInterface` can derive tensor result types from DPS init operand types.
2. **Dynamic-dimension reification.** `ReifyRankedShapedTypeOpInterface` exposes each result dimension as an `OpFoldResult`.
3. **Module-level refinement.** `--hip-infer-shapes` uses reification to narrow dynamic result dimensions before bufferization while preserving DPS type equality.
4. **Static verification.** Operations with non-trivial static shape contracts may add targeted verifiers.

The implementation prefers standard MLIR interfaces, helpers, and canonicalization patterns over dialect-specific shape infrastructure.

Shape ownership is intentionally split:

| Job | Owner | Pipeline position |
|---|---|---|
| Establish rank on an unranked loop-carried value | Importer contract and `--hip-infer-loop-body-shapes` | Before ONNX-to-HIP conversion |
| Narrow `?` dimensions within known rank | `--hip-infer-shapes` | After ONNX-to-HIP conversion |
| Fold `tensor.dim` through reification and reshape chains | `--hip-resolve-tensor-dims` | After shape refinement, before bufferization |

ONNX-level shape inference remains upstream's responsibility. `--hip-infer-shapes` does not convert `UnrankedTensorType` into a ranked type; see [unranked-tensor-handling.md](unranked-tensor-handling.md).

## Shape representation

The foundation represents compile-time extents in `RankedTensorType` and
runtime extents as ordinary index SSA carried by `OpFoldResult`. It does not
maintain a persistent inter-operation constraint set for facts such as “these
two dynamic dimensions are equal.” Type-level verification treats dynamic
extents as compatible unknowns. Reification derives runtime extents from
operand shapes.

This does not preclude symbolic reasoning. A future analysis may use MLIR's
`ValueBoundsOpInterface` and external models over the same dimension SSA without
changing the infer/reify interfaces defined here. Prefer that standard mechanism
before introducing another feature-specific symbolic analysis. Frontend payload
provenance, such as reconstructing a shape tensor's values, is a separate
problem from proving affine equality between dimensions.

Phase 1 of `--hip-infer-shapes` is intentionally local rather than a global
fixpoint: each operation is refined once in producer-before-consumer order, and
cast barriers preserve existing consumer signatures instead of propagating
narrowed types through the whole graph. The loop-signature phase iterates only
because outlined loop body signatures form an explicit cyclic contract.

## DPS shape contract

In tensor mode, each HIP DPS tensor result must equal the corresponding `outs` operand type. In memref mode, the operation has no tensor result; it writes directly through the destination memref.

This gives two related but distinct jobs:

- **InferType** avoids restating result types at converter call sites when the result is already represented by a typed destination.
- **Reify** makes static and dynamic result dimensions available to downstream transformations.

## MLIR interfaces

| Interface | Role in hip-ep |
|---|---|
| `InferTypeOpInterface` | Builds result types from DPS init operand types |
| `ReifyRankedShapedTypeOpInterface` | Materializes static dimensions as attributes and dynamic dimensions as SSA values |
| `InferShapedTypeOpInterface` | Not used as the primary HIP DPS contract |
| `HipDpsOpInterface` | Dialect marker interface extending `DestinationStyleOpInterface`; owns the shared default reification body |

`HipDpsOpInterface` is a generated MLIR `OpInterface`, but it is not a replacement for the standard InferType/Reify contracts. It marks HIP DPS compute operations and provides their shared default reification behavior. In tensor mode it walks `DestinationStyleOpInterface::getDpsInits()` and returns each destination's mixed sizes through `tensor::getMixedSizes`, exactly one vector per SSA result. In memref mode there are no SSA results, so it succeeds with an empty list.

Operations whose shape contract is more specific than "result shape equals destination shape" opt out of the default and provide a dedicated reification implementation.

## TableGen wiring

`Hip_DpsOp` is a structural two-parameter root (`mnemonic`, `traits`) that
centralizes the interfaces and variadic tensor results used by HIP compute
operations. Shape behavior is selected by a named family rather than by
independent booleans or injected function-body parameters:

- `Hip_DpsOp_AutoReify` forwards to the shared `HipDpsOpInterface` body.
- `Hip_DpsOp_WithInfer` emits single-result typing from a named outs accessor.
- `Hip_DpsOp_AutoReifyInfer` combines those two common behaviors.
- `Hip_DpsOp_Broadcast` and `Hip_DpsOp_Reduction` own fixed family reification
  bodies that call shared C++ helpers.

Families use typed TableGen `code` fragments only for their forwarding methods.
The shape rules and validation remain in `HipShapeUtils` C++ helpers, while
semantic long-tail operations keep handwritten reification methods.

Multi-result and specialized operations may keep inferred result construction disabled when a single generated body cannot describe all results.

## Shape-contract mechanisms

Choose the smallest mechanism that matches the operation's semantics:

| Shape contract | Mechanism |
|---|---|
| Result shape equals DPS init shape, including most multi-result DPS ops | Shared `HipDpsOpInterface` default |
| Result shape equals a named input | `reifyElementwiseSameShape` or a small dedicated thunk |
| NumPy-style broadcast | `Hip_DpsOp_Broadcast` and broadcast helpers |
| Reduction with constant axes/keepdims | `Hip_DpsOp_Reduction` and reduction helpers |
| Permutation | `reifyTransposeByPerm` |
| Gather/GatherND/GatherElements | Gather-specific helpers or thunks |
| OneHot, Compress, TopK | Dedicated reification thunks |
| Pad, Tile, Expand, Slice, Range | Fold-or-bail helpers with fallback to DPS-init shape |
| MatMul/Gemm/MatMulNBits | Dedicated shape logic based on operand dimensions and attributes |
| Attention or normalization with multiple destinations | One shape vector per DPS init unless an op supplies a dedicated thunk |
| Convolution, pooling, or resize with converter-computed destinations | DPS-init shape, with semantic validity handled by conversion or verification |
| Runtime-dependent count, such as NonZero or Compress | DPS-init shape; unresolved dimensions remain dynamic |

Shared declarations live in `HipShapeUtils.h`; common implementation lives in
`HipShapeUtils.cpp`, with focused category translation units introduced by the
stack layer that first consumes each family. This foundation includes
matmul/Gemm, reduction, gather, and shape-operation helpers; later family PRs
add attention and convolution/pooling implementations. Operations that select
a manual-reification family define their member functions in
`HipReifyResultShapesImpl.cpp`.

Frontend-neutral destination construction lives in
`hip/Conversion/HipConversionUtils.h`: result-shape compatibility, broadcast
destination construction, and HIP context
lookup. `OnnxToHipUtils` retains only ONNX import semantics and wrappers, so a
future frontend can target HIP without depending on the ONNX conversion layer.

Pure descriptor transformations such as Reshape, Squeeze, and Unsqueeze generally lower to standard tensor operations rather than HIP DPS compute operations. Their shape inference and dim folding use MLIR's standard tensor interfaces and external models, not a second HIP-specific contract.

## Static result typing

For a single-result operation in `Hip_DpsOp_WithInfer` (directly or through a
more specific family), ODS provides an inferred-type `Op::create` overload. The
generated `inferReturnTypes` reads the typed DPS init and emits:

- one result type in tensor mode;
- no result type in memref mode.

Converters may use this overload instead of passing an explicit result-type list. Migration is per operation; explicit-type builders remain valid. Static type inference and dynamic-dimension reification are orthogonal: an operation may implement either interface without the other.

`InferTypeOpInterface` does not replace semantic shape verification. An operation such as MatMul may still verify that static operand dimensions satisfy its contract.

The generated InferType body intentionally covers the uniform single-result case. Multi-result operations and operations whose result types are not a one-to-one copy of DPS init types may retain explicit builders or provide custom `inferReturnTypes`.

## Dynamic-dimension reification

Reification returns one `OpFoldResult` for each result dimension:

- static dimensions become `IntegerAttr`;
- dynamic dimensions become existing SSA values or `tensor.dim`/`memref.dim` operations;
- value-dependent helpers fold constants when possible and otherwise fall back to the DPS init's mixed sizes.

Reification is allowed to create IR at the caller's insertion point. Helpers therefore reuse operand dimensions where possible, fold constant operands and attributes, and avoid pretending that a runtime-computed extent is static. For operations whose runtime extent cannot be represented before execution, the honest result remains dynamic.

Reification is per result: `reifyResultShapes` returns one shape vector for every tensor result. The number and rank of those vectors must match the operation's tensor results even when the implementation derives them from DPS init operands.

### Shared converter/reification shape helpers

Converter destination construction and result reification are two views of one
shape rule. `HipShapeUtils` therefore separates pure `infer*` helpers, which
validate static shapes without a builder, from `reify*` helpers, which may
insert index SSA into the input IR only after validation succeeds.

A reifier must validate every precondition before changing the input IR.
If folding can expose a conflict, build speculative operations in a detached
block and insert them through the caller's builder only on success.
Failure must leave the IR unchanged, including when the valid result shape is
rank zero; `FailureOr` distinguishes that empty success from failure.
Broadcast destination construction in `HipConversionUtils.cpp` uses the same
reifier as result-shape queries. For two dynamic extents, both use
`select(lhs == 1, rhs, lhs)`. This keeps the allocated size consistent with
the reported result shape, including through reshape and bufferization.
The builder checks folded extents against the imported result type before it
inserts new operations. Variadic Min/Max conversion retains the complete new
chain in a detached block until every pair succeeds.
Imported and inferred extents follow standard
shaped-type compatibility: a dynamic extent on either side is compatible,
while unequal static extents are contradictions.

Common DPS verification is similarly centralized in `verifyDpsComputeOp`. It
checks ranked tensor/memref uniformity, destination count, result count, and
tensor result/init type equality before a category-specific verifier examines
shape semantics.

## `--hip-infer-shapes`

`--hip-infer-shapes` is a module pass that runs after ONNX-to-HIP conversion and before One-Shot Bufferize. It is restricted to HIP dialect operations.

### Phase 1: refine HIP DPS results

The pass collects operations in post-order so producers inside nested regions are considered before enclosing users. An operation is eligible only when all of the following hold:

- it is in the HIP dialect and implements `ReifyRankedShapedTypeOpInterface`;
- it has at least one ranked tensor result;
- reification succeeds and returns one shape vector per operation result.

For each eligible operation:

1. call `reifyResultShapes`;
2. preserve existing static dimensions;
3. replace a dynamic dimension when reification produces a constant;
4. for a result paired with a DPS init, rebuild its single-use `tensor.empty` producer with the refined type;
5. update the operation result type;
6. insert `tensor.cast` barriers for non-DPS uses that still expect the original type.

Non-tensor and unranked results are skipped individually. If a DPS destination cannot be safely rebuilt—such as a function argument, shared producer, or unsupported init-defining operation—that result remains unchanged. Reification failure or a result-count mismatch skips the operation. A per-result rank mismatch violates the reification interface contract and is treated as an implementation error.

The pass inserts one cast per refined result when needed. Pre-existing non-DPS-init uses are redirected through it, except a `func.return` whose declared result type was already synchronized to the refined type. This preserves the old type at unrelated consumer boundaries while the refined result and rebuilt destination keep the DPS equality invariant. Standard tensor folding can still recover static dimensions through the compatible cast.

### Phase 2: synchronize loop signatures

When Phase 1 refines a value feeding `hip.loop`, the pass propagates types to a fixed point:

- synchronize loop result types;
- synchronize the outlined body function's loop-carried argument and result types;
- rerun local body refinement when narrowed entry types expose additional static information.

The synchronization copies each loop init type to the corresponding loop result, outlined-body carried argument, and outlined-body function result. When the body signature changes, the pass re-runs local HIP result refinement so the existing terminator operands catch up. Casts are inserted on non-DPS uses of narrowed loop results when consumers still require the old type. This keeps loop operations, body signatures, terminators, and carried values type-consistent without rewriting unrelated function signatures. Phase 2 iterates to a fixed point, with a hard cap guarding against non-monotone implementation bugs.

## Tensor-dimension resolution

`hip-resolve-tensor-dims` runs after shape refinement and the following canonicalize/CSE. It folds `tensor.dim` through reification interfaces and reshape/view chains, including standard `tensor.expand_shape`, `tensor.collapse_shape`, and `tensor.pad`, so allocation-size arithmetic sees root tensor dimensions rather than opaque descriptor edits.

This pass is complementary to `hip-infer-shapes`: shape inference narrows result types; tensor-dim resolution simplifies queries that remain in SSA.

Do not confuse the two dim-resolution surfaces:

- MLIR's `--resolve-shaped-type-result-dims` is the focused LIT surface for testing an operation's reification contract.
- Production compiles use `--hip-resolve-tensor-dims`, which applies upstream reify-driven dim folds and tensor canonicalization to a fixed point.

**Registry contract:** `tensor::registerInferTypeOpInterfaceExternalModels` must be registered in the dialect registry. Without it, upstream patterns silently no-op on standard tensor reshape/pad operations and opaque dim queries survive into bufferization, increasing pool fragmentation.

## Pipeline placement

### Reshape shape entries

ONNX Reshape conversion keeps host shape scalars through static tensor casts,
rank-zero/rank-one views, and constant slice/insert paths. The lookup is bounded
and stops at unknown producers. Unknown payloads still use synchronized
`hip.readback_scalar` operations.

A tensor dimension is nonnegative. A non-narrowing index cast or a selection
of nonnegative values preserves that fact. Such an entry cannot be the ONNX
`-1` sentinel. Conversion skips inference for that entry and omits the element
count and division when no entry can be `-1`. This keeps unnecessary late shape
dependencies out of allocation sizes. Real `-1` requests, zero substitution,
and `allowzero` retain their existing rules. Exact broadcast sizing is unchanged.

Compute conversion can visit Reshape before its shape producers. A final walk
forwards host-built shape entries and removes impossible `-1` comparisons.
The normal canonicalizer then removes unused inference arithmetic. This walk
does not run general folding or change the pipeline order.

The changes run within `convert-onnx-to-hip`; they do not add a pass.

### Pass order

[pipeline_pass_menu.md](../pipeline_pass_menu.md) documents pass names and extension anchors. The order source of truth is `lib/Dialect/Transforms/Pipelines.cpp`; the relevant segment is:

```text
simplify-onnx
→ hip-add-context-arg
→ onnx-loop-outline
→ onnx-if-outline
→ hip-infer-loop-body-shapes
→ convert-onnx-to-hip
→ hip-infer-shapes
→ canonicalize
→ CSE
→ func.func(hip-split-duplicate-dps-inits)
→ func.func(hip-resolve-tensor-dims)
→ one-shot-bufferize
```

`--hip-infer-shapes` runs before bufferization so static refinements affect destination construction, memref allocation sizes, and downstream pool planning.

Canonicalization and CSE run immediately afterward to fold dimensions made static by refinement and to deduplicate independently emitted shape arithmetic. `hip-split-duplicate-dps-inits` then repairs same-op DPS init aliasing before bufferization.

Static refinements propagate into bufferization sizing and downstream pool planning. For graph outputs they may also simplify `hip.alloc_output` shape operands; extents that remain dynamic are represented as `-1` in model metadata and are sized in-graph at runtime. Runtime-dependent counts such as `hip.nonzero` remain dynamic at both levels.

A data-dependent extent that reaches a graph output must additionally be a real SSA value *before* the allocation, because `hip.alloc_output` takes the extent as an operand and ORT rejects an output request whose shape differs from the one it computed for the run. The converter therefore materializes the count with a device scan plus a synchronized `hip.readback_dim` and sizes the DPS init with it; reification then reports that init extent. `onnx.Compress` follows this pattern (scanning its `condition` with `hip.nonzero`), so a padded-input encoder that drops its pad slices reports the kept-slice count rather than the padded capacity. Reporting the upper bound instead is not merely conservative — it is the wrong output shape.

## Pre-conversion loop-body rank inference

`--hip-infer-loop-body-shapes` is a narrow pre-conversion backstop. It runs after loop outlining and before ONNX-to-HIP conversion to establish rank for unranked loop-carried values that would otherwise block conversion; it is not the general HIP shape-inference mechanism.

For each outlined `hip.loop` body it:

1. seeds carried block arguments from `v_init`;
2. forward-infers supported unranked ONNX results (currently Concat);
3. applies the ONNX loop-carried type contract as a fallback;
4. reconciles the body function signature with the terminator.

The loop-carried fallback is authoritative: if a carried body output is still unranked, the pass assigns the corresponding ranked `v_init` type as required by the ONNX Loop contract. Interior unranked values without a forward rule remain unchanged. Post-conversion `--hip-infer-shapes` performs static-dimension narrowing within the established rank.

## Adding or changing a HIP DPS operation

1. **Choose the contract row** in [Shape-contract mechanisms](#shape-contract-mechanisms).
2. **Choose a named DPS behavior family**; add a family only when no existing
   reify/infer combination fits.
3. **Add a shared helper** in `HipShapeUtils` only when no existing category fits.
4. **Add a member reification implementation** in
   `HipReifyResultShapesImpl.cpp` only for a manual-reify family.
5. **Return one shape vector per tensor result** and preserve dynamic extents honestly when no pre-execution SSA value exists.
6. **Use the inferred-type builder** in the converter when the generated or custom InferType contract supports it.
7. **Add a verifier** only for non-trivial static contracts not already closed by DPS typing.
8. **Add LIT coverage** for tensor and memref modes, static and dynamic shapes, fallback behavior, multi-result behavior, and negative cases as applicable.

Do not create a new shape interface when the standard InferType/Reify interfaces can express the contract.

## Tests

Primary regression coverage:

| File | Contract |
|---|---|
| `test/lit/Dialect/hip-infer-shapes.mlir` | Module-level static-dimension refinement and cast barriers |
| `test/lit/Dialect/hip-infer-loop-body-shapes.mlir` | Pre-conversion rank establishment |
| `test/lit/Dialect/hip-dps-op-interface.mlir` | Shared `HipDpsOpInterface` reification |
| `test/lit/Dialect/hip-broadcast-reify-shapes.mlir` | Shared broadcast reification and rank-zero success |
| `test/lit/Conversion/onnx-to-hip/test_broadcast_destination_consistency.mlir` | Equal allocation and reified extents through collapse and bufferization |
| `test/lit/Conversion/onnx-to-hip/test_broadcast_folded_failure_atomicity.mlir` | No partial destination or variadic chain after a folded shape conflict |
| `test/lit/Dialect/hip-matmul-reify-shapes.mlir` | Per-op reification through `--resolve-shaped-type-result-dims` |
| `test/lit/Dialect/hip-matmul-shape-verifier.mlir` | Static MatMul shape validation |
| `test/lit/Dialect/hip-loop-verifier.mlir` | Loop-carried type contract |
| `test/lit/Dialect/hip-resolve-tensor-dims.mlir` | Production pre-bufferization dim folding |

Complex operations may use dedicated files; common shape categories should extend the consolidated infer-shapes coverage.

## Current limitations

- ONNX MatMul rank-1 operands require promotion to rank 2 before constructing `hip.matmul`; the runtime and current verifier require rank at least 2.
- Converter migration to inferred-type builders is incremental; explicit result-type builders remain supported.
- A future multi-result operation that needs custom `InferTypeOpInterface` logic may require a dedicated result-type inference implementation file.
- Runtime-dependent extents without pre-execution SSA remain dynamic.
- Phase 1 only rebuilds supported single-use destination producers; shared or externally supplied destinations are left unchanged.
- The pre-conversion loop-body pass has intentionally narrow ONNX coverage and is not a whole-graph solver.
