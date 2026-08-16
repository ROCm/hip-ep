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

## Symbolic metadata transport

ONNX `dim_param` names are authoritative frontend identities, but ranked MLIR
tensor types retain only static integers and dynamic markers. MorphiZen
therefore records each named tensor's symbolic dimensions in the deterministic
`HSDI1` metadata format. The ORT bridge reserves the metadata key, MLIR model
serialization projects it temporarily onto the module, and loading imports it
back into model metadata.

The normative wire grammar, canonical ordering, bounds, empty/missing
semantics, and versioning rule are defined beside the public encoder and
decoder in
[`symbolic_dims.hpp`](../../morphizen/morphizen-graph/include/morphizen/symbolic_dims.hpp).
The encoder emits only that canonical form, and the decoder rejects malformed,
non-canonical, or unknown-version input.

The transport alone does not change destination construction or HIP shape
semantics. A later conversion layer may consume the metadata conservatively; a
missing identity always means no proof.

Compiled artifacts are keyed by the source or canonical compiler graph,
initializer bytes for persistent artifacts, canonical symbolic metadata, and
the resolved compiler contract. Process-local initializer addresses are
normalized out of graph identity. Prebuilt artifacts without finalized
initializer identity are ignored, and restored cache metadata must match the
already-finalized key. A prebuilt mismatch is a cache miss and triggers fresh
compilation; an EPContext mismatch is reported to ORT because its source graph
is unavailable for recompilation.

External initializer paths are resolved against the model directory and
lexically normalized without resolving symlinks. For persistent artifacts, the
compiler verifies initializer bytes immediately before and after consuming the
compiler graph and rejects a mutation. This closes the hash-to-compile window;
it does not make later changes to a streaming source file safe after
compilation.

## Shape knowledge and symbolic scope

The foundation represents compile-time extents in `RankedTensorType` and
runtime extents as ordinary index SSA carried by `OpFoldResult`. It does not
maintain a persistent inter-operation constraint set for facts such as “these
two dynamic dimensions are equal.” Consequently, type-level verification treats
dynamic extents as compatible unknowns, while runtime contracts check
relationships that are not statically visible.

This does not preclude symbolic reasoning. A future analysis may use MLIR's
`ValueBoundsOpInterface` and external models over the same dimension SSA without
changing the infer/reify interfaces defined here. Prefer that standard mechanism
before introducing another feature-specific symbolic analysis. Frontend payload
provenance, such as reconstructing a shape tensor's values, is a separate
problem from proving affine equality between dimensions.

ONNX also carries authoritative dimension-variable identity (`dim_param`).
MorphiZen preserves that frontend fact as compile-time-only module metadata.
Immediately before ONNX compute conversion, a bounded operation-local plan may
reuse one operand extent when every dynamic non-unit broadcast contributor has
the same eligible non-empty identity. Missing, different, malformed, unnamed,
multi-bound, nested-subgraph, and post-rewrite values retain the exact runtime
broadcast expression. Pre-existing operation-local plan attributes are rejected
rather than trusted. The metadata and generated plan are consumed during ONNX
conversion and do not survive into HIP IR; `ReifyRankedShapedTypeOpInterface`
and `--hip-infer-shapes` remain frontend-neutral and callback-free.

MorphiZen's finalized cache key includes the immutable source model (or
canonical in-memory compiler graph), persistent-artifact initializer bytes,
canonical symbolic metadata, resolved compiler contract, encoding version, and
resolver-policy version in domain-separated SHA-256 fields. Process-local
initializer addresses are normalized out of graph identity. A prebuilt artifact
without finalized initializer identity is ignored, and a restored cache context
must contain the same finalized key; cache metadata cannot replace it.

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
`Hip_DpsOp_Broadcast` and `Hip_DpsOp_Reduction` also generate fixed verifier
bodies that compose
`verifyDpsComputeOp`, so tensor/memref uniformity and tensor result-to-init type
parity are checked before semantic shape rules. Broadcast uses the same pure
NumPy rule as reification and destination construction. Reduction requires a
structurally-proven constant axes source in tensor and memref form. Runtime
axes are rejected regardless of result type.

The `SameShape`, `Semantic`, `Payload`, and `OutsAuthoritative` wrappers name
reviewed contract categories without requiring the exhaustive inventory audit
to be enabled while feature migrations are still in progress.

Multi-result and specialized operations may keep inferred result construction disabled when a single generated body cannot describe all results.

## Shape-contract mechanisms

Choose the smallest mechanism that matches the operation's semantics:

| Shape contract | Mechanism |
|---|---|
| Result shape equals DPS init shape, including most multi-result DPS ops | Shared `HipDpsOpInterface` default |
| Result shape equals a named input | `reifyElementwiseSameShape` or a small dedicated thunk |
| NumPy-style broadcast | `Hip_DpsOp_Broadcast`, `reifyBroadcastResultShape`, and the shared converter bridge |
| Reduction with constant axes/keepdims | `Hip_DpsOp_Reduction` and reduction helpers |
| Permutation | `reifyTransposeByPerm` |
| Gather/GatherND/GatherElements | Gather-specific helpers or thunks |
| OneHot, Compress, TopK | Dedicated reification thunks |
| Pad, Tile, Expand, Slice, Range | Fold-or-bail helpers with fallback to DPS-init shape |
| MatMul/Gemm/MatMulNBits | Dedicated shape logic based on operand dimensions and attributes |
| Attention or normalization with multiple destinations | One shape vector per DPS init unless an op supplies a dedicated thunk |
| Forward Conv (rank-3 converter/rank-4 HIP op) | Shared signed-floor spatial-window formula used by converter, reification, and verifier |
| Rank-4 NCHW ConvTranspose | Shared ONNX formula used by converter, reification, and verifier |
| CausalConvWithState | Runtime-supported 1D output/state formulas from input and depthwise kernel |
| Resize | DPS-init shape, with semantic validity handled by conversion |
| Runtime-dependent count, such as NonZero or Compress | DPS-init shape; unresolved dimensions remain dynamic |

Shared declarations live in `HipShapeUtils.h`; common implementation lives in
`HipShapeUtils.cpp`, with focused category translation units introduced by the
stack layer that first consumes each family. This foundation includes
matmul/Gemm, reduction, gather, and shape-operation helpers; later family PRs
add attention and convolution/pooling implementations. Operations that select
a manual-reification family define their member functions in
`HipReifyResultShapesImpl.cpp`.

Frontend-neutral destination construction lives in
`hip/Conversion/HipConversionUtils.h`: result-shape compatibility, reified
`tensor.empty` creation, broadcast destination construction, and HIP context
lookup. `OnnxToHipUtils` retains only ONNX import semantics and wrappers, so a
future frontend can target HIP without depending on the ONNX conversion layer.

Pure descriptor transformations such as Reshape, Squeeze, and Unsqueeze
generally lower to standard tensor operations rather than HIP DPS compute
operations. Their result-extent inference and dim folding use MLIR's standard
tensor interfaces and external models, not a second HIP-specific contract.
`ReifyRankedShapedTypeOpInterface` exposes shaped-result extents, and
ValueBounds can reason about dimensions and scalar/index bounds; neither
directly transports the integer payload of a rank-1 ONNX shape tensor. Typed
ONNX operations and shape helpers could replace name-based transfer dispatch,
while Shape-dialect integration would require representing the payload as
explicit shape values.

After pre-lowering ONNX rewrites reach their fixed point and before the first
`onnx.Constant`-to-`hip.constant` carrier sweep, one function-level MLIR sparse
forward dataflow solve computes Reshape target-payload provenance. Its monotonic
lattice carries canonical tensor dimensions and an optional complete host
payload through Shape, constant Gather/Slice, axis-zero Concat, Identity,
Unsqueeze, scalar/vector Reshape, constants, and proven `tensor.from_elements`
chains. Add, MatMul, and Cast transfer canonical dimension roots without
claiming payload provenance. Joins keep a dimension fact only when incoming
roots agree and keep a payload only when the complete vectors agree; there is
no `hip.loop` dimension propagation.

Literal target tensors do not require payload provenance. Conversion reads a
dense `hip.constant` carrier directly, validates the ONNX Reshape rules, and
materializes the target entries without consulting an externalized global or
performing synchronized readback. Provenance is reserved for runtime-derived
shape programs: a complete proof rebuilds their target from constants,
canonical `tensor.dim` values, and proven host scalar SSA. Disagreement,
unknown producers, partially-known payloads, and device-produced payloads leave
the operand untouched. If no earlier shape-independent or static rewrite
applies, the dynamic runtime-shaped fallback then uses synchronized readback.

For `allowzero=1`, the provenance path accepts a target containing `-1` only
when every other entry is proven strictly positive. A merely nonnegative entry
could be zero, which ONNX forbids alongside `-1`, so it retains the unknown
fallback. A target proven nonnegative throughout may bypass generic `-1`
inference.

Materialization stamps pass-internal markers for the dynamic runtime-shaped
Reshape fallback. Before that fallback's first mutation, it revalidates the
marker dependencies it consumes, target rank and length, host-safe scalar
structure, nonnegative/no-`-1` claims, and mapped input-dimension identities.
Earlier shape-independent or static Reshape rewrites do not consume marker
claims and may bypass their validation. The fallback does not rerun the
dataflow solver, so the markers are a handoff across this fixed pipeline
boundary rather than durable proof after arbitrary IR mutation. The analysis is
skipped for functions without an eligible dynamic Reshape.

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
materialize index SSA only after validation succeeds.

A reifier must validate every precondition before touching the builder.
Failure must leave the IR unchanged, including when the valid result shape is
rank zero; `FailureOr` distinguishes that empty success from failure.
Conversion-side destination builders in `HipConversionUtils.cpp` validate
through the same pure shape rule and check imported static result metadata
before creating `tensor.empty`. Exact dynamic destination sizing is activated
together with frontend identity proofs that fold redundant merge SSA. Imported
and inferred extents follow standard shaped-type compatibility: a dynamic extent
on either side is compatible, while unequal static extents are contradictions.
Only after every fallible static check succeeds may reification emit dimension
SSA and destination builders create the destination.

Common DPS verification is similarly centralized in `verifyDpsComputeOp`. It
checks ranked tensor/memref uniformity, destination count, result count, and
tensor result/init type equality before a category-specific verifier examines
shape semantics.

Broadcast dimensions are right-aligned. Static 1 yields to the other side;
equal non-unit static dimensions agree; dynamic/static-non-1 tightens to the
static extent under the ONNX input contract. Two dynamic dimensions emit:

```mlir
%lhs_is_one = arith.cmpi eq, %lhs_dim, %c1 : index
%extent = arith.select %lhs_is_one, %rhs_dim, %lhs_dim : index
```

Do not replace this with an integer maximum: broadcasting dimensions 0 and 1
produces 0, not 1. Variadic Max/Min share one
`lowerVariadicBroadcastChain` helper that derives every pairwise intermediate
type from this shared broadcast shape.

Forward Conv uses a validated signed-floor spatial-window formula:
`floor((input + pads - effectiveKernel) / stride) + 1`. Signed floor and
intermediate arithmetic are required because the numerator can be negative
even when the final extent is the valid value zero. Dilation contributes through
`effectiveKernel = (kernel - 1) * dilation + 1`.

Conv conversion applies the shared rule to the original rank-3 NCL shape before
its NC1L expansion; `hip.conv` itself uses the rank-4 form. An omitted ONNX Conv
`kernel_shape` is derived from static weight spatial dimensions. ConvTranspose
likewise uses one rule for destination construction, reification, and static
verification.

Dynamic spatial-window arithmetic uses signed i128 from the input-dimension
cast through effective-kernel multiplication, padded input, signed numerator,
floor division, and raw output. This width covers every combination of
nonnegative i64/index extents and i64 attributes without intermediate
wraparound and stays equivalent to the static APInt rule. Before narrowing,
each final extent must be in `[0, INT64_MAX]`; invalid extents select zero so
destination allocation remains bounded. Runtime rejection of those invalid
dynamic combinations belongs to the later runtime-hardening layer.

CausalConvWithState uses the backend's implemented 1D contract. For input
`[B,C,L]` and depthwise weight `[C,1,K]`, output is `[B,C,L]` and
`present_state` is `[B,C,K-1]`. Optional `past_state` must have that same state
shape, but it is a validator rather than an extent source: when it is absent or
more dynamic than the weight, the state length still comes from `weight.K-1`.
The mixed helper emits subtraction only after its pure helper has validated
rank, depthwise layout, channel agreement, optional bias/state, and the current
runtime restriction `ndim=1`.

Reductions resolve to one internal out-to-in dimension map, consumed by both
`inferReductionShape` (static extents) and `reifyReductionResultShape` (mixed
extents used for destination construction and reification). The mapping matters
for `keepdims = 0`: reducing axes `[1, 2]` of a rank-4 input maps output
dimension 1 to input dimension 3, so a positional input-dimension copy is
incorrect.

`resolveReductionAxes` decides once whether axes are usable and returns
`std::nullopt` when they are only known at runtime. Constant opset-13+ axes,
attribute axes, absent axes, and empty axes use the shared semantic map.
Runtime axes always produce a conversion error before creating `tensor.empty`.
Negative axes are normalized and sorted, then must form one contiguous span.
The runtime kernel flattens exactly that span: `reduce_size` is
`input_elements / output_elements`, and `inner_size` is the product of input
dimensions after the span's final axis. Non-contiguous sets such as `[0, 2]`
cannot be represented and are rejected.

Conversion materializes the normalized axes as an inline constant. Bufferization
preserves it as a constant `memref.global`. The HIP op also carries a durable
`normalized_axes` attribute; tensor and memref verification compare that
attribute exactly against the structural constant source, so a forged marker is
invalid. Reification uses the same checked pair. Lowering consumes the
already-verified normalized attribute after dialect conversion has rewritten
the global source, and never infers axes by comparing input and output extents,
which would be ambiguous for equal extents such as reducing axis 0 versus axis
1 of a `4x4` tensor.

MatMul and Gemm accept a dynamic contraction K as unknown-compatible. Static
equal K remains valid, while static unequal K is rejected by the pure shape
rule before reification or conversion emits IR. HIP-to-LLVM passes both runtime
extents independently (`MatMul`: A[-1]/B[-2]; `Gemm`: transpose-aware A/B K)
to the wrappers. The wrappers compare them before descriptor creation, cache
lookup, or dispatch and key caches only with the equal value.

Runtime rejection is failure-contained. A K mismatch, negative/overflowing
dimension, dynamically concealed partial batch broadcast, invalid output
pointer, or checked output element/byte overflow records the shared recoverable
error flag and skips BLAS work. When the exact nonempty output byte count is
known and storage is valid, the wrapper queues an exact zero-fill so downstream
consumers never observe uninitialized output. A zero-element output may have
null storage and dispatches no BLAS work.

### MatMul strided-batch representability

The hipBLASLt MatMul lowering takes the batch count from the reified output
shape and carries independent A/B batch strides, so either whole matrix may
broadcast across the other's batches. One constant stride per operand can
express exactly two layouts: stride 0 reuses a single matrix across every output
batch, and a stride of the matrix size walks one matrix per output batch. An
operand's matrix count must therefore be either 1 or the output's.

A partial per-axis broadcast falls strictly between the two — batch `[2, 1]`
against an output batch of `[2, 3]` holds 2 matrices where the output needs 6.
`verifyStridedBatchMatmul` rejects partial layouts visible in the static types.
Dynamic extents can conceal the same layout, including across multiple batch
axes, so those layouts are accepted statically and validated at runtime.
Before flattening, lowering right-aligns A/B batch axes (using 1 for implicit
leading axes) and requires per axis `A == B || A == 1 || B == 1`. It computes
the exact reification broadcast choice `select(A == 1, B, A)` and requires the
runtime output descriptor extent to equal it. The axis predicates are ANDed
into `batch_axes_valid`.

Lowering then multiplies every leading extent to form the output and
per-operand matrix counts. It selects matrix-size stride only when an operand
count equals the output count, otherwise stride zero. The runtime wrapper first
requires `batch_axes_valid`, then requires each count to be exactly 1 or the
output count, all before descriptor/cache creation or BLAS dispatch. The
per-axis bit is necessary because incompatible batches such as `[2,3]` and
`[3,2]` have the same flattened matrix count.

`wrap_hipblasLtMatmul` carries `batch_axes_valid`, both contraction extents,
both operand batch counts, and both strides. B's matrix stride is formed from
B's own contraction extent, not A's. A false axis bit or invalid matrix count
records the shared recoverable error before any BLAS work and zeroes a known
nonempty output. HIP LLVM-IR and native model artifacts compiled against the
former 11-argument MatMul ABI or the previous one-K Gemm ABI must be
invalidated.

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
| `test/lit/Dialect/hip-broadcast-reify-shapes.mlir` | Broadcast dynamic SSA, zero extents, and rank-zero success |
| `test/lit/Dialect/hip-broadcast-shape-verifier.mlir` | Generated NumPy broadcast shape verification |
| `test/lit/Dialect/hip-reduction-shape-verifier.mlir` | Generated reduction shape and axes verification |
| `test/lit/Conversion/onnx-to-hip/test_reshape_shape_provenance.mlir` | Proven host Reshape shapes, dataflow joins/shared producers, unknown-payload fallback, and `-1` handling |
| `test/lit/Dialect/hip-matmul-reify-shapes.mlir` | Per-op reification through `--resolve-shaped-type-result-dims` |
| `test/lit/Dialect/hip-matmul-shape-verifier.mlir` | Static MatMul shape validation |
| `test/lit/Conversion/onnx-to-hip/test_reduce_sum.mlir` | Reduction destinations, including the non-positional `keepdims = 0` dimension mapping |
| `test/lit/Conversion/onnx-to-hip/test_reduce_runtime_axes_invalid.mlir` | Conversion-time rejection of all runtime axes and non-contiguous constant axes |
| `test/lit/Dialect/hip-loop-verifier.mlir` | Loop-carried type contract |
| `test/lit/Dialect/hip-resolve-tensor-dims.mlir` | Production pre-bufferization dim folding |

Complex operations may use dedicated files; common shape categories should extend the consolidated infer-shapes coverage.

## Current limitations

- ONNX MatMul rank-1 operands require promotion to rank 2 before constructing `hip.matmul`; the runtime and current verifier require rank at least 2.
- Runtime reduction axes and non-contiguous constant axis sets are unsupported by the current flattened-span kernel ABI.
- Converter migration to inferred-type builders is incremental; explicit result-type builders remain supported.
- A future multi-result operation that needs custom `InferTypeOpInterface` logic may require a dedicated result-type inference implementation file.
- Runtime-dependent extents without pre-execution SSA remain dynamic.
- Phase 1 only rebuilds supported single-use destination producers; shared or externally supplied destinations are left unchanged.
- The pre-conversion loop-body pass has intentionally narrow ONNX coverage and is not a whole-graph solver.
