<!--
Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
Licensed under the MIT License.
-->
# HIP dialect shape inference

**Date:** 2026-07-28
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

`Hip_DpsOp` centralizes the interface boilerplate used by HIP compute operations.

| Parameter | Default | Purpose |
|---|---|---|
| `outsAccessor` | `"Output"` | ODS accessor used by generated result-type inference |
| `autoReify` | `1` | Emit a dispatcher to the shared `HipDpsOpInterface` reification body |
| `autoInfer` | `0` | Emit `inferReturnTypes` using the DPS init tensor type |
| `declareInfer` | `0` | Add `InferTypeOpInterface`; `autoInfer=1` requires it, while `declareInfer=1, autoInfer=0` requires a hand-written `inferReturnTypes` implementation |
| `customReifyBody` | empty | Provide an inline custom body for parameterized DPS sub-bases |
| `sameShapeSource` | empty | Record the ODS accessor stem of a same-shape contract's named semantic source |
| `customVerifyBody` | empty | Provide an inline verifier body for a parameterized DPS sub-base |

`Hip_DpsOp_SameShape`, `Hip_DpsOp_Broadcast`, and `Hip_DpsOp_Reduction` use
these hooks to share named-source, broadcast, and reduction reification across
operation families without per-op C++ implementations. The same-shape base
accepts independent source and output accessor stems (`input`/`output`,
`x`/`y`, `data`/`output`), infers the result type from the DPS init, and emits a
verifier that checks the structural DPS contract before the named source.
Leaves with existing verifiers call the same shared verifier helper after
preserving their structural and op-specific checks.

The broadcast and reduction bases also generate their verifier bodies. Both
compose `verifyDpsComputeOp`, so tensor/memref uniformity and tensor
result-to-init type parity are checked before semantic shape rules. Broadcast
uses the same pure NumPy rule as reification and destination construction.
Reduction validates exact output shape only when the axes payload is
compile-time-known; runtime axes validate their rank/type and the keepdims/noop
attributes, then retain the outs-authoritative fallback.

Multi-result and specialized operations may keep inferred result construction disabled when a single generated body cannot describe all results.

## Shape-contract mechanisms

Choose the smallest mechanism that matches the operation's semantics:

| Shape contract | Mechanism |
|---|---|
| Result shape equals DPS init shape, including most multi-result DPS ops | Shared `HipDpsOpInterface` default |
| Result shape equals a named input | `Hip_DpsOp_SameShape`, `reifyElementwiseSameShape`, and `verifySameShapeDpsOp` |
| NumPy-style broadcast | `Hip_DpsOp_Broadcast`, `reifyBroadcastResultShape`, and the shared converter bridge |
| Reduction with constant axes/keepdims | `Hip_DpsOp_Reduction` and reduction helpers |
| Permutation | Shared pure `inferTransposeShape` plus `reifyTransposeByPerm` and verifier |
| Gather/GatherND | Shared pure Gather/GatherND helpers plus reify/verifier thunks; dynamic GatherND tuple width falls back to outs |
| GatherElements | `Hip_DpsOp_SameShape` with `indices` as its named source |
| GatherBlockQuantized | Shared logical-dequantized Gather rule; packed int4 expands the surviving quantize axis |
| OneHot, Compress, TopK | Dedicated reification thunks |
| Pad | Exact affine input-dimension arithmetic for constant/stamped pads; runtime pads use converter readback and reify from outs |
| Slice | Exact constant/static-bound rule; dynamic sliced-axis clamps and runtime parameters preserve capacity/outs policy |
| Expand, Range | Fold-or-bail helpers with fallback to DPS-init shape |
| Tile | Constant repeats use exact mixed shape arithmetic; runtime repeats use one bulk synchronized readback and reify from that exact init |
| MatMul/Gemm | Shared operand-based helpers used by conversion and reification |
| MatMulNBits | Shared pure rule based on A and N, reused by reify, conversion, and verification |
| Forward Conv (rank-3 converter/rank-4 HIP op) | Shared signed-floor spatial-window formula used by converter, reification, and verifier |
| Window Pool (spatial rank 1..3) | Shared signed floor/ceil spatial-window formula; optional indices reuse the values shape |
| Rank-4 NCHW ConvTranspose | Shared ONNX formula used by converter, reification, and verifier |
| LayerNormalization | Y equals input; Mean/InvStdDev use keepdims reduction shape over `[axis, rank)` |
| LinearAttention | Shared output/state formula used by converter, reification, and verifier |
| CausalConvWithState | Runtime-supported 1D output/state formulas from input and depthwise kernel |
| SkipSimplifiedLayerNormalization | Output and optional residual sum equal input; training stats rejected |
| GroupQueryAttention | Semantic query/head mapping plus `max(past capacity, total_seq_len)` or logical length alone without past; verifier covers structural/rank/head and statically decidable cache compatibility without reading payload lengths |
| MultiHeadAttention | Default rank-3 fp16 Q/K/V result equals query; optional/cache forms route to GQA or are rejected |
| Resize | N/C from input plus static spatial extents from the imported output template |
| Runtime-dependent count, such as NonZero | DPS-init shape; unresolved dimensions remain dynamic |

The public helper API lives in `HipShapeUtils.h`; category implementations live
in the `HipShapeUtils*.cpp` files. Operations that set `autoReify=0` and are not
covered by a parameterized sub-base define their member functions in
`HipReifyResultShapesImpl.cpp`.

The same-shape inventory guard additionally requires every `same_shape` op to
use `Hip_DpsOp_SameShape`. Converter destinations that need to materialize this
contract use
`createSameShapeEmptyTensor`, so dynamic extents and static compatibility come
from the same named source as reification and verification.

Every HIP DPS op has one reviewed contract category in
[hip-shape-contract-inventory.md](hip-shape-contract-inventory.md). TableGen's
`hipShapeContract` field is source-level inventory metadata (not emitted IR),
and `hip-shape-contract-inventory.mlir` fails when the document and op
definitions drift or a new DPS op remains unclassified.

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

After pre-lowering ONNX rewrites reach their fixed point and before constant
externalization, one function-level MLIR sparse forward dataflow solve computes
Reshape target-payload provenance. Its monotonic lattice carries canonical
tensor dimensions and an optional complete host payload through Shape, constant
Gather/Slice, axis-zero Concat, Identity, Unsqueeze, scalar/vector Reshape,
constants, and proven `tensor.from_elements` chains. Add, MatMul, and Cast
transfer canonical dimension roots without claiming payload provenance. Joins
keep a dimension fact only when incoming roots agree and keep a payload only
when the complete vectors agree; there is no `hip.loop` dimension propagation.
A complete proof rebuilds the target from constants, canonical `tensor.dim`
values, and proven host scalar SSA. Disagreement, unknown producers,
partially-known payloads, and device-produced payloads leave the operand
untouched. If no earlier shape-independent or static rewrite applies, the
dynamic runtime-shaped fallback then uses synchronized readback.

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

For a single-result DPS operation with `declareInfer=1` and `autoInfer=1`, ODS provides an inferred-type `Op::create` overload. The generated `inferReturnTypes` reads the typed DPS init and emits:

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

Converter destination construction and operation reification must not
independently implement the same shape category. Broadcast, Gemm, MatMul,
reductions, Conv, and Pool use helpers in `HipShapeUtils` that return
`FailureOr<SmallVector<OpFoldResult>>`. The `FailureOr` is required because a
valid rank-zero result has a successful empty shape.

`HipShapeUtils` splits each category into a pure `infer*` function of static
shapes and a `reify*` function that may materialize index SSA. Every `reify*`
helper validates its preconditions through the matching `infer*` function
**before** it touches the builder, so a failure never leaves stray dimension ops
behind. Both the pattern-rewrite contract and
`ReifyRankedShapedTypeOpInterface` require the IR to be unchanged when a
rewrite or reification reports failure; upstream's
`ResolveShapedTypeResultDims` erases such stray ops explicitly, and this
codebase avoids creating them in the first place. Emitting IR is therefore the
last thing a helper does.

The same split gives targeted operation verifiers a shape rule to check against.
MatMul and Gemm use `verifyHipOpShape` with their `infer*` helper, so their
`outs` shape, converter destination, and `reifyResultShapes` are all held to one
shape function. Broadcast and reduction bases likewise generate verifiers from
their shared rules. Broadcast checks every statically decidable NumPy extent;
reduction checks the exact output shape when axes are compile-time-known and
retains the outs-authoritative fallback for runtime axes:

```c++
LogicalResult MatmulOp::verify() {
  // ... DPS contract first ...
  return verifyHipOpShape(*this, [&] {
    return inferMatmulShape(aShape, bShape,
                            [&] { return this->emitOpError(); });
  });
}
```

Only the `infer*` helpers are public. The dimension mappings and static
broadcast folds they are built from stay internal to the category
`HipShapeUtils*.cpp` implementation files, so the header exposes shape *rules*
rather than the machinery behind them.

Broadcast dimensions are right-aligned. Static 1 yields to the other side;
equal non-unit static dimensions agree; dynamic/static-non-1 tightens to the
static extent under the ONNX input contract. Two dynamic dimensions emit:

```mlir
%lhs_is_one = arith.cmpi eq, %lhs_dim, %c1 : index
%extent = arith.select %lhs_is_one, %rhs_dim, %lhs_dim : index
```

Do not replace this with an integer maximum: broadcasting dimensions 0 and 1
produces 0, not 1.

ONNX conversion keeps the imported ranked result type and uses the shared
`OpFoldResult`s only to populate `tensor.empty` dynamic-size operands. When a
reified dimension is constant but the imported dimension is dynamic, the
converter materializes a constant index size. `--hip-infer-shapes` remains the
single owner of later type narrowing, destination rebuilding, and cast
barriers.

Variadic Max/Min share one `lowerVariadicBroadcastChain` helper that derives
every pairwise intermediate type from the shared broadcast shape. Gemm derives
M/N from A/B with transpose-aware indices and validates optional C without using
C as an extent source. MatMul broadcasts only the leading batch slices, then
appends M from A[-2] and N from B[-1].

GatherBlockQuantized applies Gather to the logical dequantized data shape, not
the physical byte-storage shape. For the HIP op's byte-packed 4-bit storage,
`logical_data[quantize_axis] = 2 * data[quantize_axis]`; all other extents are
unchanged. The result is
`logical_data[:gather_axis] ++ indices.shape ++
logical_data[gather_axis+1:]`. If Gather removes the quantize axis itself, no
result extent is doubled. The shared helper also validates the statically known
block grid (`scales[quantize_axis] =
ceil(logical_data[quantize_axis] / block_size)`), non-quantized extents,
optional zero-point packing, axes, ranks, and runtime-supported attributes.
Converter destination construction, op reification, and static verification all
call this rule. The runtime receives the physical data descriptor and expands
its quantize-axis extent before launching the logical-element kernel.

Forward Conv and Pool share one validated spatial-window primitive:
`floor((input + pads - effectiveKernel) / stride) + 1`; Pool selects signed
ceil division when `ceil_mode = 1`. Signed
floor/ceil operations are required because the numerator can be negative even
when the final extent is the valid value zero. Conv conversion applies the rule
to the original rank-3 NCL shape before its NC1L expansion; `hip.conv` itself
uses the rank-4 form. An omitted ONNX Conv `kernel_shape` is derived from static
weight spatial dimensions. Pool values and optional MaxPool indices are
allocated and reified from the same mixed shape vector.

Reductions resolve to one internal out-to-in dimension map, consumed by both
`inferReductionShape` (static extents, used for destination types) and
`reifyReductionResultShape` (mixed extents, used for destination construction
and `reifyResultShapes`). The mapping matters for `keepdims = 0`, where dropping
reduced axes makes the output dimension order non-positional in the input:
reducing axes `[1, 2]` of a rank-4 input maps output dimension 1 to input
dimension 3. A positional copy from the input is correct only when no reduced
axis precedes a kept one.

Whether the axes are usable at all is decided once, by
`resolveReductionAxes`, which returns `std::nullopt` when they are only known at
runtime. Both the destination and reification key off that single answer: the
converter falls back to a positional copy and reification lifts the `outs`
shape. Deciding it twice is how the two drift apart — gating the converter on
the axes operand *count* while reification gates on the operand being
*constant* leaves opset-13+ constant axes handled inconsistently.

LinearAttention uses one two-result rule in `HipShapeUtils`:
`Dk = query[-1] / Hq`, `Dv = value[-1] / Hkv`, output shape
`[B, T, max(Hq, Hkv) * Dv]`, and recurrent-state shape
`[B, Hkv, Dk, Dv]`. Conversion, `reifyResultShapes`, and the op verifier all
call that rule. In particular, a missing `past_state` does not make key's
positional dimensions authoritative for the state destination.

CausalConvWithState uses the backend's implemented 1D contract. For input
`[B,C,L]` and depthwise weight `[C,1,K]`, output is `[B,C,L]` and
`present_state` is `[B,C,K-1]`. Optional `past_state` must have that same state
shape, but it is a validator rather than an extent source: when it is absent or
more dynamic than the weight, the state length still comes from `weight.K-1`.
The mixed helper emits subtraction only after its pure helper has validated
rank, depthwise layout, channel agreement, optional bias/state, and the current
runtime restriction `ndim=1`.

SkipSimplifiedLayerNormalization's implemented outputs are the normalized
output and optional `input_skip_bias_sum`; both have the input shape. The
runtime flattens every non-final input axis into `num_rows`, requires skip to
match input, and takes `hidden_dim` from rank-1 gamma (and optional bias).
The ONNX schema's optional mean and inverse-standard-deviation outputs are
training outputs and are not implemented by `hip.skip_rms_norm`. Conversion
accepts every inference output arity (one to four schema slots with omitted
stats) but rejects a real stats tensor rather than treating the last present
tensor as the residual output.

GroupQueryAttention has one payload-dependent exception. With past/present
buffer sharing, a matching past extent can be larger than the currently valid
logical prefix; with a growing non-shared cache, the logical prefix can be
larger than the past allocation. Conversion therefore sizes each dynamic
present-cache sequence dimension as the nonnegative index maximum of its
matching past dim 2 and `total_seq_len`. Without past operands, the dynamic
present extent is `total_seq_len` directly. Conversion performs exactly one
synchronized logical-length readback per op when any dynamic present or QK
extent needs it, reusing that value for separate past-key/past-value maxima.
Optional QK always uses the logical extent rather than cache capacity.
Reification keeps the DPS-init fallback for these payload-dependent extents.

Pad resolves compile-time pads and optional axes either from constant operands
or from attributes stamped before constant externalization. `inferPadShape`
validates the parameter lengths and axes without emitting IR; both conversion
and reification then use `reifyPadShape` for the exact affine rule
`output[d] = input[d] + begin[d] + end[d]`. Dynamic input dimensions therefore
produce `tensor.dim` plus a constant offset. Genuinely runtime pads retain the
converter's synchronized payload readback, while dialect reification performs
no payload read and lifts the already-exact destination.

Slice has two distinct shape authorities. When starts, ends, axes, and steps
are compile-time values and every sliced input axis has a static extent,
`inferSliceShape` applies the ONNX clamp rule exactly. Conversion uses that
shape for `tensor.extract_slice` or the native negative-step destination, and
reification returns static sliced extents while forwarding untouched dynamic
input dimensions. If a sliced input extent is dynamic, exact clamping requires
runtime min/max arithmetic; runtime-valued slice parameters require payload
readback as well. Those cases deliberately keep the existing physical
destination capacity equal to the input extents. `wrap_slice` computes a
separate `logical_extent`, launches only the logical slice, and zeroes unused
capacity. Reification therefore lifts the physical destination shape and does
not claim that capacity is the logical ONNX result shape. Changing that policy
would require proving the output-allocation ABI and all downstream consumers
against logical extents, so it is intentionally unsupported here.

Tile's refinement is complete. Constant or stamped repeats share
`inferTileShape`/`reifyTileShape`, including exact multiplication of dynamic
input dimensions. Runtime repeats are read once in bulk by the converter to
build the exact destination; reification lifts that destination and never
duplicates the payload readback.

Resize is semantic rather than payload-dependent at the HIP level. ONNX
`sizes`/`scales` have already been resolved by the importer and are not operands
of `hip.resize`; the runtime recovers each scale from the input/output
descriptors. The shared rule therefore takes N/C from the input and the spatial
extents from the imported output type, requiring every spatial extent to be
static. A dynamic input N/C extent requires a dynamic output template extent;
a static input extent may refine a dynamic template. This deliberately does not
claim dynamic spatial support without carrying the payload.

The default `hip.multi_head_attention` rule matches its implemented runtime
path, not the full Microsoft schema: Q/K/V are separate rank-3 fp16 tensors,
Q/K/V batch and hidden extents agree, K/V sequence extents agree, hidden is
positive and divisible by `num_heads`, `unidirectional` is 0 or 1,
`mask_filter_value` remains its -10000 default, and the sole output is exactly
`[query.B, query.S, query.hidden]`. The runtime honors an explicit `scale`
(with zero retaining the automatic `1/sqrt(head_size)` sentinel). Biases,
masks, packed layouts, past/cache inputs, cache indirection, and present/QK
outputs are rejected before the HIP op is created. The existing decoder
self-attention and rank-4 cross-attention routes to `hip.gqa` run first and
retain their separate contracts.

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
Dynamic extents can conceal the same layout, so the lowering also computes each
operand's runtime matrix count. `wrap_hipblasLtMatmul` dispatches only when each
count is either 1 or the output batch count; otherwise it records an error in
the runtime state's device error flag and skips hipBLASLt. The generated
interface observes that flag after its existing stream synchronization and
returns a recoverable non-zero inference status to ORT.

This preserves ordinary dynamic batched matmul
(`[?, H, M, K] @ [?, H, K, N]`) without treating every non-output matrix count
as whole-matrix broadcast. The stride is 0 only for one matrix and the matrix
size only for one matrix per output batch.

`wrap_hipblasLtMatmul` carries both operand batch counts and both strides.
LLVM-IR artifacts compiled against the previous wrapper ABI must be invalidated.

## `--hip-infer-shapes`

`--hip-infer-shapes` is a module pass that runs after ONNX-to-HIP conversion and before One-Shot Bufferize. It is restricted to HIP dialect operations.

### Phase 1: refine HIP DPS results

Within the supported lexical, single-block HIP/outlined-loop structure, the pass collects operations in post-order so producers are considered before their users. This ordering is not CFG, call-graph, or symbolic inference. An operation is eligible only when all of the following hold:

- it is in the HIP dialect and implements `ReifyRankedShapedTypeOpInterface`;
- it has at least one ranked tensor result;
- reification succeeds and returns one shape vector per operation result.

For each eligible operation:

1. call upstream `mlir::reifyResultShapes`, allowing the standard helper to dispatch the applicable reification interface;
2. preserve existing static dimensions;
3. replace a dynamic dimension when reification produces a constant;
4. for a result paired with a DPS init, rebuild its single-use `tensor.empty` producer with the refined type;
5. update the operation result type;
6. insert `tensor.cast` barriers for non-DPS uses that still expect the original type.

If a DPS destination cannot be safely rebuilt—such as a function argument, shared producer, or unsupported init-defining operation—that result remains unchanged. Reification failure remains a best-effort skip. Successful reification is different: before mutating any result, the pass validates the complete operation-wide response. It must contain exactly one vector per result, every result must be ranked and shaped, every vector must contain exactly the result rank's dimensions, every constant extent must be nonnegative, and a constant must not contradict an existing static extent. Any violation is an op-author/compiler defect: the pass emits a precise operation diagnostic and fails in release builds as well as assertion-enabled builds.

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
2. **Set TableGen parameters** on `Hip_DpsOp` or use an existing DPS sub-base; use `declareInfer=1, autoInfer=0` only when supplying custom InferType logic.
3. **Add a shared helper** in `HipShapeUtils` only when no existing category fits.
4. **Add a member reification implementation** in `HipReifyResultShapesImpl.cpp` only for `autoReify=0` operations not covered by a sub-base.
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
| `test/lit/Dialect/hip-infer-shapes-malformed-reify.mlir` | Release-build failure policy for malformed successful reification |
| `test/lit/Dialect/hip-infer-loop-body-shapes.mlir` | Pre-conversion rank establishment |
| `test/lit/Dialect/hip-dps-op-interface.mlir` | Shared `HipDpsOpInterface` reification |
| `test/lit/Dialect/hip-broadcast-reify-shapes.mlir` | Broadcast dynamic SSA, zero extents, and rank-zero success |
| `test/lit/Dialect/hip-broadcast-shape-verifier.mlir` | Generated NumPy broadcast shape verification |
| `test/lit/Dialect/hip-reduction-shape-verifier.mlir` | Generated reduction shape and axes verification |
| `test/lit/Conversion/onnx-to-hip/test_reshape_shape_provenance.mlir` | Proven host Reshape shapes, dataflow joins/shared producers, unknown-payload fallback, and `-1` handling |
| `test/lit/Dialect/hip-gather-block-quantized-reify-shapes.mlir` | Packed-int4 logical Gather shape reification |
| `test/lit/Dialect/hip-gather-block-quantized-shape-verifier.mlir` | GatherBlockQuantized logical shape and runtime-contract validation |
| `test/lit/Dialect/hip-resize-reify-shapes.mlir` | Dynamic N/C and static spatial Resize reification |
| `test/lit/Dialect/hip-resize-shape-verifier.mlir` | Resize semantic destination validation |
| `test/lit/Dialect/hip-gemm-reify-shapes.mlir` | Transpose-aware Gemm M/N reification |
| `test/lit/Dialect/hip-matmul-reify-shapes.mlir` | Per-op reification through `--resolve-shaped-type-result-dims` |
| `test/lit/Dialect/hip-matmul-shape-verifier.mlir` | Static MatMul shape validation, including accepted dynamic batch layouts and rejected partial per-axis broadcast |
| `test/lit/Dialect/hip-conv-reify-shapes.mlir` | Conv dynamic spatial signed-floor reification |
| `test/lit/Dialect/hip-conv-shape-verifier.mlir` | Conv rank/channel/kernel and destination-shape validation |
| `test/lit/Dialect/hip-pool-reify-shapes.mlir` | Pool floor/ceil reification and shared values/indices extents |
| `test/lit/Dialect/hip-pool-shape-verifier.mlir` | Pool mode/type/output-count and destination-shape validation |
| `test/lit/Dialect/hip-causal-conv-reify-shapes.mlir` | CausalConv output and dynamic `weight.K-1` state reification |
| `test/lit/Dialect/hip-causal-conv-shape-verifier.mlir` | CausalConv 1D layout, state, dtype, and attribute validation |
| `test/lit/Dialect/hip-skip-rms-norm-reify-shapes.mlir` | One/two-output Skip RMS Norm input-shape reification |
| `test/lit/Dialect/hip-skip-rms-norm-shape-verifier.mlir` | Skip RMS Norm input/gamma/bias/output/type contracts |
| `test/lit/Conversion/onnx-to-hip/test_conv.mlir` | Conv converter destination shape and omitted-kernel derivation |
| `test/lit/Conversion/onnx-to-hip/test_pool.mlir` | Pool converter dynamic floor/ceil destination formulas |
| `test/lit/Conversion/onnx-to-hip/test_causal_conv_with_state.mlir` | CausalConv destinations including dynamic kernel state length |
| `test/lit/Conversion/onnx-to-hip/test_skip_rms_norm.mlir` | Skip RMS Norm inference output-slot combinations |
| `test/lit/Conversion/hip-to-llvm/test_matmul.mlir` | Per-operand batch counts and strides: compile-time 0 / matrix size, dynamic count comparison, and the runtime-validation ABI |
| `test/lit/Conversion/onnx-to-hip/test_reduce_sum.mlir` | Reduction destinations, including the non-positional `keepdims = 0` dimension mapping |
| `test/lit/Dialect/hip-loop-verifier.mlir` | Loop-carried type contract |
| `test/lit/Dialect/hip-resolve-tensor-dims.mlir` | Production pre-bufferization dim folding |

Complex operations may use dedicated files; common shape categories should extend the consolidated infer-shapes coverage.

## Current limitations

- ONNX MatMul rank-1 operands require promotion to rank 2 before constructing `hip.matmul`; the runtime and current verifier require rank at least 2.
- MatMul supports whole-matrix batch broadcast and one-matrix-per-output-batch operands, including dynamic batch extents. Static partial per-axis broadcast is rejected during compilation; a dynamically concealed partial layout returns a recoverable runtime error.
- Reduction destinations fall back to a positional copy from the input when the reduced axes are only known at runtime; reification mirrors that fallback rather than inventing a shape.
- Converter migration to inferred-type builders is incremental; explicit result-type builders remain supported.
- A future multi-result operation that needs custom `InferTypeOpInterface` logic may require a dedicated result-type inference implementation file.
- Runtime-dependent extents without pre-execution SSA remain dynamic.
- Phase 1 only rebuilds supported single-use destination producers; shared or externally supplied destinations are left unchanged.
- The pre-conversion loop-body pass has intentionally narrow ONNX coverage and is not a whole-graph solver.
