/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_H

#include "mlir/IR/Builders.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>

namespace mlir {
namespace hip {

/// Parse the payload of a rank-0 or rank-1 dense integer tensor into signed
/// i64 values. `expectedRank` may restrict callers to scalar or vector form.
bool parseDenseIntElements(DenseElementsAttr dense,
                           SmallVectorImpl<int64_t> &out,
                           std::optional<int64_t> expectedRank = std::nullopt);

/// Match a rank-0/rank-1 integer constant and parse it with
/// `parseDenseIntElements`. Recognized structural sources are an inline
/// `arith.constant` tensor and a `memref.get_global` referencing a constant
/// global with a dense initializer. The latter preserves constant provenance
/// across tensor-to-memref bufferization. Generic HIP dialect code does not
/// inspect frontend operations.
bool matchConstantIntTensor(Value value, SmallVectorImpl<int64_t> &out,
                            std::optional<int64_t> expectedRank = std::nullopt);

// Every `reify*` helper validates its preconditions before touching the
// builder. A failed rewrite or reification therefore leaves no stray IR.

/// Compute the shape of `A @ B` for matmul with NumPy-style batch broadcast
/// over the leading dims. By default last two dims of `aShape` are `[M, K]`
/// and last two dims of `bShape` are `[K, N]`. When `transA` / `transB` are
/// set the corresponding operand's last two dims are swapped before the
/// contraction (compile-time fusion of `Transpose(perm=[..,r,r-2])`).
///
/// Dynamic contraction dimensions are unknown-compatible and are checked for
/// equality by the runtime. Two statically known unequal contraction extents
/// are rejected. Batch dimensions use `OpTrait::util::getBroadcastedShape`.
/// On failure, emits a diagnostic through `emitError`.
FailureOr<SmallVector<int64_t>>
inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                 function_ref<InFlightDiagnostic()> emitError,
                 int64_t transA = 0, int64_t transB = 0);

/// Verify that MatMul's broadcasted batches are representable by one constant
/// strided-batch offset per operand. A stride can only express "one matrix
/// broadcast across every output batch" (stride 0) or "one matrix per output
/// batch" (stride == matrix size), so an operand is rejected only when it
/// provably needs something in between: a partial broadcast that pads some
/// batch axes up to the output extent while carrying batches on others.
///
/// Dynamic batch layouts are accepted after all statically visible partial
/// broadcasts are rejected. At runtime, each operand's matrix count must be 1
/// or the output batch count; otherwise the runtime wrapper reports a
/// recoverable error before BLAS dispatch.
LogicalResult
verifyStridedBatchMatmul(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                         function_ref<InFlightDiagnostic()> emitError);

/// Compute ONNX Gemm's rank-2 `{M, N}` result shape from static extents.
/// Validates that A and B are rank 2, that `transA`/`transB` are 0 or 1, that
/// statically known transpose-aware contraction extents agree (dynamic extents
/// are unknown-compatible), and that the optional C is unidirectionally
/// broadcastable onto `{M, N}`. `cShape` is `std::nullopt` when C is absent; C
/// never contributes M or N.
FailureOr<SmallVector<int64_t>>
inferGemmShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
               std::optional<ArrayRef<int64_t>> cShape, int64_t transA,
               int64_t transB, function_ref<InFlightDiagnostic()> emitError);

/// Verify that the actual `outs` operand shapes of a DPS HIP op match the
/// shapes returned by `computeExpected`. `op` must implement
/// `DestinationStyleOpInterface`.
///
/// `computeExpected` is invoked once and must return one shape per init
/// operand (== one per `OpResult` for tensor mode; same count for memref
/// mode, just no SSA result). An empty outer vector signals that the
/// shape-arithmetic helper already emitted a diagnostic — this function
/// returns `failure()` without re-emitting.
///
/// Element-type checks are intentionally not handled here: dtype-changing
/// ops (cast, equal, less, not, and) keep their own element-type checks in
/// their op-local verifiers.
/// Verify one DPS destination against a pure `infer*` shape rule.
LogicalResult
verifyHipOpShape(Operation *op,
                 function_ref<FailureOr<SmallVector<int64_t>>()> inferShape,
                 unsigned initIndex = 0);

/// Verify the common tensor/memref and result/init invariants of a HIP DPS
/// compute operation.
LogicalResult verifyDpsComputeOp(Operation *op, ArrayRef<Value> dataOperands,
                                 unsigned numInits);

/// Pure NumPy right-aligned broadcast over static shapes.
FailureOr<SmallVector<int64_t>>
inferBroadcastShape(ArrayRef<ArrayRef<int64_t>> shapes,
                    function_ref<InFlightDiagnostic()> emitError);

/// Generated-verifier target for `Hip_DpsOp_Broadcast`.
LogicalResult verifyBroadcastDpsOp(Operation *op, ValueRange operands);

/// Build an `OpFoldResult` for one dimension of a reify-callable op's
/// result:
///   - if `staticDim` is not `kDynamic`, returns `b.getIndexAttr(staticDim)`
///     (no IR emitted).
///   - otherwise emits `tensor.dim` against `source` at `sourceDim`. The
///     dim op folds to an `arith.constant` automatically when
///     `source.getType()` has a static size at `sourceDim`.
///
/// `source` is required to have `RankedTensorType` -- the
/// `ReifyRankedShapedTypeOpInterface` contract restricts its callers to
/// ops with tensor results, so memref-typed sources cannot reach a reify
/// path today.
OpFoldResult reifyDimOrConstant(OpBuilder &b, Location loc, int64_t staticDim,
                                Value source, int64_t sourceDim);

/// Reify the result shape of a shape-preserving DPS op as the runtime
/// shape of `source`: each static dim becomes `IndexAttr`, each dynamic
/// dim becomes `tensor.dim %source, %i`. Used by ops whose result has
/// the same shape as one designated input (e.g. rope, rms_norm, qmoe).
///
/// The `FailureOr` distinguishes a valid rank-zero shape from failure.
FailureOr<SmallVector<OpFoldResult>>
reifyElementwiseSameShape(OpBuilder &b, Location loc, Value source);

/// One-shot reifier target for named-source same-shape operations.
LogicalResult
reifyElementwiseSameShapeFor(OpBuilder &b, Location loc, Value source,
                             Operation *op,
                             ReifiedRankedShapedTypeDims &reified);

/// Compute the NumPy-broadcast result shape over `operands`. Static
/// broadcastability is validated before any `tensor.dim` or arithmetic op is
/// emitted.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastResultShape(OpBuilder &b, Location loc, ValueRange operands,
                          function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> canonicalOperandForResultDim = {});

/// Reify ONNX MatMul's result shape: broadcast the leading batch dimensions,
/// append M from `A[-2]`, and append N from `B[-1]`. Rank-1 MatMul is outside
/// the current HIP op contract.
FailureOr<SmallVector<OpFoldResult>>
reifyMatmulResultShape(OpBuilder &b, Location loc, Value A, Value B,
                       function_ref<InFlightDiagnostic()> emitError,
                       int64_t transA = 0, int64_t transB = 0);

/// Pure MatMulNBits result shape: A's leading dimensions followed by N.
FailureOr<SmallVector<int64_t>> inferMatMulNBitsShape(ArrayRef<int64_t> aShape,
                                                      int64_t N);

/// Reify MatMulNBits' result shape: `A`'s leading dimensions followed by the
/// static `N` attribute. The quantized weight is stored transposed, so the
/// contraction dimension never appears in the result -- which is why the last
/// dimension comes from `N` and not from `A`.
///
/// `A` must be a `RankedTensorType`-typed Value of rank at least 1.
FailureOr<SmallVector<OpFoldResult>>
reifyMatMulNBitsResultShape(OpBuilder &b, Location loc, Value A, int64_t N);

/// Reify ONNX Gemm's rank-2 `{M, N}` result using transpose-aware dimensions.
/// Optional C is checked for static unidirectional broadcast compatibility but
/// never supplies M or N.
FailureOr<SmallVector<OpFoldResult>>
reifyGemmResultShape(OpBuilder &b, Location loc, Value A, Value B,
                     Value optionalC, int64_t transA, int64_t transB,
                     function_ref<InFlightDiagnostic()> emitError);

/// Compute the ONNX forward-convolution result shape for rank-3 NCL or rank-4
/// NCHW operands. The result is `{input[0], weights[0], spatial...}` and each
/// spatial extent follows the signed-floor window formula.
///
/// An empty `kernelShape` means the ONNX attribute was omitted. In that case
/// the kernel is derived from the static spatial dimensions of `weightShape`;
/// dynamic weight spatial dimensions require an explicit kernel.
FailureOr<SmallVector<int64_t>>
inferConvShape(ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
               ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
               ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
               int64_t group, function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferConvShape`. Validates through the static helper
/// before materializing any dimension arithmetic. Dynamic spatial extents are
/// safely narrowed from signed i128; `runtimeValid`, when requested, receives
/// the combined range check consumed by `hip.conv`.
FailureOr<SmallVector<OpFoldResult>>
reifyConvResultShape(OpBuilder &b, Location loc, Value input, Value weights,
                     ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
                     ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
                     int64_t group,
                     function_ref<InFlightDiagnostic()> emitError,
                     Value *runtimeValid = nullptr);

/// Compute the supported rank-4 NCHW ONNX ConvTranspose result shape.
FailureOr<SmallVector<int64_t>> inferConvTransposeShape(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> outputPadding, int64_t group,
    function_ref<InFlightDiagnostic()> emitError);

/// Mixed-shape form of `inferConvTransposeShape`. Validates through the static
/// helper before materializing any dimension arithmetic.
FailureOr<SmallVector<OpFoldResult>>
reifyConvTransposeResultShape(OpBuilder &b, Location loc, Value input,
                              Value weights, ArrayRef<int64_t> kernelShape,
                              ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
                              ArrayRef<int64_t> dilations,
                              ArrayRef<int64_t> outputPadding, int64_t group,
                              function_ref<InFlightDiagnostic()> emitError);

/// Reify the result shape of a transpose op as `output[i] = input[perm[i]]`.
/// `perm` must be a permutation of `[0, rank-1)` and have the same length
/// as `input`'s rank.
///
/// Each output dim `i`:
///   - emits `IndexAttr(input.shape[perm[i]])` when that dim is static,
///   - emits `tensor.dim %input, perm[i]` otherwise.
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                     ArrayRef<int64_t> perm);

/// Reify the result shape of a gather op as
/// `output = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]`.
/// `axis` is normalized into `[0, data.rank)` (negative axis follows ONNX
/// convention). Returns failure on a malformed axis.
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyGatherWithAxis(OpBuilder &b, Location loc, Value data, Value indices,
                    int64_t axis);

/// Reify the result shape of a `gather_nd` op as
/// `batch_dims_from_data ++ indices.shape[batch_dims:-1] ++
///  data.shape[batch_dims + indices.shape[-1]:]`.
/// Per ONNX GatherND semantics, output rank =
/// `q + r - indices.shape[-1] - 1 - batch_dims`, where `q = rank(indices)`
/// and `r = rank(data)`. The helper bails (returns empty) when the
/// trailing index-tuple width (`indices.shape[-1]`) is dynamic — the
/// output rank itself is then unknown and reify cannot run.
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>> reifyGatherND(OpBuilder &b, Location loc,
                                                   Value data, Value indices,
                                                   int64_t batchDims);

/// ONNX reduction result shape over `axes`, from static extents only.
///
/// `axes` holds the already-resolved reduced axis indices (ONNX negative-axis
/// convention); an empty list means no reduction. `keepdims = 0` drops reduced
/// axes from the output rank, so the output dimension order is *not* positional
/// in the input: reducing axes `[1, 2]` of a rank-4 input maps output dimension
/// 1 to input dimension 3.
///
/// This and `reifyReductionResultShape` share one internal output-to-input
/// dimension mapping, so a destination built from either cannot disagree with
/// the shape `reifyResultShapes` reports. Returns failure when axes are invalid
/// or do not form the one contiguous span supported by the runtime kernel.
FailureOr<SmallVector<int64_t>> inferReductionShape(ArrayRef<int64_t> dataShape,
                                                    ArrayRef<int64_t> axes,
                                                    int64_t keepdims);

/// Normalize ONNX negative axes, reject duplicates/out-of-range values, sort
/// them, and require one contiguous span. Empty axes remain empty.
FailureOr<SmallVector<int64_t>> normalizeReductionAxes(int64_t dataRank,
                                                       ArrayRef<int64_t> axes);

/// Resolve a structurally-proven tensor or memref constant axes operand,
/// applying `noop_with_empty_axes` semantics before normalization. Runtime
/// operands and non-contiguous/malformed axis sets fail.
FailureOr<SmallVector<int64_t>>
resolveConstantReductionAxes(Value axes, int64_t dataRank,
                             int64_t noopWithEmptyAxes);

/// Generated-verifier target for `Hip_DpsOp_Reduction`. Tensor and memref
/// forms both require a structurally-proven constant axes source, one
/// representable contiguous normalized span, and an exact semantic result
/// shape.
LogicalResult verifyReductionDpsOp(Operation *op, Value data, Value axes,
                                   int64_t keepdims, int64_t noopWithEmptyAxes);

/// ONNX reduction result shape over `axes` as mixed extents, emitting
/// `tensor.dim` only for dimensions that are dynamic in `data`. Same mapping
/// as `inferReductionShape`; see it for the `keepdims` semantics. `data` must
/// be a `RankedTensorType`-typed Value.
FailureOr<SmallVector<OpFoldResult>>
reifyReductionResultShape(OpBuilder &b, Location loc, Value data,
                          ArrayRef<int64_t> axes, int64_t keepdims);

/// One-shot reify body for ONNX-style reduction ops. Structurally-proven
/// constant `axes` use the shared semantic dimension map. Runtime,
/// malformed, and non-contiguous axes fail without emitting IR.
///
/// `op` must have a `RankedTensorType` `data` operand.
/// Returns `failure()` on defensive type/result checks, invalid constant axes,
/// or an unsupported axes source/span.
///
/// Used as the body of `Hip_DpsOp_Reduction`'s auto-emitted reify
/// dispatcher; see `Hip_DpsOp_Reduction` in `HipOps.td`.
LogicalResult reifyReductionShape(OpBuilder &b, Location loc, Value data,
                                  Value axes, int64_t keepdims,
                                  int64_t noopWithEmptyAxes, Operation *op,
                                  ReifiedRankedShapedTypeDims &reified);

/// One-shot reify body for elementwise NumPy-broadcast ops (add, mul,
/// sub, div, min, mod, equal, less, and, where, ...). Wraps
/// `reifyBroadcastShape` with the per-op guards (no-results bail,
/// every operand must be `RankedTensorType`) and writes the lifted
/// dim list into `reified`.
///
/// `operands` is the list of broadcast input operands in the order
/// they should be aligned (right-aligned for NumPy broadcast).
/// Returns `failure()` on any defensive bail or when broadcast itself
/// fails (verifier should already have caught the latter; reify bails
/// to avoid materializing nonsense IR).
///
/// Used as the body of `Hip_DpsOp_Broadcast`'s auto-emitted reify
/// dispatcher; see `Hip_DpsOp_Broadcast` in `HipOps.td`.
LogicalResult reifyBroadcastShapeFor(OpBuilder &b, Location loc,
                                     ValueRange operands, Operation *op,
                                     ReifiedRankedShapedTypeDims &reified);

/// Reify the result shape of an ONNX-style `pad` op:
///   `output[d] = data.shape[d] + pre_pad[d] + post_pad[d]`
/// where `pre_pad[d]` / `post_pad[d]` come from `pads[axes.find(d)]` /
/// `pads[axes.find(d) + N]` (default `axes = [0, rank)`, `N = num_axes`).
///
/// Fold-or-bail strategy: tries to introspect `pads` and (if non-null)
/// `axes` as constant int vectors, then per-dim computes static output
/// extents from `data.shape`. Returns `success()` and writes the per-dim
/// `OpFoldResult`s into `out` ONLY when EVERY output dim is statically
/// known; returns `failure()` otherwise (the per-op reify thunk falls
/// back to `HipDpsOp::reifyResultShapes`'s outs-lift default so reify
/// still succeeds end-to-end). This avoids emitting per-dim
/// `arith.addi(tensor.dim, const)` chains that don't fold and would
/// clutter the IR; the typical Tier-1 case is `pads` from an ONNX
/// attribute (constant) + a fully-static `data.shape`, which folds
/// entirely to `IndexAttr` results.
///
/// `axes` may be null (ONNX "pad all axes" default).
LogicalResult reifyPadShape(OpBuilder &b, Location loc, Value data, Value pads,
                            Value axes, SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `tile` op:
///   `output[d] = input.shape[d] * repeats[d]`
/// where `repeats` is a rank-1 i64 tensor of length `input.rank`.
///
/// Same fold-or-bail strategy as `reifyPadShape`: tries to introspect
/// `repeats` as a constant int vector, computes static output extents
/// from `input.shape`, and returns `success()` only when EVERY dim is
/// statically known.
LogicalResult reifyTileShape(OpBuilder &b, Location loc, Value input,
                             Value repeats, SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `expand` op:
///   broadcast `input.shape` against `shape`'s constant values
///   (right-aligned, leading-1 padded on whichever side is shorter).
///
/// Same fold-or-bail strategy. `shape` is the target-shape operand
/// (rank-1 i64 tensor); when it is an `arith.constant` with a
/// `DenseIntElementsAttr`, the helper runs MLIR's
/// `OpTrait::util::getBroadcastedShape` and lifts the result shape.
/// Returns `failure()` when the broadcast result has any dynamic dim
/// (so the caller falls back to outs-lifting).
LogicalResult reifyExpandShape(OpBuilder &b, Location loc, Value input,
                               Value shape, SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `slice` op. `output[axis]`
/// for each `axis` in `axes` is `ceil_div(end - start, step)` (negative
/// indices and steps clamp per ONNX rules); for axes not in `axes`,
/// `output[d] = data.shape[d]`.
///
/// Pure fold-or-bail (no fallback for partial constants). The dim-arith
/// chain `arith.divsi(arith.subi(end, start), step)` per axis would
/// clutter the IR persistently when any of starts / ends / axes / steps
/// is non-constant; returning `failure()` early lets the per-op reify
/// thunk fall back to the `HipDpsOp` outs-lift default for a single
/// `tensor.dim` per dim instead.
///
/// `axes` and `steps` may be null (ONNX defaults: `[0, rank)` and
/// all-ones respectively).
LogicalResult reifySliceShape(OpBuilder &b, Location loc, Value data,
                              Value starts, Value ends, Value axes, Value steps,
                              SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `range` op:
///   `output.shape[0] = ceil_div(max(0, limit - start), delta)` (limit
///   exclusive; clamp to 0 when `limit < start && delta > 0`, etc.).
///
/// `start`, `limit`, `delta` are scalar (rank-0) tensors. Pure
/// fold-or-bail: when ALL three are `arith.constant` with a single int
/// value, computes the static output count and returns a one-element
/// `IndexAttr` vector. Else `failure()` -- the dim-arith chain to
/// compute the count at runtime is rarely useful for refinement and
/// would persist in IR.
LogicalResult reifyRangeShape(OpBuilder &b, Location loc, Value start,
                              Value limit, Value delta,
                              SmallVectorImpl<OpFoldResult> &out);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
