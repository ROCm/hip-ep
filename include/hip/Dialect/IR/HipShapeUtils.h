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

//===----------------------------------------------------------------------===//
// Contract shared by every helper in this header
//
// The `infer*` helpers are pure functions of static shapes: they take no
// builder and emit no IR. The `reify*` helpers may materialize index SSA, and
// each one validates every precondition through its `infer*` counterpart
// BEFORE touching the builder. A `reify*` failure therefore never leaves
// stray ops behind, which both the pattern-rewrite contract and
// `ReifyRankedShapedTypeOpInterface` require (see
// `GreedyPatternRewriteDriver`'s expensive checks and
// `ResolveShapedTypeResultDims`).
//
// Converter destination construction and `reifyResultShapes` call the same
// helper for a given op, so the DPS `outs` shape and the shape observed by
// consumers cannot disagree. See `docs/design/hip-shape-inference.md`.
//===----------------------------------------------------------------------===//

/// Compute the shape of `A @ B` for matmul with NumPy-style batch broadcast
/// over the leading dimensions. The matrix dimensions are `A[..., M, K]` and
/// `B[..., K, N]`.
///
/// Dynamic contraction dimensions are treated as compatible. Batch dimensions
/// use `OpTrait::util::getBroadcastedShape`. On failure, emits a diagnostic
/// through `emitError`.
FailureOr<SmallVector<int64_t>>
inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                 function_ref<InFlightDiagnostic()> emitError);

/// Verify that MatMul's broadcasted batches are representable by one constant
/// strided-batch offset per operand. A stride can only express "one matrix
/// broadcast across every output batch" (stride 0) or "one matrix per output
/// batch" (stride == matrix size), so an operand is rejected only when it
/// provably needs something in between: a partial broadcast that pads some
/// batch axes up to the output extent while carrying batches on others.
///
/// Extents that are not statically 1 count as carrying batches, so an unknown
/// extent never hides a partial broadcast. Ordinary batched matmul with
/// dynamic leading extents (`[?, H, M, K] @ [?, H, K, N]`) is representable
/// and accepted.
LogicalResult
verifyStridedBatchMatmul(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                         function_ref<InFlightDiagnostic()> emitError);

/// Compute ONNX Gemm's rank-2 `{M, N}` result shape from static extents.
/// Validates that A and B are rank 2, that `transA`/`transB` are 0 or 1, that
/// the transpose-aware contraction extents agree, and that the optional C is
/// unidirectionally broadcastable onto `{M, N}`. `cShape` is `std::nullopt`
/// when C is absent; C never contributes M or N.
FailureOr<SmallVector<int64_t>>
inferGemmShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
               std::optional<ArrayRef<int64_t>> cShape, int64_t transA,
               int64_t transB, function_ref<InFlightDiagnostic()> emitError);

/// Verify that the `outs` shape of a single-destination DPS HIP op matches the
/// shape returned by `inferShape`. `op` must implement
/// `DestinationStyleOpInterface` and have exactly one DPS init.
///
/// `inferShape` is invoked once and returns `failure()` when the underlying
/// `infer*` helper already emitted a diagnostic, which this function
/// propagates without re-emitting. Pair it directly with an `infer*` helper:
///
///   return verifyHipOpShape(*this, [&] {
///     return inferMatmulShape(aShape, bShape,
///                             [&] { return this->emitOpError(); });
///   });
///
/// Element-type checks are intentionally not handled here: dtype-changing
/// ops (cast, equal, less, not, and) keep their own element-type checks in
/// their op-local verifiers.
LogicalResult
verifyHipOpShape(Operation *op,
                 function_ref<FailureOr<SmallVector<int64_t>>()> inferShape);

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
/// `source` must be a `RankedTensorType`-typed Value -- this helper is
/// called from `reifyResultShapes` impls, which are invoked only in
/// tensor mode per the interface contract.
SmallVector<OpFoldResult> reifyElementwiseSameShape(OpBuilder &b, Location loc,
                                                    Value source);

/// Compute a NumPy-broadcast result shape from already-reified operand shapes.
/// Static extents remain `IndexAttr`; dynamic/dynamic pairs materialize the
/// runtime broadcast rule as ordinary index SSA.
///
/// Before (choosing either dynamic operand is incorrect when it is 1):
///   %lhs_dim = tensor.dim %lhs, %c0
///   %init = tensor.empty(%lhs_dim) : tensor<?xf32>
/// After:
///   %lhs_dim = tensor.dim %lhs, %c0
///   %rhs_dim = tensor.dim %rhs, %c0
///   %lhs_is_one = arith.cmpi eq, %lhs_dim, %c1 : index
///   %extent = arith.select %lhs_is_one, %rhs_dim, %lhs_dim : index
///   %init = tensor.empty(%extent) : tensor<?xf32>
///
/// The `FailureOr` distinguishes failure from a successful rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastShape(OpBuilder &b, Location loc,
                    ArrayRef<SmallVector<OpFoldResult>> inputShapes,
                    function_ref<InFlightDiagnostic()> emitError);

/// Compute the NumPy-broadcast result shape over ranked tensor `operands`.
/// This is the ValueRange convenience wrapper around the mixed-shape helper
/// above and uses `tensor::getMixedSizes` for each operand.
///
/// Used by elementwise ops that take broadcast-shape operands and write
/// the broadcast result into their `outs` (add, mul, sub, div, min, mod,
/// equal, less, and, where, ...). The output dtype is taken from the
/// op's `outs` operand and is independent of this helper — comparisons
/// (equal, less) emit i1 outs while the operands are typically f32/f16,
/// and the helper handles both cases identically (it only looks at
/// shapes).
///
/// All operands must be `RankedTensorType`-typed Values.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastResultShape(OpBuilder &b, Location loc, ValueRange operands,
                          function_ref<InFlightDiagnostic()> emitError);

/// Reify ONNX MatMul's result shape: broadcast the leading batch dimensions,
/// append M from `A[-2]`, and append N from `B[-1]`. Rank-1 MatMul is outside
/// the current HIP op contract.
FailureOr<SmallVector<OpFoldResult>>
reifyMatmulResultShape(OpBuilder &b, Location loc, Value A, Value B,
                       function_ref<InFlightDiagnostic()> emitError);

/// Reify ONNX Gemm's rank-2 `{M, N}` result using transpose-aware dimensions.
/// Optional C is checked for static unidirectional broadcast compatibility but
/// never supplies M or N.
FailureOr<SmallVector<OpFoldResult>>
reifyGemmResultShape(OpBuilder &b, Location loc, Value A, Value B,
                     Value optionalC, int64_t transA, int64_t transB,
                     function_ref<InFlightDiagnostic()> emitError);

/// Reify the result shape of a transpose op as `output[i] = input[perm[i]]`.
/// `perm` must be a permutation of `[0, rank-1)` and have the same length
/// as `input`'s rank — the verifier should already guarantee this; the
/// helper bails (returns empty) on mismatch.
///
/// Each output dim `i`:
///   - emits `IndexAttr(input.shape[perm[i]])` when that dim is static,
///   - emits `tensor.dim %input, perm[i]` otherwise.
///
/// `input` must be a `RankedTensorType`-typed Value.
SmallVector<OpFoldResult> reifyTransposeByPerm(OpBuilder &b, Location loc,
                                               Value input,
                                               ArrayRef<int64_t> perm);

/// Reify the result shape of a gather op as
/// `output = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:]`.
/// `axis` is normalized into `[0, data.rank)` (negative axis follows ONNX
/// convention). The helper bails (returns empty) on a malformed axis.
///
/// `data` and `indices` must be `RankedTensorType`-typed Values.
SmallVector<OpFoldResult> reifyGatherWithAxis(OpBuilder &b, Location loc,
                                              Value data, Value indices,
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
/// `data` and `indices` must be `RankedTensorType`-typed Values.
SmallVector<OpFoldResult> reifyGatherND(OpBuilder &b, Location loc, Value data,
                                        Value indices, int64_t batchDims);

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
/// the shape `reifyResultShapes` reports. Returns failure when an axis is out
/// of range for `dataShape`.
FailureOr<SmallVector<int64_t>> inferReductionShape(ArrayRef<int64_t> dataShape,
                                                    ArrayRef<int64_t> axes,
                                                    int64_t keepdims);

/// ONNX reduction result shape over `axes` as mixed extents, emitting
/// `tensor.dim` only for dimensions that are dynamic in `data`. Same mapping
/// as `inferReductionShape`; see it for the `keepdims` semantics. `data` must
/// be a `RankedTensorType`-typed Value.
FailureOr<SmallVector<OpFoldResult>>
reifyReductionResultShape(OpBuilder &b, Location loc, Value data,
                          ArrayRef<int64_t> axes, int64_t keepdims);

/// Reify the result shape of a reduction op (reduce_sum / reduce_max /
/// reduce_prod) given `data`, the `axes` operand (rank-1 i64 tensor),
/// and the `keepdims` / `noop_with_empty_axes` attributes.
///
/// Introspects `axes` as an `arith.constant` (the typical case after the
/// OnnxToHip converter materializes it from the ONNX attribute), resolves
/// ONNX's empty-axes semantics against `noop_with_empty_axes` — reduce every
/// axis when 0, reduce nothing when 1 — and delegates the shape rule to
/// `reifyReductionResultShape`.
///
/// Returns `success()` and writes the reified dim list into `out` when
/// `axes` can be introspected. Returns `failure()` when `axes` is not a
/// recognised constant — the caller should then fall back to
/// `reifyElementwiseSameShape(output)` to keep the reify interface
/// non-failing.
///
/// Uses `LogicalResult` (rather than the empty-vector sentinel used by
/// the other helpers in this header) because a valid rank-0 reduction
/// result has an empty dim list, which would otherwise be
/// indistinguishable from the bail path.
LogicalResult reifyReductionWithKeepdims(OpBuilder &b, Location loc, Value data,
                                         Value axes, int64_t keepdims,
                                         int64_t noopWithEmptyAxes,
                                         SmallVectorImpl<OpFoldResult> &out);

/// One-shot reify body for ONNX-style reduction ops (reduce_sum,
/// reduce_max, reduce_prod). Tries `reifyReductionWithKeepdims` first
/// to recover per-input-dim mappings from a constant `axes` operand.
/// When `axes` is not a recognised constant, falls back to the shared
/// `HipDpsOp` outs-lift default so the reify interface always
/// succeeds (the only honest answer when we cannot decide which dims
/// were reduced is the type of the `outs` operand the converter
/// already picked).
///
/// `op` must implement both `HipDpsOp` (so the fallback can walk
/// `getDpsInits()`) and have a `RankedTensorType` `data` operand.
/// Returns `failure()` only on the no-tensor-results / non-tensor
/// `data` defensive paths; otherwise always returns `success()`.
///
/// Used as the body of `Hip_DpsOp_Reduction`'s auto-emitted reify
/// dispatcher; see `Hip_DpsOp_Reduction` in `HipOps.td`.
LogicalResult reifyReductionShape(OpBuilder &b, Location loc, Value data,
                                  Value axes, int64_t keepdims,
                                  int64_t noopWithEmptyAxes, Operation *op,
                                  ReifiedRankedShapedTypeDims &reified);

/// One-shot reify body for elementwise NumPy-broadcast ops (add, mul,
/// sub, div, min, mod, equal, less, and, where, ...). Wraps
/// `reifyBroadcastResultShape` with the per-op guards (no-results bail,
/// every operand must be `RankedTensorType`) and writes the lifted
/// dim list into `reified`.
///
/// `operands` is the list of broadcast input operands in the order
/// they should be aligned (right-aligned for NumPy broadcast).
/// Returns `failure()` on any defensive bail or when broadcast itself fails.
/// A successful rank-zero result writes one empty shape into `reified`.
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
