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

namespace mlir {
namespace hip {

/// Compute the shape of `A @ B` for matmul with NumPy-style batch broadcast
/// over the leading dims. By default last two dims of `aShape` are `[M, K]`
/// and last two dims of `bShape` are `[K, N]`. When `transA` / `transB` are
/// set the corresponding operand's last two dims are swapped before the
/// contraction (compile-time fusion of `Transpose(perm=[..,r,r-2])`).
///
/// Returns the inferred shape on success. Returns an empty `SmallVector` and
/// emits a diagnostic via `emitError` on rank-, K-, or batch-broadcast
/// mismatch.
///
/// `ShapedType::kDynamic` is treated as a wildcard:
///   - K_a or K_b dynamic -> K match passes (result K is dropped anyway).
///   - Batch dim broadcast follows NumPy / TF / ONNX MatMul semantics
///     (delegated to `mlir::OpTrait::util::getBroadcastedShape`):
///       * 1 broadcasts against any dim.
///       * dynamic + static>1 -> static (the dynamic side must be 1 or
///         match the static side at runtime per the broadcast contract;
///         taking the static side is the strictly-correct tightening).
///       * dynamic + dynamic -> dynamic.
///       * static + static, equal -> static; unequal and neither is 1
///         -> error.
SmallVector<int64_t>
inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                 function_ref<InFlightDiagnostic()> emitError,
                 int64_t transA = 0, int64_t transB = 0);

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
LogicalResult verifyHipOpShape(
    Operation *op,
    function_ref<SmallVector<SmallVector<int64_t>>()> computeExpected);

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

/// Compute the NumPy-broadcast result shape over `operands` and lift each
/// output dim to an `OpFoldResult`. Static result dims become `IndexAttr`
/// (no IR emitted); dynamic result dims become `tensor.dim` against
/// whichever operand contributes the runtime extent — right-aligned, and
/// preferring the canonical side (in-range and != 1) when multiple
/// operands could contribute. The canonical-side preference matches the
/// batch-dim contract in `MatmulOp::reifyResultShapes` and ensures that
/// a future `tensor.dim` of the result folds back to the operand that
/// actually determines the size at runtime.
///
/// Used by elementwise ops that take broadcast-shape operands and write
/// the broadcast result into their `outs` (add, mul, sub, div, min, mod,
/// equal, less, and, where, ...). The output dtype is taken from the
/// op's `outs` operand and is independent of this helper — comparisons
/// (equal, less) emit i1 outs while the operands are typically f32/f16,
/// and the helper handles both cases identically (it only looks at
/// shapes).
///
/// All operands must be `RankedTensorType`-typed Values (the interface
/// contract for `reifyResultShapes` callers). Returns an empty vector
/// if broadcast fails — verifiers should already have caught this, but
/// reify bails defensively to avoid materializing nonsense IR.
SmallVector<OpFoldResult> reifyBroadcastShape(OpBuilder &b, Location loc,
                                              ValueRange operands);

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

/// Reify the result shape of a reduction op (reduce_sum / reduce_max /
/// reduce_prod) given `data`, the `axes` operand (rank-1 i64 tensor),
/// and the `keepdims` / `noop_with_empty_axes` attributes.
///
/// Tries to introspect `axes` as an `arith.constant` (the typical case
/// after the OnnxToHip converter materializes it from the ONNX
/// attribute). When successful:
///   - keepdims=1: axes-listed dims become `IndexAttr(1)`; non-axes
///     dims pass through from `data`.
///   - keepdims=0: axes-listed dims are dropped from the output rank;
///     non-axes dims pass through.
///   - Empty axes + noop_with_empty_axes=0: ALL dims become 1
///     (keepdims=1) or output is rank-0 (keepdims=0).
///   - Empty axes + noop_with_empty_axes=1: output equals input
///     (no reduction).
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
