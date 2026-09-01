/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_SHAPE_OPS_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_SHAPE_OPS_H

#include "hip/Dialect/IR/HipShapeUtilsCommon.h"

namespace mlir {
namespace hip {

/// ONNX Size always produces one rank-zero i64 result.
FailureOr<SmallVector<int64_t>> inferSizeShape();

/// Compute the result shape of an ONNX-style `pad` op:
///   `output[d] = data.shape[d] + pre_pad[d] + post_pad[d]`
/// where `pre_pad[d]` / `post_pad[d]` come from `pads[axes.find(d)]` /
/// `pads[axes.find(d) + N]` (default `axes = [0, rank)`, `N = num_axes`).
///
/// The pure form validates axes, lengths, duplicate axes, and statically
/// computable extents. Dynamic data dimensions remain dynamic.
FailureOr<SmallVector<int64_t>>
inferPadShape(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> pads,
              std::optional<ArrayRef<int64_t>> axes);
///
/// The mixed form first resolves compile-time pads/axes from the optional
/// stamped attributes or constant operands and validates through
/// `inferPadShape`. Only then does it emit exact affine index SSA for dynamic
/// input dimensions. Payload-dynamic pads/axes return failure so the caller
/// can preserve its synchronized-readback (converter) or outs-lift (reify)
/// policy; dialect reification never reads tensor payload.
///
/// `axes` may be null (ONNX "pad all axes" default).
LogicalResult reifyPadShape(OpBuilder &b, Location loc, Value data, Value pads,
                            Value axes,
                            std::optional<ArrayRef<int64_t>> staticPads,
                            std::optional<ArrayRef<int64_t>> staticAxes,
                            SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `tile` op:
///   `output[d] = input.shape[d] * repeats[d]`
/// where `repeats` is a rank-1 i64 tensor of length `input.rank`.
///
/// The pure static form validates rank/length/non-negative repeats.
FailureOr<SmallVector<int64_t>> inferTileShape(ArrayRef<int64_t> inputShape,
                                               ArrayRef<int64_t> repeats);

/// Mixed form for constant repeats. Dynamic input dims produce exact index SSA;
/// payload-dynamic repeats return failure so the caller can use bulk readback
/// (converter) or outs-lift (reify).
LogicalResult reifyTileShape(OpBuilder &b, Location loc, Value input,
                             Value repeats,
                             std::optional<ArrayRef<int64_t>> staticRepeats,
                             SmallVectorImpl<OpFoldResult> &out);

/// Reify the result shape of an ONNX-style `expand` op:
///   broadcast `input.shape` against `shape`'s constant values
///   (right-aligned, leading-1 padded on whichever side is shorter).
///
/// Pure static/dynamic broadcast validation used before conversion emits IR.
FailureOr<SmallVector<int64_t>> inferExpandShape(ArrayRef<int64_t> inputShape,
                                                 ArrayRef<int64_t> targetShape);

///
/// Same fold-or-bail strategy. `shape` is the target-shape operand
/// (rank-1 i64 tensor); when it is an `arith.constant` with a
/// `DenseIntElementsAttr`, the helper runs MLIR's
/// `OpTrait::util::getBroadcastedShape` and lifts the result shape.
/// Returns `failure()` when the broadcast result has any dynamic dim
/// (so the caller falls back to outs-lifting).
LogicalResult
reifyExpandShape(OpBuilder &b, Location loc, Value input, Value shape,
                 SmallVectorImpl<OpFoldResult> &out,
                 std::optional<ArrayRef<int64_t>> staticShape = std::nullopt);

/// Compute the result shape of an ONNX-style `slice` op. `output[axis]`
/// for each `axis` in `axes` is `ceil_div(end - start, step)` (negative
/// indices and steps clamp per ONNX rules); for axes not in `axes`,
/// `output[d] = data.shape[d]`.
///
/// Sliced dynamic input extents remain dynamic in the pure result; the mixed
/// helper below materializes their exact clamp arithmetic as SSA.
FailureOr<SmallVector<int64_t>>
inferSliceShape(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> starts,
                ArrayRef<int64_t> ends, std::optional<ArrayRef<int64_t>> axes,
                std::optional<ArrayRef<int64_t>> steps);

/// Exact host-side Slice controls materialized from signed i64 SSA parameters.
/// Every array has one entry per data dimension. Invalid runtime controls
/// produce `valid=false`, zero extents/starts, and unit steps.
struct MaterializedSliceParameters {
  Value valid;
  SmallVector<Value> starts;
  SmallVector<Value> steps;
  SmallVector<Value> extents;
};

/// Materialize the ONNX Slice normalization rule using standard arith SSA.
/// Structural/type/length validation completes before the first operation is
/// emitted. `readbackValid` is the status from the single grouped runtime
/// readback, or a constant true supplied by an all-static caller.
LogicalResult materializeSliceParameters(OpBuilder &b, Location loc, Value data,
                                         ArrayRef<Value> starts,
                                         ArrayRef<Value> ends,
                                         std::optional<ArrayRef<Value>> axes,
                                         std::optional<ArrayRef<Value>> steps,
                                         Value readbackValid,
                                         MaterializedSliceParameters &out);
///
/// The value-based mixed form folds all four parameter tensors and delegates to
/// the compile-time-value overload. Payload-dynamic parameters return failure;
/// conversion handles them through one grouped `hip.readback_control` and the
/// SSA materializer above.
///
/// `axes` and `steps` may be null (ONNX defaults: `[0, rank)` and
/// all-ones respectively).
LogicalResult reifySliceShape(OpBuilder &b, Location loc, Value data,
                              Value starts, Value ends, Value axes, Value steps,
                              SmallVectorImpl<OpFoldResult> &out);

/// Reify an exact Slice shape from compile-time parameters. Validation
/// completes before any IR is emitted; sliced dynamic input dimensions become
/// ONNX clamp and ceil-div index SSA. Negative indices and steps are preserved.
LogicalResult reifySliceShape(OpBuilder &b, Location loc, Value data,
                              ArrayRef<int64_t> starts, ArrayRef<int64_t> ends,
                              std::optional<ArrayRef<int64_t>> axes,
                              std::optional<ArrayRef<int64_t>> steps,
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
LogicalResult
reifyRangeShape(OpBuilder &b, Location loc, Value start, Value limit,
                Value delta, SmallVectorImpl<OpFoldResult> &out,
                std::optional<ArrayRef<int64_t>> staticValues = std::nullopt);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_SHAPE_OPS_H
