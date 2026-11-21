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

/// Match an inline `arith.constant` rank-0/rank-1 integer tensor and parse it
/// with `parseDenseIntElements`. Generic HIP dialect code intentionally does
/// not inspect frontend operations or conversion-side storage wrappers.
bool matchConstantIntTensor(Value value, SmallVectorImpl<int64_t> &out,
                            std::optional<int64_t> expectedRank = std::nullopt);

// A failed reification must leave the IR unchanged.

/// Infer the matmul result shape with right-aligned batch broadcasting.
/// transA and transB swap the last two dimensions before contraction.
/// Dynamic extents must satisfy the matmul contract at runtime.
/// Return failure and emit a diagnostic for incompatible ranks or dimensions.
FailureOr<SmallVector<int64_t>>
inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                 function_ref<InFlightDiagnostic()> emitError,
                 int64_t transA = 0, int64_t transB = 0);

/// Verify one DPS destination against the shape returned by inferShape.
/// The operation must implement DestinationStyleOpInterface.
/// Propagate inference failure without a second diagnostic.
/// Operation verifiers must check element types separately.
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

/// Return a static index attribute or a folded tensor.dim of source.
/// The source must be a ranked tensor.
OpFoldResult reifyDimOrConstant(OpBuilder &b, Location loc, int64_t staticDim,
                                Value source, int64_t sourceDim);

/// Return the shape of the designated input for a shape-preserving operation.
/// Keep static extents as attributes. Query dynamic extents with tensor.dim.
/// FailureOr distinguishes failure from a valid rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyElementwiseSameShape(OpBuilder &b, Location loc, Value source);

/// Store the designated input shape as the operation's reified result shape.
LogicalResult
reifyElementwiseSameShapeFor(OpBuilder &b, Location loc, Value source,
                             Operation *op,
                             ReifiedRankedShapedTypeDims &reified);

/// Compute the NumPy-broadcast result shape over `operands`. Failure leaves the
/// IR unchanged, including when folding dynamic dimensions exposes a conflict.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastResultShape(OpBuilder &b, Location loc, ValueRange operands,
                          function_ref<InFlightDiagnostic()> emitError);

/// Reify a transpose with output[i] = input[perm[i]].
/// The permutation must contain each input axis exactly once.
/// FailureOr distinguishes failure from a valid rank-zero shape.
FailureOr<SmallVector<OpFoldResult>>
reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                     ArrayRef<int64_t> perm);

/// Reify gather as data.shape[:axis] + indices.shape + data.shape[axis+1:].
/// Normalize a negative axis against the data rank. Reject an invalid axis.
FailureOr<SmallVector<OpFoldResult>>
reifyGatherWithAxis(OpBuilder &b, Location loc, Value data, Value indices,
                    int64_t axis);

/// Reify gather_nd from the data shape, index shape, and batch dimensions.
/// Return failure if the index-tuple width is dynamic or the axes are invalid.
/// FailureOr distinguishes failure from a valid rank-zero shape.
FailureOr<SmallVector<OpFoldResult>> reifyGatherND(OpBuilder &b, Location loc,
                                                   Value data, Value indices,
                                                   int64_t batchDims);

/// Reify a reduction with constant axes.
/// Keep reduced axes at extent 1 when keepdims is set. Otherwise, remove them.
/// Empty axes select all axes unless noopWithEmptyAxes is set.
/// Return failure for unknown or invalid axes. A rank-zero result is valid.
LogicalResult reifyReductionWithKeepdims(OpBuilder &b, Location loc, Value data,
                                         Value axes, int64_t keepdims,
                                         int64_t noopWithEmptyAxes,
                                         SmallVectorImpl<OpFoldResult> &out);

/// Reify a reduction from constant axes when possible.
/// If the axes cannot be used, query the DPS destination shape.
/// Require a ranked tensor input and at least one result.
LogicalResult reifyReductionShape(OpBuilder &b, Location loc, Value data,
                                  Value axes, int64_t keepdims,
                                  int64_t noopWithEmptyAxes, Operation *op,
                                  ReifiedRankedShapedTypeDims &reified);

/// Store the exact right-aligned broadcast shape in reified.
/// Reject missing results, non-tensor operands, or incompatible shapes.
/// The Hip_DpsOp_Broadcast family uses this interface implementation.
LogicalResult reifyBroadcastShapeFor(OpBuilder &b, Location loc,
                                     ValueRange operands, Operation *op,
                                     ReifiedRankedShapedTypeDims &reified);

/// Fold pad extents as input + leading padding + trailing padding.
/// Require constant pads, constant axes, and static output extents.
/// A null axes operand selects all input axes.
/// Return failure when these conditions do not hold.
LogicalResult reifyPadShape(OpBuilder &b, Location loc, Value data, Value pads,
                            Value axes, SmallVectorImpl<OpFoldResult> &out);

/// Fold tile extents as input.shape[d] * repeats[d].
/// Require constant repeats and static output extents. Otherwise, return
/// failure.
LogicalResult reifyTileShape(OpBuilder &b, Location loc, Value input,
                             Value repeats, SmallVectorImpl<OpFoldResult> &out);

/// Broadcast the input shape against a constant target-shape tensor.
/// Return failure for unknown, invalid, or dynamic output extents.
LogicalResult reifyExpandShape(OpBuilder &b, Location loc, Value input,
                               Value shape, SmallVectorImpl<OpFoldResult> &out);

/// Fold slice extents with the ONNX index, step, and clamping rules.
/// Require constant bounds and static output extents. Otherwise, return
/// failure. Null axes select all input axes. Null steps use a step of 1.
LogicalResult reifySliceShape(OpBuilder &b, Location loc, Value data,
                              Value starts, Value ends, Value axes, Value steps,
                              SmallVectorImpl<OpFoldResult> &out);

/// Fold the range length from constant integer start, limit, and delta tensors.
/// The limit is exclusive. Each operand must contain one element.
/// Return failure if the length cannot be folded.
LogicalResult reifyRangeShape(OpBuilder &b, Location loc, Value start,
                              Value limit, Value delta,
                              SmallVectorImpl<OpFoldResult> &out);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_H
