/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_REDUCTION_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_REDUCTION_H

#include "hip/Dialect/IR/HipShapeUtilsCommon.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hip {

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

/// Return whether a reduction leaf's generated runtime implements
/// `elementType`. `operationName` is the canonical HIP operation name (for
/// example, "hip.reduce_sum"). This is the single dtype matrix used by
/// conversion, dialect verification, and lowering.
bool isSupportedReductionElementType(StringRef operationName, Type elementType);

/// Human-readable supported dtype list paired with
/// `isSupportedReductionElementType`, for diagnostics.
StringRef getSupportedReductionElementTypes(StringRef operationName);

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

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_REDUCTION_H
