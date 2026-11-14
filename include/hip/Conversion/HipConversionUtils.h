/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_CONVERSION_HIP_CONVERSION_UTILS_H
#define HIP_CONVERSION_HIP_CONVERSION_UTILS_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/ArrayRef.h"

namespace mlir {
namespace hip {

/// Check an imported ranked type against an inferred shape.
/// A dynamic extent is compatible with any extent. Static extents must match.
bool isResultTypeCompatibleWithInferredShape(
    RankedTensorType resultType, llvm::ArrayRef<int64_t> inferredShape);

/// Build a tensor.empty with the exact NumPy broadcast shape.
/// Keep the imported result type and encoding. Reject known shape conflicts.
/// On failure, leave the IR unchanged.
FailureOr<Value> createBroadcastEmptyTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType resultType,
                                            ValueRange operands);

/// Build the tensor.empty destination of a reduction over constant axes.
/// Extents follow reifyReductionWithKeepdims, so the destination agrees with
/// the result-shape reification: with keepdims=0 a dynamic result extent is
/// taken from the data dim it survives from, not from the same position.
/// Reject invalid axes and conflicts with resultType, leaving the IR
/// unchanged.
FailureOr<Value> createReductionEmptyTensor(
    OpBuilder &builder, Location loc, RankedTensorType resultType, Value data,
    llvm::ArrayRef<int64_t> axes, int64_t keepdims, int64_t noopWithEmptyAxes);

/// Return !hip.context from function argument 0.
FailureOr<Value> getContextArg(Operation *op, PatternRewriter &rewriter);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIP_CONVERSION_UTILS_H
