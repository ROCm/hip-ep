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

/// Return whether an imported ranked type is compatible with a pure inferred
/// shape. A dynamic extent on either side is compatible; unequal static
/// extents are contradictions.
bool isResultTypeCompatibleWithInferredShape(
    RankedTensorType resultType, llvm::ArrayRef<int64_t> inferredShape);

/// Validate against the shared NumPy broadcast shape rule, then build a
/// tensor.empty using the established conversion-time extent source policy.
/// This builder does not materialize exact dynamic broadcast merges.
FailureOr<Value> createBroadcastEmptyTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType resultType,
                                            ValueRange operands);

/// Return !hip.context from function argument 0.
FailureOr<Value> getContextArg(Operation *op, PatternRewriter &rewriter);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIP_CONVERSION_UTILS_H
