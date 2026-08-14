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
class DenseElementsAttr;

namespace arith {
class IndexCastOp;
} // namespace arith

namespace hip {

/// Return whether an arith.index_cast preserves every nonnegative value under
/// the generated ABI's 64-bit index model.
bool isLosslessShapeIndexCast(arith::IndexCastOp op);

/// Return the dense payload of an arith.constant or inline hip.constant whose
/// payload type exactly matches the SSA value type.
DenseElementsAttr matchHipCompileTimeConstantTensor(Value value);

/// Return whether an imported ranked type is proven compatible with a pure
/// inferred shape. Static imported extents require equal static inference.
bool isResultTypeCompatibleWithInferredShape(
    RankedTensorType resultType, llvm::ArrayRef<int64_t> inferredShape);

/// Build a tensor.empty with \p resultType and the dynamic sizes described by
/// \p reifiedShape. Validation completes before index or destination IR emits.
FailureOr<Value>
createEmptyTensorFromReifiedShape(OpBuilder &builder, Location loc,
                                  RankedTensorType resultType,
                                  llvm::ArrayRef<OpFoldResult> reifiedShape);

/// Build a tensor.empty from the shared NumPy broadcast shape rule. Pure shape
/// and imported-result checks complete before shape SSA emits.
FailureOr<Value> createBroadcastEmptyTensor(OpBuilder &builder, Location loc,
                                            RankedTensorType resultType,
                                            ValueRange operands);

/// Return !hip.context from function argument 0.
FailureOr<Value> getContextArg(Operation *op, PatternRewriter &rewriter);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIP_CONVERSION_UTILS_H
