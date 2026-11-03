/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_INTERNAL_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_INTERNAL_H

#include "hip/Dialect/IR/HipShapeUtils.h"

#include <string>

namespace mlir::hip::detail {

/// Return the shape of a ranked tensor or memref.
/// Return an empty view for unsupported types. Callers must check the type.
ArrayRef<int64_t> getShapeOf(Value value);

/// Format a shape for diagnostics. Write a dynamic extent as `?`.
std::string formatShape(ArrayRef<int64_t> shape);

/// Fold already-reified operand shapes with NumPy broadcast semantics.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastShape(OpBuilder &b, Location loc,
                    ArrayRef<SmallVector<OpFoldResult>> inputShapes,
                    function_ref<InFlightDiagnostic()> emitError);

} // namespace mlir::hip::detail

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_INTERNAL_H
