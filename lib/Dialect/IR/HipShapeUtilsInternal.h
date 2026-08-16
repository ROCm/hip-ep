/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_INTERNAL_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_INTERNAL_H

#include "hip/Dialect/IR/HipShapeUtils.h"

#include <string>

namespace mlir::hip::detail {

/// Read the shape of a ranked tensor or memref. Returns an empty view for
/// unsupported types; verifier callers reject those before calling, while
/// reification callers use the empty view as a silent bail-out.
ArrayRef<int64_t> getShapeOf(Value value);

/// Pretty-print a static shape for implementation diagnostics.
std::string formatShape(ArrayRef<int64_t> shape);

/// Fold already-reified operand shapes with NumPy broadcast semantics.
FailureOr<SmallVector<OpFoldResult>>
reifyBroadcastShape(OpBuilder &b, Location loc,
                    ArrayRef<SmallVector<OpFoldResult>> inputShapes,
                    function_ref<InFlightDiagnostic()> emitError,
                    ArrayRef<int64_t> canonicalOperandForResultDim = {});

/// Apply `dim * scale + offset`, folding constants when possible. Static
/// arithmetic is checked before narrowing to an index attribute. Dynamic SSA
/// arithmetic intentionally retains the existing index operations; callers
/// must validate every statically-derived scale and offset before calling.
FailureOr<OpFoldResult> scaleAndOffsetDim(OpBuilder &b, Location loc,
                                          OpFoldResult dim, int64_t scale,
                                          int64_t offset);

} // namespace mlir::hip::detail

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_INTERNAL_H
