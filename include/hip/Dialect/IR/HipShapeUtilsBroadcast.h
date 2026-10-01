/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_BROADCAST_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_BROADCAST_H

#include "hip/Dialect/IR/HipShapeUtilsCommon.h"

namespace mlir {
namespace hip {

/// Pure NumPy right-aligned broadcast over static shapes. Dynamic dimensions
/// are honest wildcards. This is the common static rule used by conversion,
/// reification, and verification.
FailureOr<SmallVector<int64_t>>
inferBroadcastShape(ArrayRef<ArrayRef<int64_t>> shapes,
                    function_ref<InFlightDiagnostic()> emitError);

/// Generated-verifier target for `Hip_DpsOp_Broadcast`.
LogicalResult verifyBroadcastDpsOp(Operation *op, ValueRange operands);

/// Compute the NumPy-broadcast result shape over ranked tensor `operands`,
/// using `tensor::getMixedSizes` for each. Static extents remain `IndexAttr`;
/// dynamic/dynamic pairs materialize the runtime broadcast rule as index SSA.
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
                          function_ref<InFlightDiagnostic()> emitError,
                          ArrayRef<int64_t> canonicalOperandForResultDim = {});

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

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_BROADCAST_H
