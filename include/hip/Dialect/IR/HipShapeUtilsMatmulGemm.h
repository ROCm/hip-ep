/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#ifndef HIP_DIALECT_IR_HIP_SHAPE_UTILS_MATMUL_GEMM_H
#define HIP_DIALECT_IR_HIP_SHAPE_UTILS_MATMUL_GEMM_H

#include "hip/Dialect/IR/HipShapeUtilsCommon.h"

namespace mlir {
namespace hip {

/// Compute the shape of `A @ B` for matmul with NumPy-style batch broadcast
/// over the leading dimensions. The matrix dimensions are `A[..., M, K]` and
/// `B[..., K, N]`.
/// `transA` and `transB` swap the corresponding last two dimensions before
/// selecting M, K, and N; batch dimensions are unaffected.
///
///
/// Dynamic contraction dimensions are unknown-compatible and are checked for
/// equality by the runtime. Two statically known unequal contraction extents
/// are rejected. Batch dimensions use `OpTrait::util::getBroadcastedShape`.
/// On failure, emits a diagnostic through `emitError`.
FailureOr<SmallVector<int64_t>>
inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                 function_ref<InFlightDiagnostic()> emitError,
                 int64_t transA = 0, int64_t transB = 0);

/// Verify that MatMul's broadcasted batches are representable by one constant
/// strided-batch offset per operand. A stride can only express "one matrix
/// broadcast across every output batch" (stride 0) or "one matrix per output
/// batch" (stride == matrix size), so an operand is rejected only when it
/// provably needs something in between: a partial broadcast that pads some
/// batch axes up to the output extent while carrying batches on others.
///
/// Dynamic batch layouts are accepted after all statically visible partial
/// broadcasts are rejected. At runtime, each operand's matrix count must be 1
/// or the output batch count; otherwise the runtime wrapper reports a
/// recoverable error before BLAS dispatch.
LogicalResult
verifyStridedBatchMatmul(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                         function_ref<InFlightDiagnostic()> emitError);

/// Compute ONNX Gemm's rank-2 `{M, N}` result shape from static extents.
/// Validates that A and B are rank 2, that `transA`/`transB` are 0 or 1, that
/// statically known transpose-aware contraction extents agree (dynamic extents
/// are unknown-compatible), and that the optional C is unidirectionally
/// broadcastable onto `{M, N}`. `cShape` is `std::nullopt` when C is absent; C
/// never contributes M or N.
FailureOr<SmallVector<int64_t>>
inferGemmShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
               std::optional<ArrayRef<int64_t>> cShape, int64_t transA,
               int64_t transB, function_ref<InFlightDiagnostic()> emitError);

/// Reify ONNX MatMul result shape: broadcast leading batch dimensions, then
/// append M and N selected after applying optional last-two-dimension
/// `transA`/`transB` swaps. Rank-1 MatMul is outside the current contract.
FailureOr<SmallVector<OpFoldResult>>
reifyMatmulResultShape(OpBuilder &b, Location loc, Value A, Value B,
                       function_ref<InFlightDiagnostic()> emitError,
                       int64_t transA = 0, int64_t transB = 0);

/// Pure MatMulNBits result shape: A's leading dimensions followed by N.
FailureOr<SmallVector<int64_t>> inferMatMulNBitsShape(ArrayRef<int64_t> aShape,
                                                      int64_t N);

/// Reify MatMulNBits' result shape: `A`'s leading dimensions followed by the
/// static `N` attribute. The quantized weight is stored transposed, so the
/// contraction dimension never appears in the result -- which is why the last
/// dimension comes from `N` and not from `A`.
///
/// `A` must be a `RankedTensorType`-typed Value of rank at least 1.
FailureOr<SmallVector<OpFoldResult>>
reifyMatMulNBitsResultShape(OpBuilder &b, Location loc, Value A, int64_t N);

/// Reify ONNX Gemm's rank-2 `{M, N}` result using transpose-aware dimensions.
/// Optional C is checked for static unidirectional broadcast compatibility but
/// never supplies M or N.
FailureOr<SmallVector<OpFoldResult>>
reifyGemmResultShape(OpBuilder &b, Location loc, Value A, Value B,
                     Value optionalC, int64_t transA, int64_t transB,
                     function_ref<InFlightDiagnostic()> emitError);

} // namespace hip
} // namespace mlir

#endif // HIP_DIALECT_IR_HIP_SHAPE_UTILS_MATMUL_GEMM_H
