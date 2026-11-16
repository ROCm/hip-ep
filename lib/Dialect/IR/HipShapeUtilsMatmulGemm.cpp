/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsMatmulGemm.cpp - MatMul/Gemm shape helpers -----------===//

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Traits.h"

using namespace mlir;

FailureOr<SmallVector<int64_t>>
mlir::hip::inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                            function_ref<InFlightDiagnostic()> emitError,
                            int64_t transA, int64_t transB) {
  if (aShape.empty()) {
    emitError() << "matmul A must have rank >= 1, got rank " << aShape.size();
    return failure();
  }
  if (bShape.empty()) {
    emitError() << "matmul B must have rank >= 1, got rank " << bShape.size();
    return failure();
  }

  const bool aVector = aShape.size() == 1;
  const bool bVector = bShape.size() == 1;
  if ((aVector && transA) || (bVector && transB)) {
    emitError() << "matmul cannot transpose a rank-1 operand";
    return failure();
  }
  int64_t m = aVector ? 1 : transA ? aShape.back() : aShape[aShape.size() - 2];
  int64_t kA = transA ? aShape[aShape.size() - 2] : aShape.back();
  int64_t kB = bVector  ? bShape.back()
               : transB ? bShape.back()
                        : bShape[bShape.size() - 2];
  int64_t n = bVector ? 1 : transB ? bShape[bShape.size() - 2] : bShape.back();
  if (!ShapedType::isDynamic(kA) && !ShapedType::isDynamic(kB) && kA != kB) {
    emitError() << "matmul contraction dim mismatch: A.shape[-1]=" << kA
                << " vs B.shape[-2]=" << kB;
    return failure();
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(aVector ? 1 : 2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(bVector ? 1 : 2);
  SmallVector<int64_t> result;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, result)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << mlir::hip::detail::formatShape(aBatch)
                << " B.batch=" << mlir::hip::detail::formatShape(bBatch);
    return failure();
  }
  if (!aVector)
    result.push_back(m);
  if (!bVector)
    result.push_back(n);
  return result;
}
