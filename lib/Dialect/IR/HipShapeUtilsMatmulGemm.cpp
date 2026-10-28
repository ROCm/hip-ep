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
  if (aShape.size() < 2) {
    emitError() << "matmul A must have rank >= 2, got rank " << aShape.size();
    return failure();
  }
  if (bShape.size() < 2) {
    emitError() << "matmul B must have rank >= 2, got rank " << bShape.size();
    return failure();
  }

  int64_t m = transA ? aShape.back() : aShape[aShape.size() - 2];
  int64_t kA = transA ? aShape[aShape.size() - 2] : aShape.back();
  int64_t kB = transB ? bShape.back() : bShape[bShape.size() - 2];
  int64_t n = transB ? bShape[bShape.size() - 2] : bShape.back();
  if (!ShapedType::isDynamic(kA) && !ShapedType::isDynamic(kB) && kA != kB) {
    emitError() << "matmul contraction dim mismatch: A.shape[-1]=" << kA
                << " vs B.shape[-2]=" << kB;
    return failure();
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> result;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, result)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << mlir::hip::detail::formatShape(aBatch)
                << " B.batch=" << mlir::hip::detail::formatShape(bBatch);
    return failure();
  }
  result.push_back(m);
  result.push_back(n);
  return result;
}
