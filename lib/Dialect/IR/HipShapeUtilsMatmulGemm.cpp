/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsMatmulGemm.cpp - MatMul/Gemm shape helpers -----------===//

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Traits.h"

using namespace mlir;

SmallVector<int64_t>
mlir::hip::inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                            function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() < 2) {
    emitError() << "matmul A must have rank >= 2, got rank " << aShape.size();
    return {};
  }
  if (bShape.size() < 2) {
    emitError() << "matmul B must have rank >= 2, got rank " << bShape.size();
    return {};
  }

  int64_t m = aShape[aShape.size() - 2];
  int64_t kA = aShape.back();
  int64_t kB = bShape[bShape.size() - 2];
  int64_t n = bShape.back();
  if (!ShapedType::isDynamic(kA) && !ShapedType::isDynamic(kB) && kA != kB) {
    emitError() << "matmul contraction dim mismatch: A.shape[-1]=" << kA
                << " vs B.shape[-2]=" << kB;
    return {};
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> result;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, result)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << mlir::hip::detail::formatShape(aBatch)
                << " B.batch=" << mlir::hip::detail::formatShape(bBatch);
    return {};
  }
  result.push_back(m);
  result.push_back(n);
  return result;
}
