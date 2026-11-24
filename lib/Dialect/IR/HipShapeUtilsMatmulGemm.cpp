/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsMatmulGemm.cpp - MatMul/Gemm shape helpers -----------===//

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Traits.h"
#include "llvm/ADT/Sequence.h"

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

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyMatmulShape(OpBuilder &b, Location loc, Value A, Value B,
                            int64_t transA, int64_t transB,
                            function_ref<InFlightDiagnostic()> emitError) {
  ArrayRef<int64_t> aShape = mlir::hip::detail::getShapeOf(A);
  ArrayRef<int64_t> bShape = mlir::hip::detail::getShapeOf(B);
  if (aShape.empty() || bShape.empty())
    return failure();

  // Validate before emitting dimension queries, including pre-verification
  // calls.
  FailureOr<SmallVector<int64_t>> inferredShape =
      inferMatmulShape(aShape, bShape, emitError, transA, transB);
  if (failed(inferredShape))
    return failure();
  ArrayRef<int64_t> outShape = *inferredShape;

  size_t outRank = outShape.size();
  size_t aRank = aShape.size();
  size_t bRank = bShape.size();

  // Loop-invariant: right-alignment padding for A's and B's batch dims.
  size_t batchRank = outRank - 2;
  size_t aPad = batchRank - (aRank >= 2 ? aRank - 2 : 0);
  size_t bPad = batchRank - (bRank >= 2 ? bRank - 2 : 0);

  SmallVector<OpFoldResult> dims;
  dims.reserve(outRank);
  for (size_t i : llvm::seq<size_t>(0, outRank)) {
    // M dim: A[-2], or A[-1] when transA.
    if (i + 2 == outRank) {
      size_t aDim = transA ? aRank - 1 : aRank - 2;
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], A, aDim));
      continue;
    }
    // N dim: B[-1], or B[-2] when transB.
    if (i + 1 == outRank) {
      size_t bDim = transB ? bRank - 2 : bRank - 1;
      dims.push_back(
          mlir::hip::reifyDimOrConstant(b, loc, outShape[i], B, bDim));
      continue;
    }
    // Batch dim: prefer the side that contributes the size (in range, not 1).
    // When neither contributes, prefer A in range so folds see a stable source.
    int64_t aDim = i < aPad ? 1 : aShape[i - aPad];
    int64_t bDim = i < bPad ? 1 : bShape[i - bPad];
    bool aCanonical = i >= aPad && aDim != 1;
    bool bCanonical = i >= bPad && bDim != 1;
    bool pickA = aCanonical || (!bCanonical && i >= aPad);
    Value src = pickA ? A : B;
    size_t srcDim = pickA ? i - aPad : i - bPad;
    dims.push_back(
        mlir::hip::reifyDimOrConstant(b, loc, outShape[i], src, srcDim));
  }
  return dims;
}
