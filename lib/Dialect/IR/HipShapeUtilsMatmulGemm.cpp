/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsMatmulGemm.cpp - MatMul/Gemm shape helpers -----------===//

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/IR/Block.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hip;

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

  if ((transA != 0 && transA != 1) || (transB != 0 && transB != 1)) {
    emitError() << "matmul transA and transB must be 0 or 1";
    return failure();
  }

  int64_t M = transA ? aShape.back() : aShape[aShape.size() - 2];
  int64_t Ka = transA ? aShape[aShape.size() - 2] : aShape.back();
  int64_t Kb = transB ? bShape.back() : bShape[bShape.size() - 2];
  int64_t N = transB ? bShape[bShape.size() - 2] : bShape.back();

  if (!ShapedType::isDynamic(Ka) && !ShapedType::isDynamic(Kb) && Ka != Kb) {
    emitError() << "matmul contraction dim mismatch: A.shape[-1]=" << Ka
                << " vs B.shape[-2]=" << Kb;
    return failure();
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> result;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, result)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << detail::formatShape(aBatch)
                << " B.batch=" << detail::formatShape(bBatch);
    return failure();
  }
  result.reserve(result.size() + 2);
  result.push_back(M);
  result.push_back(N);
  return result;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferGemmShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                          std::optional<ArrayRef<int64_t>> cShape,
                          int64_t transA, int64_t transB,
                          function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() != 2 || bShape.size() != 2) {
    emitError() << "gemm A and B must be rank-2 tensors";
    return failure();
  }
  if ((transA != 0 && transA != 1) || (transB != 0 && transB != 1)) {
    emitError() << "gemm transA and transB must be 0 or 1";
    return failure();
  }

  int64_t aK = aShape[transA ? 0 : 1];
  int64_t bK = bShape[transB ? 1 : 0];
  if (!ShapedType::isDynamic(aK) && !ShapedType::isDynamic(bK) && aK != bK) {
    emitError() << "gemm contraction dim mismatch: A has " << aK
                << " but B has " << bK;
    return failure();
  }

  SmallVector<int64_t> result = {aShape[transA ? 1 : 0],
                                 bShape[transB ? 0 : 1]};
  if (!cShape)
    return result;

  if (cShape->size() > 2) {
    emitError() << "gemm C must be a ranked tensor of rank at most 2";
    return failure();
  }
  // C broadcasts onto {M, N}; it cannot change either result extent.
  size_t pad = result.size() - cShape->size();
  for (size_t i : llvm::seq<size_t>(pad, result.size())) {
    int64_t cDim = (*cShape)[i - pad];
    if (ShapedType::isDynamic(cDim) || ShapedType::isDynamic(result[i]) ||
        cDim == 1 || cDim == result[i])
      continue;
    emitError() << "gemm C dimension " << cDim
                << " is not broadcastable to output dimension " << result[i];
    return failure();
  }
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyMatmulResultShape(OpBuilder &b, Location loc, Value A, Value B,
                                  function_ref<InFlightDiagnostic()> emitError,
                                  int64_t transA, int64_t transB) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!aType || !bType) {
    emitError() << "matmul operands must be ranked tensors";
    return failure();
  }
  if (failed(inferMatmulShape(aType.getShape(), bType.getShape(), emitError,
                              transA, transB)))
    return failure();

  // Folded batch dimensions can expose a conflict after type validation.
  // Keep new operations detached until all dimensions are compatible.
  Block pending;
  OpBuilder shapeBuilder(b.getContext());
  shapeBuilder.setInsertionPointToEnd(&pending);
  SmallVector<SmallVector<OpFoldResult>> batchShapes;
  for (Value operand : {A, B}) {
    auto type = cast<RankedTensorType>(operand.getType());
    SmallVector<OpFoldResult> batch;
    for (int64_t i : llvm::seq<int64_t>(0, type.getRank() - 2))
      batch.push_back(reifyDimOrConstant(shapeBuilder, loc, type.getDimSize(i),
                                         operand, i));
    batchShapes.push_back(std::move(batch));
  }
  FailureOr<SmallVector<OpFoldResult>> result =
      detail::reifyBroadcastShape(shapeBuilder, loc, batchShapes, emitError);
  if (failed(result))
    return failure();
  int64_t mDim = aType.getRank() - (transA ? 1 : 2);
  int64_t nDim = bType.getRank() - (transB ? 2 : 1);
  result->push_back(
      reifyDimOrConstant(shapeBuilder, loc, aType.getDimSize(mDim), A, mDim));
  result->push_back(
      reifyDimOrConstant(shapeBuilder, loc, bType.getDimSize(nDim), B, nDim));
  // Insert through the caller's builder to notify its listener.
  while (!pending.empty()) {
    Operation *op = &pending.front();
    op->remove();
    b.insert(op);
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferMatMulNBitsShape(ArrayRef<int64_t> aShape, int64_t N) {
  if (aShape.empty() || N < 0)
    return failure();
  SmallVector<int64_t> result(aShape.begin(), aShape.end());
  result.back() = N;
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyMatMulNBitsResultShape(OpBuilder &b, Location loc, Value A,
                                       int64_t N) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  if (!aType || failed(inferMatMulNBitsShape(aType.getShape(), N)))
    return failure();

  ArrayRef<int64_t> aShape = aType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(aShape.size());
  for (size_t i : llvm::seq<size_t>(0, aShape.size() - 1))
    dims.push_back(reifyDimOrConstant(b, loc, aShape[i], A, i));
  dims.push_back(b.getIndexAttr(N));
  return dims;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGemmResultShape(OpBuilder &b, Location loc, Value A, Value B,
                                Value optionalC, int64_t transA, int64_t transB,
                                function_ref<InFlightDiagnostic()> emitError) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!aType || !bType) {
    emitError() << "gemm A and B must be ranked tensors";
    return failure();
  }
  std::optional<ArrayRef<int64_t>> cShape;
  if (optionalC) {
    auto cType = dyn_cast<RankedTensorType>(optionalC.getType());
    if (!cType) {
      emitError() << "gemm C must be a ranked tensor";
      return failure();
    }
    cShape = cType.getShape();
  }
  if (failed(inferGemmShape(aType.getShape(), bType.getShape(), cShape, transA,
                            transB, emitError)))
    return failure();

  size_t mDim = transA ? 1 : 0;
  size_t nDim = transB ? 0 : 1;
  return SmallVector<OpFoldResult>{
      reifyDimOrConstant(b, loc, aType.getDimSize(mDim), A, mDim),
      reifyDimOrConstant(b, loc, bType.getDimSize(nDim), B, nDim)};
}
