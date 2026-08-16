/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsMatmulGemm.cpp - MatMul and Gemm shape helpers --------===//
//
// Category implementation for the public shape helpers declared in
// `hip/Dialect/IR/HipShapeUtils.h`.
//
//===----------------------------------------------------------------------===//

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Traits.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "llvm/ADT/APInt.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>
#include <limits>
#include <optional>

using namespace mlir;
using namespace mlir::hip;

FailureOr<SmallVector<int64_t>>
mlir::hip::inferMatmulShape(ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
                            function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() < 2) {
    emitError() << "matmul A must have rank >= 2, got rank " << aShape.size();
    return failure();
  }
  if (bShape.size() < 2) {
    emitError() << "matmul B must have rank >= 2, got rank " << bShape.size();
    return failure();
  }

  int64_t M = aShape[aShape.size() - 2];
  int64_t Ka = aShape.back();
  int64_t Kb = bShape[bShape.size() - 2];
  int64_t N = bShape.back();

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

namespace {

/// Whether `batch` -- one operand's leading extents, right-aligned against
/// `outputBatch` -- holds a matrix count that one constant stride can express.
///
/// A stride of 0 reuses a single matrix for every output batch and a stride of
/// the matrix size walks one matrix per output batch, so the operand's matrix
/// count must be either 1 or the output's. A partial broadcast lands strictly
/// between the two: `[2, 1]` against an output batch of `[2, 3]` holds 2
/// matrices where the output needs 6, so neither stride is correct.
///
/// Extents that are not statically 1 count as carrying batches and output
/// extents that are not statically 1 count as broadcast targets, so an unknown
/// extent is never assumed away.
bool isSingleStrideBatchLayout(ArrayRef<int64_t> batch,
                               ArrayRef<int64_t> outputBatch) {
  // Guaranteed by getBroadcastedShape, and load-bearing: the unsigned `pad`
  // below would wrap and silently report "representable" if it were violated.
  assert(batch.size() <= outputBatch.size() &&
         "operand batch cannot outrank the broadcasted output batch");
  size_t pad = outputBatch.size() - batch.size();
  bool broadcastsUp = false;
  bool carriesBatches = false;
  for (size_t i : llvm::seq<size_t>(0, outputBatch.size())) {
    // Axes below `pad` are the implicit leading ones of right-alignment.
    int64_t extent = i < pad ? 1 : batch[i - pad];
    if (extent == 1)
      broadcastsUp |= outputBatch[i] != 1;
    else
      carriesBatches = true;
  }
  return !(broadcastsUp && carriesBatches);
}

} // namespace

LogicalResult mlir::hip::verifyStridedBatchMatmul(
    ArrayRef<int64_t> aShape, ArrayRef<int64_t> bShape,
    function_ref<InFlightDiagnostic()> emitError) {
  if (aShape.size() < 2 || bShape.size() < 2) {
    emitError() << "strided-batch matmul requires rank >= 2 operands";
    return failure();
  }

  ArrayRef<int64_t> aBatch = aShape.drop_back(2);
  ArrayRef<int64_t> bBatch = bShape.drop_back(2);
  SmallVector<int64_t> outputBatch;
  if (!OpTrait::util::getBroadcastedShape(aBatch, bBatch, outputBatch)) {
    emitError() << "matmul batch broadcast failure: A.batch="
                << detail::formatShape(aBatch)
                << " B.batch=" << detail::formatShape(bBatch);
    return failure();
  }

  if (!isSingleStrideBatchLayout(aBatch, outputBatch) ||
      !isSingleStrideBatchLayout(bBatch, outputBatch)) {
    emitError() << "matmul partial per-axis batch broadcast is not supported "
                   "by the strided-batch runtime: A.batch="
                << detail::formatShape(aBatch)
                << " B.batch=" << detail::formatShape(bBatch);
    return failure();
  }

  return success();
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
  // ONNX Gemm broadcasts C onto `{M, N}` unidirectionally: C never widens the
  // result, so every C extent must be 1 or equal to the output extent.
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

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyMatmulResultShape(
    OpBuilder &b, Location loc, Value A, Value B,
    function_ref<InFlightDiagnostic()> emitError) {
  auto aType = dyn_cast<RankedTensorType>(A.getType());
  auto bType = dyn_cast<RankedTensorType>(B.getType());
  if (!aType || !bType) {
    emitError() << "matmul operands must be ranked tensors";
    return failure();
  }
  // Validate before emitting any `tensor.dim`, so a failure leaves the IR
  // unchanged (see the contract in HipShapeUtils.h). Strided-batch
  // representability is a backend capability rather than a shape rule, so it
  // stays with `MatmulOp::verify` and the converter.
  if (failed(inferMatmulShape(aType.getShape(), bType.getShape(), emitError)))
    return failure();

  SmallVector<OpFoldResult> aSizes = tensor::getMixedSizes(b, loc, A);
  SmallVector<OpFoldResult> bSizes = tensor::getMixedSizes(b, loc, B);
  ArrayRef<OpFoldResult> aBatch = ArrayRef<OpFoldResult>(aSizes).drop_back(2);
  ArrayRef<OpFoldResult> bBatch = ArrayRef<OpFoldResult>(bSizes).drop_back(2);
  SmallVector<SmallVector<OpFoldResult>> batchShapes = {
      SmallVector<OpFoldResult>(aBatch.begin(), aBatch.end()),
      SmallVector<OpFoldResult>(bBatch.begin(), bBatch.end())};
  FailureOr<SmallVector<OpFoldResult>> result =
      detail::reifyBroadcastShape(b, loc, batchShapes, emitError);
  if (failed(result))
    return failure();
  result->push_back(aSizes[aSizes.size() - 2]);
  result->push_back(bSizes.back());
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
  // Validate before emitting any `tensor.dim`, so a failure leaves the IR
  // unchanged (see the contract in HipShapeUtils.h).
  if (failed(inferGemmShape(aType.getShape(), bType.getShape(), cShape, transA,
                            transB, emitError)))
    return failure();

  // C is validated above but never contributes an extent: Gemm's result is
  // exactly `{M, N}` from the transpose-aware A and B dimensions.
  SmallVector<OpFoldResult> aSizes = tensor::getMixedSizes(b, loc, A);
  SmallVector<OpFoldResult> bSizes = tensor::getMixedSizes(b, loc, B);
  return SmallVector<OpFoldResult>{aSizes[transA ? 1 : 0],
                                   bSizes[transB ? 0 : 1]};
}
