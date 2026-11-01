/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsGather.cpp - Gather-family shape helpers -------------===//

#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/Sequence.h"

using namespace mlir;

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                                ArrayRef<int64_t> perm) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t rank = inputType.getRank();
  if (static_cast<int64_t>(perm.size()) != rank)
    return failure();

  // Validate the complete permutation before materializing any tensor.dim.
  SmallVector<bool> seen(rank, false);
  for (int64_t permutedDim : perm) {
    if (permutedDim < 0 || permutedDim >= rank || seen[permutedDim])
      return failure();
    seen[permutedDim] = true;
  }

  SmallVector<OpFoldResult> dims;
  dims.reserve(perm.size());
  for (int64_t permutedDim : perm)
    dims.push_back(reifyDimOrConstant(b, loc, inputShape[permutedDim], input,
                                      permutedDim));
  return dims;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGatherWithAxis(OpBuilder &b, Location loc, Value data,
                               Value indices, int64_t axis) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType)
    return failure();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  if (axis < 0)
    axis += dataRank;
  if (axis < 0 || axis >= dataRank)
    return failure();

  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  SmallVector<OpFoldResult> dims;
  dims.reserve(dataRank - 1 + indicesRank);
  for (int64_t i : llvm::seq<int64_t>(0, axis))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  for (int64_t i : llvm::seq<int64_t>(0, indicesRank))
    dims.push_back(reifyDimOrConstant(b, loc, indicesShape[i], indices, i));
  for (int64_t i : llvm::seq<int64_t>(axis + 1, dataRank))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  return dims;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGatherND(OpBuilder &b, Location loc, Value data, Value indices,
                         int64_t batchDims) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType)
    return failure();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  if (indicesRank < 1)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();
  int64_t tupleWidth = indicesShape[indicesRank - 1];
  if (ShapedType::isDynamic(tupleWidth))
    return failure();
  if (batchDims < 0 || batchDims > indicesRank - 1 ||
      batchDims + tupleWidth > dataRank)
    return failure();

  SmallVector<OpFoldResult> dims;
  dims.reserve(batchDims + (indicesRank - 1 - batchDims) +
               (dataRank - batchDims - tupleWidth));
  for (int64_t i : llvm::seq<int64_t>(0, batchDims))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  for (int64_t i : llvm::seq<int64_t>(batchDims, indicesRank - 1))
    dims.push_back(reifyDimOrConstant(b, loc, indicesShape[i], indices, i));
  for (int64_t i : llvm::seq<int64_t>(batchDims + tupleWidth, dataRank))
    dims.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
  return dims;
}
