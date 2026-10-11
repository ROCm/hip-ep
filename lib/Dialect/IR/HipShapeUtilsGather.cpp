/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsGather.cpp - Transpose and gather shape helpers -------===//
//
// Category implementation for the public shape helpers declared in
// `hip/Dialect/IR/HipShapeUtilsGather.h`.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipShapeUtilsGather.h"
#include "HipShapeUtilsInternal.h"

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
mlir::hip::inferTransposeShape(ArrayRef<int64_t> inputShape,
                               ArrayRef<int64_t> perm) {
  if (failed(detail::validatePermutation(perm, inputShape.size())))
    return failure();
  SmallVector<int64_t> result;
  result.reserve(perm.size());
  for (int64_t dim : perm)
    result.push_back(inputShape[dim]);
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyTransposeByPerm(OpBuilder &b, Location loc, Value input,
                                ArrayRef<int64_t> perm) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  // Validate the entire permutation before materializing any tensor.dim. A
  // failure must leave the IR unchanged.
  if (failed(inferTransposeShape(inputShape, perm)))
    return failure();

  SmallVector<OpFoldResult> dims;
  dims.reserve(perm.size());
  for (int64_t dim : perm)
    dims.push_back(reifyDimOrConstant(b, loc, inputShape[dim], input, dim));
  return dims;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferGatherShape(ArrayRef<int64_t> dataShape,
                            ArrayRef<int64_t> indicesShape, int64_t axis) {
  int64_t dataRank = dataShape.size();
  if (axis < 0)
    axis += dataRank;
  if (axis < 0 || axis >= dataRank)
    return failure();

  SmallVector<int64_t> result;
  result.reserve(dataRank - 1 + indicesShape.size());
  llvm::append_range(result, dataShape.take_front(axis));
  llvm::append_range(result, indicesShape);
  llvm::append_range(result, dataShape.drop_front(axis + 1));
  return result;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferOneHotShape(ArrayRef<int64_t> indicesShape,
                            std::optional<int64_t> depth, int64_t axis) {
  int64_t outputRank = static_cast<int64_t>(indicesShape.size()) + 1;
  if (axis < 0)
    axis += outputRank;
  if (axis < 0 || axis >= outputRank || (depth && *depth < 0))
    return failure();

  SmallVector<int64_t> result(indicesShape.begin(), indicesShape.end());
  result.insert(result.begin() + axis, depth.value_or(ShapedType::kDynamic));
  return result;
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
  if (failed(
          inferGatherShape(dataType.getShape(), indicesType.getShape(), axis)))
    return failure();
  // Negative-axis normalization (ONNX convention).
  if (axis < 0)
    axis += dataRank;

  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();

  // Output = data.shape[:axis] ++ indices.shape ++ data.shape[axis+1:].
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

FailureOr<SmallVector<int64_t>>
mlir::hip::inferGatherNDShape(ArrayRef<int64_t> dataShape,
                              ArrayRef<int64_t> indicesShape,
                              int64_t batchDims) {
  int64_t dataRank = dataShape.size();
  int64_t indicesRank = indicesShape.size();
  if (indicesRank < 1 || batchDims < 0 || batchDims > indicesRank - 1)
    return failure();

  int64_t tupleWidth = indicesShape.back();
  if (ShapedType::isDynamic(tupleWidth) || tupleWidth < 0 ||
      tupleWidth > dataRank - batchDims)
    return failure();
  for (int64_t dim : llvm::seq<int64_t>(0, batchDims)) {
    int64_t dataDim = dataShape[dim];
    int64_t indicesDim = indicesShape[dim];
    if (!ShapedType::isDynamic(dataDim) && !ShapedType::isDynamic(indicesDim) &&
        dataDim != indicesDim)
      return failure();
  }

  SmallVector<int64_t> result;
  result.reserve(indicesRank - 1 + dataRank - tupleWidth - batchDims);
  llvm::append_range(result, dataShape.take_front(batchDims));
  llvm::append_range(
      result, indicesShape.slice(batchDims, indicesRank - 1 - batchDims));
  llvm::append_range(result, dataShape.drop_front(batchDims + tupleWidth));
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyGatherND(OpBuilder &b, Location loc, Value data, Value indices,
                         int64_t batchDims) {
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  auto indicesType = dyn_cast<RankedTensorType>(indices.getType());
  if (!dataType || !indicesType || !indicesType.getElementType().isInteger(64))
    return failure();
  int64_t dataRank = dataType.getRank();
  int64_t indicesRank = indicesType.getRank();
  if (indicesRank < 1)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  ArrayRef<int64_t> indicesShape = indicesType.getShape();

  // Validate every static precondition before materializing dimensions.
  if (failed(inferGatherNDShape(dataShape, indicesShape, batchDims)))
    return failure();
  int64_t tupleWidth = indicesShape[indicesRank - 1];

  // Output = data.shape[:batch_dims] (the shared batch prefix) ++
  //          indices.shape[batch_dims:-1] (the gathered tuple count) ++
  //          data.shape[batch_dims + tupleWidth:] (the per-element slice).
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
