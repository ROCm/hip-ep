/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsReduction.cpp - Reduction shape helpers --------------===//

#include "hip/Dialect/IR/HipShapeUtils.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallSet.h"

using namespace mlir;

LogicalResult mlir::hip::reifyReductionWithKeepdims(
    OpBuilder &b, Location loc, Value data, Value axes, int64_t keepdims,
    int64_t noopWithEmptyAxes, SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataType.getRank();

  SmallVector<int64_t> axesList;
  if (!matchConstantIntTensor(axes, axesList))
    return failure();

  if (axesList.empty()) {
    if (noopWithEmptyAxes != 0) {
      out.reserve(dataRank);
      for (int64_t i : llvm::seq<int64_t>(0, dataRank))
        out.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
      return success();
    }
    if (keepdims) {
      out.append(dataRank, b.getIndexAttr(1));
      return success();
    }
    return success();
  }

  llvm::SmallSet<int64_t, 8> reducedSet;
  for (int64_t axis : axesList) {
    if (axis < 0)
      axis += dataRank;
    if (axis < 0 || axis >= dataRank)
      return failure();
    reducedSet.insert(axis);
  }

  out.reserve(dataRank);
  for (int64_t i : llvm::seq<int64_t>(0, dataRank)) {
    if (reducedSet.contains(i)) {
      if (keepdims)
        out.push_back(b.getIndexAttr(1));
    } else {
      out.push_back(reifyDimOrConstant(b, loc, dataShape[i], data, i));
    }
  }
  return success();
}

LogicalResult
mlir::hip::reifyReductionShape(OpBuilder &b, Location loc, Value data,
                               Value axes, int64_t keepdims,
                               int64_t noopWithEmptyAxes, Operation *op,
                               ReifiedRankedShapedTypeDims &reified) {
  if (op->getNumResults() == 0)
    return failure();
  if (!isa<RankedTensorType>(data.getType()))
    return failure();

  SmallVector<OpFoldResult> dims;
  if (succeeded(reifyReductionWithKeepdims(b, loc, data, axes, keepdims,
                                           noopWithEmptyAxes, dims))) {
    reified.assign({std::move(dims)});
    return success();
  }
  return cast<HipDpsOp>(op).reifyResultShapes(b, reified);
}
