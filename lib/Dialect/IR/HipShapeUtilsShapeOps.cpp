/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsShapeOps.cpp - Value-driven shape helpers ------------===//

#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Traits.h"
#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;

LogicalResult mlir::hip::reifyPadShape(OpBuilder &b, Location loc, Value data,
                                       Value pads, Value axes,
                                       SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataType.getRank();

  SmallVector<int64_t> padsList;
  if (!matchConstantIntTensor(pads, padsList))
    return failure();

  SmallVector<int64_t> axesList;
  if (axes) {
    if (!matchConstantIntTensor(axes, axesList))
      return failure();
    for (int64_t &axis : axesList) {
      if (axis < 0)
        axis += dataRank;
      if (axis < 0 || axis >= dataRank)
        return failure();
    }
  } else {
    llvm::append_range(axesList, llvm::seq<int64_t>(0, dataRank));
  }
  if (static_cast<int64_t>(padsList.size()) !=
      2 * static_cast<int64_t>(axesList.size()))
    return failure();

  SmallVector<std::pair<int64_t, int64_t>> perAxis(dataRank, {0, 0});
  int64_t numAxes = axesList.size();
  for (int64_t i : llvm::seq<int64_t>(0, numAxes))
    perAxis[axesList[i]] = {padsList[i], padsList[i + numAxes]};

  SmallVector<int64_t> outShape;
  outShape.reserve(dataRank);
  for (int64_t dim : llvm::seq<int64_t>(0, dataRank)) {
    if (ShapedType::isDynamic(dataShape[dim]))
      return failure();
    outShape.push_back(dataShape[dim] + perAxis[dim].first +
                       perAxis[dim].second);
  }
  for (int64_t extent : outShape)
    out.push_back(b.getIndexAttr(extent));
  return success();
}

LogicalResult mlir::hip::reifyTileShape(OpBuilder &b, Location loc, Value input,
                                        Value repeats,
                                        SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();
  ArrayRef<int64_t> inputShape = inputType.getShape();
  int64_t inputRank = inputType.getRank();

  SmallVector<int64_t> repeatsList;
  if (!matchConstantIntTensor(repeats, repeatsList))
    return failure();
  if (static_cast<int64_t>(repeatsList.size()) != inputRank)
    return failure();

  out.reserve(inputRank);
  for (int64_t dim : llvm::seq<int64_t>(0, inputRank)) {
    if (ShapedType::isDynamic(inputShape[dim]) || repeatsList[dim] < 0)
      return failure();
    out.push_back(b.getIndexAttr(inputShape[dim] * repeatsList[dim]));
  }
  return success();
}

LogicalResult mlir::hip::reifyExpandShape(OpBuilder &b, Location loc,
                                          Value input, Value shape,
                                          SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();

  SmallVector<int64_t> shapeVals;
  if (!matchConstantIntTensor(shape, shapeVals))
    return failure();
  SmallVector<int64_t> outShape;
  if (!OpTrait::util::getBroadcastedShape(inputType.getShape(), shapeVals,
                                          outShape))
    return failure();
  for (int64_t extent : outShape)
    if (ShapedType::isDynamic(extent))
      return failure();
  for (int64_t extent : outShape)
    out.push_back(b.getIndexAttr(extent));
  return success();
}

LogicalResult mlir::hip::reifySliceShape(OpBuilder &b, Location loc, Value data,
                                         Value starts, Value ends, Value axes,
                                         Value steps,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();
  ArrayRef<int64_t> dataShape = dataType.getShape();
  int64_t dataRank = dataType.getRank();

  SmallVector<int64_t> startsList, endsList, axesList, stepsList;
  if (!matchConstantIntTensor(starts, startsList) ||
      !matchConstantIntTensor(ends, endsList))
    return failure();
  if (axes) {
    if (!matchConstantIntTensor(axes, axesList))
      return failure();
  } else {
    llvm::append_range(axesList, llvm::seq<int64_t>(0, dataRank));
  }
  if (steps) {
    if (!matchConstantIntTensor(steps, stepsList))
      return failure();
  } else {
    stepsList.assign(axesList.size(), 1);
  }
  if (startsList.size() != axesList.size() ||
      endsList.size() != axesList.size() || stepsList.size() != axesList.size())
    return failure();

  SmallVector<int64_t> outShape(dataShape.begin(), dataShape.end());
  for (size_t i : llvm::seq<size_t>(0, axesList.size())) {
    int64_t axis = axesList[i];
    if (axis < 0)
      axis += dataRank;
    if (axis < 0 || axis >= dataRank)
      return failure();
    int64_t dim = dataShape[axis];
    if (ShapedType::isDynamic(dim))
      return failure();
    int64_t step = stepsList[i];
    if (step == 0)
      return failure();
    int64_t start = startsList[i];
    int64_t end = endsList[i];
    if (start < 0)
      start += dim;
    if (end < 0)
      end += dim;
    if (step > 0) {
      start = std::clamp<int64_t>(start, 0, dim);
      end = std::clamp<int64_t>(end, 0, dim);
      outShape[axis] = end > start ? (end - start + step - 1) / step : 0;
    } else {
      start = std::clamp<int64_t>(start, 0, dim - 1);
      end = std::clamp<int64_t>(end, -1, dim - 1);
      int64_t span = start - end;
      int64_t magnitude = -step;
      outShape[axis] = start > end ? (span + magnitude - 1) / magnitude : 0;
    }
  }

  for (int64_t extent : outShape)
    if (ShapedType::isDynamic(extent))
      return failure();
  for (int64_t extent : outShape)
    out.push_back(b.getIndexAttr(extent));
  return success();
}

LogicalResult mlir::hip::reifyRangeShape(OpBuilder &b, Location loc,
                                         Value start, Value limit, Value delta,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  SmallVector<int64_t> starts, limits, deltas;
  if (!matchConstantIntTensor(start, starts) ||
      !matchConstantIntTensor(limit, limits) ||
      !matchConstantIntTensor(delta, deltas))
    return failure();
  if (starts.size() != 1 || limits.size() != 1 || deltas.size() != 1)
    return failure();

  int64_t startValue = starts[0];
  int64_t limitValue = limits[0];
  int64_t deltaValue = deltas[0];
  if (deltaValue == 0)
    return failure();
  int64_t count = 0;
  if ((deltaValue > 0 && limitValue > startValue) ||
      (deltaValue < 0 && limitValue < startValue)) {
    int64_t difference = limitValue - startValue;
    int64_t step = deltaValue;
    if (step < 0) {
      difference = -difference;
      step = -step;
    }
    count = (difference + step - 1) / step;
  }
  out.push_back(b.getIndexAttr(count));
  return success();
}
