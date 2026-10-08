/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsShapeOps.cpp - Shape-transforming operation helpers --===//
//
// Category implementation for the public shape helpers declared in
// `hip/Dialect/IR/HipShapeUtils.h`.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipShapeUtilsShapeOps.h"
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

FailureOr<SmallVector<int64_t>> mlir::hip::inferSizeShape() {
  return SmallVector<int64_t>{};
}

namespace {

struct PadShapeInference {
  SmallVector<int64_t> result;
  SmallVector<int64_t> offsets;
};

FailureOr<PadShapeInference>
inferPadShapeAndOffsets(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> pads,
                        std::optional<ArrayRef<int64_t>> axes) {
  int64_t dataRank = dataShape.size();
  SmallVector<int64_t> normalizedAxes =
      axes ? SmallVector<int64_t>(axes->begin(), axes->end())
           : llvm::to_vector(llvm::seq<int64_t>(0, dataRank));
  if (pads.size() != 2 * normalizedAxes.size())
    return failure();

  SmallVector<int64_t> offsets(dataRank, 0);
  llvm::SmallBitVector seen(dataRank);
  int64_t numAxes = normalizedAxes.size();
  for (int64_t i : llvm::seq<int64_t>(0, numAxes)) {
    int64_t axis = normalizedAxes[i];
    if (axis < 0)
      axis += dataRank;
    if (axis < 0 || axis >= dataRank || seen.test(axis))
      return failure();
    seen.set(axis);

    APInt offset = APInt(128, pads[i], /*isSigned=*/true) +
                   APInt(128, pads[i + numAxes], /*isSigned=*/true);
    if (!offset.isSignedIntN(64))
      return failure();
    offsets[axis] = offset.getSExtValue();
  }

  SmallVector<int64_t> result(dataShape.begin(), dataShape.end());
  for (int64_t axis : llvm::seq<int64_t>(0, dataRank)) {
    if (ShapedType::isDynamic(dataShape[axis]))
      continue;
    APInt extent = APInt(128, dataShape[axis], /*isSigned=*/true) +
                   APInt(128, offsets[axis], /*isSigned=*/true);
    if (!extent.isSignedIntN(64) || extent.isNegative())
      return failure();
    result[axis] = extent.getSExtValue();
  }
  return PadShapeInference{std::move(result), std::move(offsets)};
}

} // namespace

FailureOr<SmallVector<int64_t>>
mlir::hip::inferPadShape(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> pads,
                         std::optional<ArrayRef<int64_t>> axes) {
  FailureOr<PadShapeInference> inferred =
      inferPadShapeAndOffsets(dataShape, pads, axes);
  if (failed(inferred))
    return failure();
  return std::move(inferred->result);
}

LogicalResult
mlir::hip::reifyPadShape(OpBuilder &b, Location loc, Value data, Value pads,
                         Value axes,
                         std::optional<ArrayRef<int64_t>> staticPads,
                         std::optional<ArrayRef<int64_t>> staticAxes,
                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();

  SmallVector<int64_t> extractedPads;
  if (!staticPads) {
    if (!matchConstantIntTensor(pads, extractedPads, /*expectedRank=*/1))
      return failure();
    staticPads = extractedPads;
  }

  SmallVector<int64_t> extractedAxes;
  std::optional<ArrayRef<int64_t>> resolvedAxes = staticAxes;
  if (!resolvedAxes && axes) {
    if (!matchConstantIntTensor(axes, extractedAxes, /*expectedRank=*/1))
      return failure();
    resolvedAxes = extractedAxes;
  }

  // Validate every payload-independent precondition before materializing the
  // first tensor.dim or arithmetic op, and retain the normalized axis mapping
  // for the mixed-shape construction below.
  FailureOr<PadShapeInference> inferred =
      inferPadShapeAndOffsets(dataType.getShape(), *staticPads, resolvedAxes);
  if (failed(inferred))
    return failure();

  int64_t dataRank = dataType.getRank();
  SmallVector<OpFoldResult> inputSizes = tensor::getMixedSizes(b, loc, data);
  out.reserve(dataRank);
  for (int64_t axis : llvm::seq<int64_t>(0, dataRank)) {
    FailureOr<OpFoldResult> extent = detail::scaleAndOffsetDim(
        b, loc, inputSizes[axis], /*scale=*/1, inferred->offsets[axis]);
    if (failed(extent))
      return failure();
    out.push_back(*extent);
  }
  return success();
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferTileShape(ArrayRef<int64_t> inputShape,
                          ArrayRef<int64_t> repeats) {
  if (inputShape.size() != repeats.size())
    return failure();
  SmallVector<int64_t> result;
  result.reserve(inputShape.size());
  for (auto [dim, repeat] : llvm::zip_equal(inputShape, repeats)) {
    if (repeat < 0)
      return failure();
    if (ShapedType::isDynamic(dim)) {
      result.push_back(ShapedType::kDynamic);
      continue;
    }
    APInt extent = APInt(128, dim, /*isSigned=*/true) *
                   APInt(128, repeat, /*isSigned=*/true);
    if (!extent.isSignedIntN(64) || extent.isNegative())
      return failure();
    result.push_back(extent.getSExtValue());
  }
  return result;
}

LogicalResult
mlir::hip::reifyTileShape(OpBuilder &b, Location loc, Value input,
                          Value repeats,
                          std::optional<ArrayRef<int64_t>> staticRepeats,
                          SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();

  SmallVector<int64_t> extractedRepeats;
  if (!staticRepeats) {
    if (!matchConstantIntTensor(repeats, extractedRepeats,
                                /*expectedRank=*/1))
      return failure();
    staticRepeats = extractedRepeats;
  }
  if (failed(inferTileShape(inputType.getShape(), *staticRepeats)))
    return failure();

  SmallVector<OpFoldResult> inputSizes = tensor::getMixedSizes(b, loc, input);
  out.reserve(inputSizes.size());
  for (auto [dim, repeat] : llvm::zip_equal(inputSizes, *staticRepeats)) {
    FailureOr<OpFoldResult> extent =
        detail::scaleAndOffsetDim(b, loc, dim, repeat, /*offset=*/0);
    if (failed(extent))
      return failure();
    out.push_back(*extent);
  }
  return success();
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferExpandShape(ArrayRef<int64_t> inputShape,
                            ArrayRef<int64_t> targetShape) {
  SmallVector<int64_t> result;
  if (!OpTrait::util::getBroadcastedShape(inputShape, targetShape, result))
    return failure();
  return result;
}

LogicalResult
mlir::hip::reifyExpandShape(OpBuilder &b, Location loc, Value input,
                            Value shape, SmallVectorImpl<OpFoldResult> &out,
                            std::optional<ArrayRef<int64_t>> staticShape) {
  out.clear();
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType)
    return failure();

  SmallVector<int64_t> shapeVals;
  if (staticShape)
    shapeVals.assign(staticShape->begin(), staticShape->end());
  else if (!matchConstantIntTensor(shape, shapeVals, /*expectedRank=*/1))
    return failure();

  FailureOr<SmallVector<int64_t>> outShape =
      inferExpandShape(inputType.getShape(), shapeVals);
  if (failed(outShape))
    return failure();

  // Fold-or-bail: any dynamic in the broadcast result means we can't
  // produce a tight shape; let the Tier-2 fallback lift from outs.
  for (int64_t d : *outShape)
    if (ShapedType::isDynamic(d))
      return failure();

  out.reserve(outShape->size());
  for (int64_t v : *outShape)
    out.push_back(b.getIndexAttr(v));
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

LogicalResult
mlir::hip::reifyRangeShape(OpBuilder &b, Location loc, Value start, Value limit,
                           Value delta, SmallVectorImpl<OpFoldResult> &out,
                           std::optional<ArrayRef<int64_t>> staticValues) {
  out.clear();

  // Each operand is a rank-0 (scalar) integer tensor. matchConstantIntTensor
  // returns a 1-element vector for the rank-0 / IntegerAttr case.
  SmallVector<int64_t> sList, lList, dList;
  if (staticValues) {
    if (staticValues->size() != 3)
      return failure();
    sList.push_back((*staticValues)[0]);
    lList.push_back((*staticValues)[1]);
    dList.push_back((*staticValues)[2]);
  } else if (!matchConstantIntTensor(start, sList, /*expectedRank=*/0) ||
             !matchConstantIntTensor(limit, lList, /*expectedRank=*/0) ||
             !matchConstantIntTensor(delta, dList, /*expectedRank=*/0)) {
    return failure();
  }
  if (sList.size() != 1 || lList.size() != 1 || dList.size() != 1)
    return failure();
  APInt s(128, sList[0], /*isSigned=*/true);
  APInt l(128, lList[0], /*isSigned=*/true);
  APInt d(128, dList[0], /*isSigned=*/true);
  if (d.isZero())
    return failure();

  // ONNX Range: count = max(0, ceil((limit - start) / delta)) for the
  // direction implied by sign(delta). Negative direction (delta < 0)
  // counts down from start to limit. Wide arithmetic is required for
  // INT64_MIN negation and opposite-sign endpoint subtraction.
  APInt count(128, 0, /*isSigned=*/true);
  if ((d.isStrictlyPositive() && l.sgt(s)) || (d.isNegative() && l.slt(s))) {
    APInt diff = l - s;
    APInt step = d;
    if (step.isNegative()) {
      diff = -diff;
      step = -step;
    }
    APInt one(128, 1, /*isSigned=*/true);
    count = (diff + step - one).sdiv(step);
  }
  if (!count.isSignedIntN(64) || count.isNegative())
    return failure();

  out.push_back(b.getIndexAttr(count.getSExtValue()));
  return success();
}
