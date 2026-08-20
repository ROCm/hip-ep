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

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"
#include "hip/Support/SliceUtils.h"

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
  if (llvm::any_of(targetShape, [](int64_t extent) { return extent < 0; }))
    return failure();
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

namespace {

struct SliceShapeInference {
  SmallVector<int64_t> result;
  SmallVector<int64_t> axes;
  SmallVector<int64_t> steps;
};

FailureOr<SliceShapeInference>
inferSliceShapeAndParams(ArrayRef<int64_t> dataShape, ArrayRef<int64_t> starts,
                         ArrayRef<int64_t> ends,
                         std::optional<ArrayRef<int64_t>> axes,
                         std::optional<ArrayRef<int64_t>> steps) {
  int64_t dataRank = dataShape.size();
  if (starts.size() != ends.size())
    return failure();
  SmallVector<int64_t> resolvedAxes =
      axes ? SmallVector<int64_t>(axes->begin(), axes->end())
           : llvm::to_vector(
                 llvm::seq<int64_t>(0, static_cast<int64_t>(starts.size())));
  SmallVector<int64_t> resolvedSteps =
      steps ? SmallVector<int64_t>(steps->begin(), steps->end())
            : SmallVector<int64_t>(resolvedAxes.size(), 1);
  if (starts.size() != resolvedAxes.size() ||
      ends.size() != resolvedAxes.size() ||
      resolvedSteps.size() != resolvedAxes.size())
    return failure();

  std::vector<int64_t> normalizedAxes;
  if (!hipdnn_ep::slice::normalizeAxes(
          dataRank, resolvedAxes.data(),
          static_cast<int64_t>(resolvedAxes.size()), normalizedAxes))
    return failure();
  resolvedAxes.assign(normalizedAxes.begin(), normalizedAxes.end());

  SmallVector<int64_t> result(dataShape.begin(), dataShape.end());
  for (size_t i : llvm::seq<size_t>(0, resolvedAxes.size())) {
    int64_t axis = resolvedAxes[i];
    if (resolvedSteps[i] == 0)
      return failure();
    int64_t dim = dataShape[axis];
    if (ShapedType::isDynamic(dim)) {
      result[axis] = ShapedType::kDynamic;
      continue;
    }
    int64_t normalizedStart = 0;
    int64_t output = 0;
    if (!hipdnn_ep::slice::normalizeAxis(
            dim, starts[i], ends[i], resolvedSteps[i], normalizedStart, output))
      return failure();
    result[axis] = output;
  }
  return SliceShapeInference{std::move(result), std::move(resolvedAxes),
                             std::move(resolvedSteps)};
}

} // namespace

FailureOr<SmallVector<int64_t>>
mlir::hip::inferSliceShape(ArrayRef<int64_t> dataShape,
                           ArrayRef<int64_t> starts, ArrayRef<int64_t> ends,
                           std::optional<ArrayRef<int64_t>> axes,
                           std::optional<ArrayRef<int64_t>> steps) {
  FailureOr<SliceShapeInference> inferred =
      inferSliceShapeAndParams(dataShape, starts, ends, axes, steps);
  if (failed(inferred))
    return failure();
  return std::move(inferred->result);
}

LogicalResult mlir::hip::materializeSliceParameters(
    OpBuilder &b, Location loc, Value data, ArrayRef<Value> starts,
    ArrayRef<Value> ends, std::optional<ArrayRef<Value>> axes,
    std::optional<ArrayRef<Value>> steps, Value readbackValid,
    MaterializedSliceParameters &out) {
  out = {};
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType || starts.size() != ends.size() ||
      (axes && axes->size() != starts.size()) ||
      (steps && steps->size() != starts.size()) || !readbackValid ||
      !readbackValid.getType().isInteger(1))
    return failure();
  auto allI64 = [](ArrayRef<Value> values) {
    return llvm::all_of(
        values, [](Value value) { return value.getType().isInteger(64); });
  };
  if (!allI64(starts) || !allI64(ends) || (axes && !allI64(*axes)) ||
      (steps && !allI64(*steps)))
    return failure();

  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantOp::create(b, loc, b.getI64Type(),
                                     b.getI64IntegerAttr(value));
  };
  auto boolConstant = [&](bool value) -> Value {
    return arith::ConstantOp::create(b, loc, b.getI1Type(),
                                     b.getBoolAttr(value));
  };
  auto cmp = [&](arith::CmpIPredicate predicate, Value lhs,
                 Value rhs) -> Value {
    return arith::CmpIOp::create(b, loc, predicate, lhs, rhs);
  };
  auto andI1 = [&](Value lhs, Value rhs) -> Value {
    return arith::AndIOp::create(b, loc, lhs, rhs);
  };
  auto select = [&](Value condition, Value trueValue,
                    Value falseValue) -> Value {
    return arith::SelectOp::create(b, loc, condition, trueValue, falseValue);
  };
  auto clamp = [&](Value value, Value low, Value high) -> Value {
    Value atLeastLow = arith::MaxSIOp::create(b, loc, value, low);
    return arith::MinSIOp::create(b, loc, atLeastLow, high);
  };

  int64_t rank = dataType.getRank();
  size_t count = starts.size();
  Value zero = constant(0);
  Value one = constant(1);
  Value minusOne = constant(-1);
  Value rankValue = constant(rank);
  Value paramsValid = readbackValid;

  SmallVector<Value> resolvedAxes;
  resolvedAxes.reserve(count);
  if (axes) {
    resolvedAxes.append(axes->begin(), axes->end());
  } else {
    for (size_t i : llvm::seq<size_t>(count))
      resolvedAxes.push_back(constant(static_cast<int64_t>(i)));
  }
  SmallVector<Value> resolvedSteps;
  resolvedSteps.reserve(count);
  if (steps)
    resolvedSteps.append(steps->begin(), steps->end());
  else
    resolvedSteps.assign(count, one);

  if (rank == 0) {
    if (count != 0)
      paramsValid = andI1(paramsValid, boolConstant(false));
    out.valid = paramsValid;
    return success();
  }

  SmallVector<Value> normalizedAxes;
  SmallVector<Value> axesInRange;
  normalizedAxes.reserve(count);
  axesInRange.reserve(count);
  for (size_t i : llvm::seq<size_t>(count)) {
    Value axis = resolvedAxes[i];
    Value axisNegative = cmp(arith::CmpIPredicate::slt, axis, zero);
    Value normalized = select(
        axisNegative, arith::AddIOp::create(b, loc, axis, rankValue), axis);
    Value atLeastZero = cmp(arith::CmpIPredicate::sge, normalized, zero);
    Value belowRank = cmp(arith::CmpIPredicate::slt, normalized, rankValue);
    Value inRange = andI1(atLeastZero, belowRank);
    paramsValid = andI1(paramsValid, inRange);
    for (size_t previous : llvm::seq<size_t>(i)) {
      Value distinct =
          cmp(arith::CmpIPredicate::ne, normalized, normalizedAxes[previous]);
      paramsValid = andI1(paramsValid, distinct);
    }
    Value nonzeroStep = cmp(arith::CmpIPredicate::ne, resolvedSteps[i], zero);
    paramsValid = andI1(paramsValid, nonzeroStep);
    normalizedAxes.push_back(normalized);
    axesInRange.push_back(inRange);
  }

  SmallVector<OpFoldResult> mixedInputSizes =
      tensor::getMixedSizes(b, loc, data);
  SmallVector<Value> inputExtents;
  inputExtents.reserve(rank);
  for (OpFoldResult extent : mixedInputSizes) {
    Value indexExtent = getValueOrCreateConstantIndexOp(b, loc, extent);
    inputExtents.push_back(
        arith::IndexCastOp::create(b, loc, b.getI64Type(), indexExtent));
  }

  SmallVector<Value> candidateStarts;
  SmallVector<Value> candidateExtents;
  candidateStarts.reserve(count);
  candidateExtents.reserve(count);
  for (size_t i : llvm::seq<size_t>(count)) {
    Value safeAxis = select(axesInRange[i], normalizedAxes[i], zero);
    Value safeAxisIndex =
        arith::IndexCastOp::create(b, loc, b.getIndexType(), safeAxis);
    Value dimIndex = tensor::DimOp::create(b, loc, data, safeAxisIndex);
    Value dim = arith::IndexCastOp::create(b, loc, b.getI64Type(), dimIndex);
    Value dimIsZero = cmp(arith::CmpIPredicate::eq, dim, zero);
    Value upper = arith::SubIOp::create(b, loc, dim, one);
    Value stepPositive = cmp(arith::CmpIPredicate::sgt, resolvedSteps[i], zero);
    Value stepNegative = cmp(arith::CmpIPredicate::slt, resolvedSteps[i], zero);

    auto normalizeIndex = [&](Value raw, Value low, Value high) -> Value {
      Value negative = cmp(arith::CmpIPredicate::slt, raw, zero);
      Value adjusted =
          select(negative, arith::AddIOp::create(b, loc, raw, dim), raw);
      return clamp(adjusted, low, high);
    };
    Value positiveStart = normalizeIndex(starts[i], zero, dim);
    Value positiveEnd = normalizeIndex(ends[i], zero, dim);
    Value negativeStart = normalizeIndex(starts[i], zero, upper);
    Value negativeEnd = normalizeIndex(ends[i], minusOne, upper);
    Value normalizedStart = select(stepPositive, positiveStart, negativeStart);

    Value positiveDistance =
        arith::SubIOp::create(b, loc, positiveEnd, positiveStart);
    positiveDistance = arith::MaxSIOp::create(b, loc, positiveDistance, zero);
    Value positiveDivisor = select(stepPositive, resolvedSteps[i], one);
    Value positiveExtent =
        arith::CeilDivSIOp::create(b, loc, positiveDistance, positiveDivisor);

    Value negativeDistance =
        arith::SubIOp::create(b, loc, negativeStart, negativeEnd);
    negativeDistance = arith::MaxSIOp::create(b, loc, negativeDistance, zero);
    Value stepIsMin = cmp(arith::CmpIPredicate::eq, resolvedSteps[i],
                          constant(std::numeric_limits<int64_t>::min()));
    Value safeNegativeStep = select(stepIsMin, minusOne, resolvedSteps[i]);
    Value stepMagnitude = arith::SubIOp::create(b, loc, zero, safeNegativeStep);
    Value negativeDivisor = select(stepNegative, stepMagnitude, one);
    Value regularNegativeExtent =
        arith::CeilDivSIOp::create(b, loc, negativeDistance, negativeDivisor);
    Value hasNegativeElements =
        cmp(arith::CmpIPredicate::sgt, negativeDistance, zero);
    Value minStepExtent = select(hasNegativeElements, one, zero);
    Value negativeExtent =
        select(stepIsMin, minStepExtent, regularNegativeExtent);
    Value extent = select(stepPositive, positiveExtent, negativeExtent);

    candidateStarts.push_back(select(dimIsZero, zero, normalizedStart));
    candidateExtents.push_back(select(dimIsZero, zero, extent));
  }

  SmallVector<Value> startsPerAxis(rank, zero);
  SmallVector<Value> stepsPerAxis(rank, one);
  SmallVector<Value> extentsPerAxis(inputExtents.begin(), inputExtents.end());
  for (int64_t axis : llvm::seq<int64_t>(rank)) {
    Value axisValue = constant(axis);
    for (size_t i : llvm::seq<size_t>(count)) {
      Value matches =
          cmp(arith::CmpIPredicate::eq, normalizedAxes[i], axisValue);
      startsPerAxis[axis] =
          select(matches, candidateStarts[i], startsPerAxis[axis]);
      stepsPerAxis[axis] =
          select(matches, resolvedSteps[i], stepsPerAxis[axis]);
      extentsPerAxis[axis] =
          select(matches, candidateExtents[i], extentsPerAxis[axis]);
    }
  }

  out.valid = paramsValid;
  out.starts.reserve(rank);
  out.steps.reserve(rank);
  out.extents.reserve(rank);
  for (int64_t axis : llvm::seq<int64_t>(rank)) {
    out.starts.push_back(select(paramsValid, startsPerAxis[axis], zero));
    out.steps.push_back(select(paramsValid, stepsPerAxis[axis], one));
    out.extents.push_back(select(paramsValid, extentsPerAxis[axis], zero));
  }
  return success();
}

LogicalResult mlir::hip::reifySliceShape(OpBuilder &b, Location loc, Value data,
                                         ArrayRef<int64_t> starts,
                                         ArrayRef<int64_t> ends,
                                         std::optional<ArrayRef<int64_t>> axes,
                                         std::optional<ArrayRef<int64_t>> steps,
                                         SmallVectorImpl<OpFoldResult> &out) {
  out.clear();
  auto dataType = dyn_cast<RankedTensorType>(data.getType());
  if (!dataType)
    return failure();

  // Validate every parameter before materializing the first tensor.dim or
  // arithmetic operation.
  FailureOr<SliceShapeInference> inferred =
      inferSliceShapeAndParams(dataType.getShape(), starts, ends, axes, steps);
  if (failed(inferred))
    return failure();

  SmallVector<OpFoldResult> inputSizes = tensor::getMixedSizes(b, loc, data);
  out.assign(inputSizes.begin(), inputSizes.end());

  auto constant = [&](int64_t value) -> Value {
    return arith::ConstantIndexOp::create(b, loc, value);
  };
  auto clamp = [&](Value value, Value low, Value high) -> Value {
    Value atLeastLow = arith::MaxSIOp::create(b, loc, value, low);
    return arith::MinSIOp::create(b, loc, atLeastLow, high);
  };

  for (size_t i : llvm::seq<size_t>(0, inferred->axes.size())) {
    int64_t axis = inferred->axes[i];
    int64_t staticExtent = inferred->result[axis];
    if (!ShapedType::isDynamic(staticExtent)) {
      out[axis] = b.getIndexAttr(staticExtent);
      continue;
    }

    Value dim = getValueOrCreateConstantIndexOp(b, loc, inputSizes[axis]);
    Value zero = constant(0);
    Value one = constant(1);
    int64_t step = inferred->steps[i];

    auto normalize = [&](int64_t index, Value low, Value high) -> Value {
      Value value = constant(index);
      if (index < 0)
        value = arith::AddIOp::create(b, loc, dim, value);
      return clamp(value, low, high);
    };

    Value extent;
    if (step > 0) {
      Value start = normalize(starts[i], zero, dim);
      Value end = normalize(ends[i], zero, dim);
      Value distance = arith::SubIOp::create(b, loc, end, start);
      distance = arith::MaxSIOp::create(b, loc, distance, zero);
      extent = arith::CeilDivSIOp::create(b, loc, distance, constant(step));
    } else {
      Value minusOne = constant(-1);
      Value upper = arith::SubIOp::create(b, loc, dim, one);
      Value start = normalize(starts[i], zero, upper);
      Value end = normalize(ends[i], minusOne, upper);
      Value distance = arith::SubIOp::create(b, loc, start, end);
      distance = arith::MaxSIOp::create(b, loc, distance, zero);
      if (step == std::numeric_limits<int64_t>::min()) {
        Value hasElements = arith::CmpIOp::create(
            b, loc, arith::CmpIPredicate::sgt, distance, zero);
        extent = arith::SelectOp::create(b, loc, hasElements, one, zero);
      } else {
        extent = arith::CeilDivSIOp::create(b, loc, distance, constant(-step));
      }
    }
    out[axis] = extent;
  }
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

  SmallVector<int64_t> startsList, endsList, axesList, stepsList;
  if (!matchConstantIntTensor(starts, startsList, /*expectedRank=*/1) ||
      !matchConstantIntTensor(ends, endsList, /*expectedRank=*/1))
    return failure();
  std::optional<ArrayRef<int64_t>> resolvedAxes;
  if (axes) {
    if (!matchConstantIntTensor(axes, axesList, /*expectedRank=*/1))
      return failure();
    resolvedAxes = axesList;
  }
  std::optional<ArrayRef<int64_t>> resolvedSteps;
  if (steps) {
    if (!matchConstantIntTensor(steps, stepsList, /*expectedRank=*/1))
      return failure();
    resolvedSteps = stepsList;
  }
  return reifySliceShape(b, loc, data, startsList, endsList, resolvedAxes,
                         resolvedSteps, out);
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
