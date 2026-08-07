/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsConvPool.cpp - Convolution and pooling shapes ---------===//
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

namespace {

std::optional<int64_t> computeWindowOffset(int64_t kernel, int64_t padBegin,
                                           int64_t padEnd, int64_t dilation) {
  APInt k(128, kernel, /*isSigned=*/true);
  APInt pb(128, padBegin, /*isSigned=*/true);
  APInt pe(128, padEnd, /*isSigned=*/true);
  APInt dil(128, dilation, /*isSigned=*/true);
  APInt one(128, 1, /*isSigned=*/true);
  APInt offset = pb + pe - ((k - one) * dil + one);
  if (!offset.isSignedIntN(64))
    return std::nullopt;
  return offset.getSExtValue();
}

/// Compute one static sliding-window extent without relying on C++'s
/// truncation-toward-zero signed division. APInt also keeps malformed, huge
/// attributes from overflowing intermediate `(kernel - 1) * dilation` math.
FailureOr<int64_t>
inferStaticWindowExtent(int64_t input, int64_t kernel, int64_t stride,
                        int64_t padBegin, int64_t padEnd, int64_t dilation,
                        bool ceilMode, int64_t axis, StringRef opName,
                        function_ref<InFlightDiagnostic()> emitError) {
  APInt in(128, input, /*isSigned=*/true);
  APInt k(128, kernel, /*isSigned=*/true);
  APInt st(128, stride, /*isSigned=*/true);
  APInt pb(128, padBegin, /*isSigned=*/true);
  APInt pe(128, padEnd, /*isSigned=*/true);
  APInt dil(128, dilation, /*isSigned=*/true);
  APInt one(128, 1, /*isSigned=*/true);

  APInt effectiveKernel = (k - one) * dil + one;
  APInt numerator = in + pb + pe - effectiveKernel;
  APInt quotient = numerator.sdiv(st);
  APInt remainder = numerator.srem(st);
  if (ceilMode) {
    if (remainder.sgt(0))
      quotient += one;
  } else if (remainder.slt(0)) {
    quotient -= one;
  }
  APInt output = quotient + one;
  if (!output.isSignedIntN(64)) {
    emitError() << opName << " inferred an out-of-range output extent at "
                << "spatial axis " << axis;
    return failure();
  }
  int64_t extent = output.getSExtValue();
  if (extent < 0) {
    emitError() << opName << " inferred a negative output extent " << extent
                << " at spatial axis " << axis;
    return failure();
  }
  return extent;
}

/// Shared validated static primitive for forward convolution and pooling.
FailureOr<SmallVector<int64_t>> inferSpatialWindowExtents(
    ArrayRef<int64_t> inputSpatial, ArrayRef<int64_t> kernelShape,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> dilations, bool ceilMode, StringRef opName,
    function_ref<InFlightDiagnostic()> emitError) {
  int64_t spatialRank = inputSpatial.size();
  if (kernelShape.size() != static_cast<size_t>(spatialRank) ||
      strides.size() != static_cast<size_t>(spatialRank) ||
      pads.size() != static_cast<size_t>(2 * spatialRank) ||
      dilations.size() != static_cast<size_t>(spatialRank)) {
    emitError() << opName
                << " expects kernel_shape, strides, and dilations of length "
                << spatialRank << " and pads of length " << 2 * spatialRank;
    return failure();
  }

  SmallVector<int64_t> outputSpatial(spatialRank, ShapedType::kDynamic);
  for (int64_t axis : llvm::seq<int64_t>(0, spatialRank)) {
    if (kernelShape[axis] <= 0 || strides[axis] <= 0 || dilations[axis] <= 0) {
      emitError() << opName
                  << " kernel_shape, strides, and dilations must be positive";
      return failure();
    }
    if (pads[axis] < 0 || pads[axis + spatialRank] < 0) {
      emitError() << opName << " pads must be non-negative";
      return failure();
    }
    if (!computeWindowOffset(kernelShape[axis], pads[axis],
                             pads[axis + spatialRank], dilations[axis])) {
      emitError() << opName << " window parameters overflow index arithmetic "
                  << "at spatial axis " << axis;
      return failure();
    }
    if (ShapedType::isDynamic(inputSpatial[axis]))
      continue;
    FailureOr<int64_t> extent = inferStaticWindowExtent(
        inputSpatial[axis], kernelShape[axis], strides[axis], pads[axis],
        pads[axis + spatialRank], dilations[axis], ceilMode, axis, opName,
        emitError);
    if (failed(extent))
      return failure();
    outputSpatial[axis] = *extent;
  }
  return outputSpatial;
}

FailureOr<SmallVector<int64_t>>
resolveConvKernelShape(ArrayRef<int64_t> weightShape,
                       ArrayRef<int64_t> kernelShape,
                       function_ref<InFlightDiagnostic()> emitError) {
  int64_t spatialRank = weightShape.size() - 2;
  SmallVector<int64_t> resolved;
  if (kernelShape.empty()) {
    resolved.reserve(spatialRank);
    for (int64_t axis : llvm::seq<int64_t>(0, spatialRank)) {
      int64_t weightDim = weightShape[axis + 2];
      if (ShapedType::isDynamic(weightDim)) {
        emitError() << "conv with dynamic weight spatial dimensions requires "
                       "an explicit kernel_shape";
        return failure();
      }
      resolved.push_back(weightDim);
    }
  } else {
    resolved.assign(kernelShape.begin(), kernelShape.end());
  }

  if (resolved.size() != static_cast<size_t>(spatialRank)) {
    emitError() << "conv kernel_shape must have length " << spatialRank;
    return failure();
  }
  for (int64_t axis : llvm::seq<int64_t>(0, spatialRank)) {
    int64_t weightDim = weightShape[axis + 2];
    if (!ShapedType::isDynamic(weightDim) && weightDim != resolved[axis]) {
      emitError() << "conv kernel_shape dimension " << resolved[axis]
                  << " does not match weights spatial dimension " << weightDim
                  << " at axis " << axis;
      return failure();
    }
  }
  return resolved;
}

OpFoldResult reifyWindowExtent(OpBuilder &b, Location loc, Value input,
                               int64_t inputDim, int64_t kernel, int64_t stride,
                               int64_t padBegin, int64_t padEnd,
                               int64_t dilation, bool ceilMode) {
  OpFoldResult inputExtent = tensor::getMixedSize(b, loc, input, inputDim);
  assert(!getConstantIntValue(inputExtent) &&
         "static window extents must be returned from static inference");
  Value inputValue = getValueOrCreateConstantIndexOp(b, loc, inputExtent);
  std::optional<int64_t> offset =
      computeWindowOffset(kernel, padBegin, padEnd, dilation);
  assert(offset && "window offset must be validated before reification");
  Value numerator = inputValue;
  if (*offset != 0)
    numerator = arith::AddIOp::create(
        b, loc, numerator, arith::ConstantIndexOp::create(b, loc, *offset));
  Value strideValue = arith::ConstantIndexOp::create(b, loc, stride);
  Value quotient =
      ceilMode ? arith::CeilDivSIOp::create(b, loc, numerator, strideValue)
                     .getResult()
               : arith::FloorDivSIOp::create(b, loc, numerator, strideValue)
                     .getResult();
  return arith::AddIOp::create(b, loc, quotient,
                               arith::ConstantIndexOp::create(b, loc, 1))
      .getResult();
}

struct ConvShapeInference {
  SmallVector<int64_t> result;
  SmallVector<int64_t> resolvedKernel;
};

FailureOr<ConvShapeInference> inferConvShapeAndKernel(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations, int64_t group,
    function_ref<InFlightDiagnostic()> emitError) {
  if (inputShape.size() != weightShape.size() ||
      (inputShape.size() != 3 && inputShape.size() != 4)) {
    emitError() << "conv input and weights must have matching rank 3 or 4";
    return failure();
  }
  if (group <= 0) {
    emitError() << "conv group must be positive";
    return failure();
  }

  int64_t inputChannels = inputShape[1];
  int64_t outputChannels = weightShape[0];
  int64_t weightChannelsPerGroup = weightShape[1];
  if (!ShapedType::isDynamic(inputChannels) && inputChannels <= 0) {
    emitError() << "conv input channels must be positive";
    return failure();
  }
  if (!ShapedType::isDynamic(outputChannels) && outputChannels <= 0) {
    emitError() << "conv output channels must be positive";
    return failure();
  }
  if (!ShapedType::isDynamic(weightChannelsPerGroup) &&
      weightChannelsPerGroup <= 0) {
    emitError() << "conv weights channels per group must be positive";
    return failure();
  }
  if (!ShapedType::isDynamic(inputChannels)) {
    if (inputChannels % group != 0) {
      emitError() << "conv input channels " << inputChannels
                  << " must be divisible by group " << group;
      return failure();
    }
    if (!ShapedType::isDynamic(weightChannelsPerGroup) &&
        inputChannels / group != weightChannelsPerGroup) {
      emitError() << "conv input channels " << inputChannels
                  << " do not match weights channels-per-group "
                  << weightChannelsPerGroup << " times group " << group;
      return failure();
    }
  }
  if (!ShapedType::isDynamic(outputChannels) && outputChannels % group != 0) {
    emitError() << "conv output channels " << outputChannels
                << " must be divisible by group " << group;
    return failure();
  }

  FailureOr<SmallVector<int64_t>> resolvedKernel =
      resolveConvKernelShape(weightShape, kernelShape, emitError);
  if (failed(resolvedKernel))
    return failure();
  FailureOr<SmallVector<int64_t>> spatial = inferSpatialWindowExtents(
      inputShape.drop_front(2), *resolvedKernel, strides, pads, dilations,
      /*ceilMode=*/false, "conv", emitError);
  if (failed(spatial))
    return failure();

  SmallVector<int64_t> result = {inputShape[0], outputChannels};
  result.append(spatial->begin(), spatial->end());
  return ConvShapeInference{std::move(result), std::move(*resolvedKernel)};
}

FailureOr<SmallVector<OpFoldResult>> reifyWindowResultShape(
    OpBuilder &b, Location loc, Value input, Value channelSource,
    int64_t channelDim, ArrayRef<int64_t> inferred,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations, bool ceilMode) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto channelType = cast<RankedTensorType>(channelSource.getType());
  int64_t spatialRank = inputType.getRank() - 2;

  SmallVector<OpFoldResult> result;
  result.reserve(inputType.getRank());
  result.push_back(
      reifyDimOrConstant(b, loc, inputType.getDimSize(0), input, 0));
  result.push_back(reifyDimOrConstant(
      b, loc, channelType.getDimSize(channelDim), channelSource, channelDim));
  for (int64_t axis : llvm::seq<int64_t>(0, spatialRank)) {
    int64_t staticExtent = inferred[axis + 2];
    result.push_back(ShapedType::isDynamic(staticExtent)
                         ? reifyWindowExtent(b, loc, input, axis + 2,
                                             kernelShape[axis], strides[axis],
                                             pads[axis],
                                             pads[axis + spatialRank],
                                             dilations[axis], ceilMode)
                         : OpFoldResult(b.getIndexAttr(staticExtent)));
  }
  return result;
}

FailureOr<int64_t>
computeConvTransposeOffset(int64_t kernel, int64_t stride, int64_t padBegin,
                           int64_t padEnd, int64_t dilation,
                           int64_t outputPadding, int64_t axis,
                           function_ref<InFlightDiagnostic()> emitError) {
  APInt k(256, kernel, /*isSigned=*/true);
  APInt st(256, stride, /*isSigned=*/true);
  APInt pb(256, padBegin, /*isSigned=*/true);
  APInt pe(256, padEnd, /*isSigned=*/true);
  APInt dil(256, dilation, /*isSigned=*/true);
  APInt outPad(256, outputPadding, /*isSigned=*/true);
  APInt one(256, 1, /*isSigned=*/true);
  APInt effectiveKernel = (k - one) * dil + one;
  APInt offset = outPad + effectiveKernel - pb - pe - st;
  if (!offset.isSignedIntN(64)) {
    emitError() << "conv_transpose affine offset is out of range at spatial "
                   "axis "
                << axis;
    return failure();
  }
  return offset.getSExtValue();
}

FailureOr<int64_t>
computeScaledExtent(int64_t extent, int64_t scale, int64_t offset,
                    StringRef name, int64_t axis,
                    function_ref<InFlightDiagnostic()> emitError) {
  APInt output = APInt(256, extent, /*isSigned=*/true) *
                     APInt(256, scale, /*isSigned=*/true) +
                 APInt(256, offset, /*isSigned=*/true);
  if (!output.isSignedIntN(64)) {
    InFlightDiagnostic diagnostic = emitError();
    diagnostic << name << " inferred an out-of-range output extent";
    if (axis >= 0)
      diagnostic << " at spatial axis " << axis;
    return failure();
  }
  int64_t narrowed = output.getSExtValue();
  if (narrowed < 0) {
    InFlightDiagnostic diagnostic = emitError();
    diagnostic << name << " inferred a negative output extent " << narrowed;
    if (axis >= 0)
      diagnostic << " at spatial axis " << axis;
    return failure();
  }
  return narrowed;
}

} // namespace

FailureOr<SmallVector<int64_t>> mlir::hip::inferConvShape(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations, int64_t group,
    function_ref<InFlightDiagnostic()> emitError) {
  FailureOr<ConvShapeInference> inferred =
      inferConvShapeAndKernel(inputShape, weightShape, kernelShape, strides,
                              pads, dilations, group, emitError);
  if (failed(inferred))
    return failure();
  return std::move(inferred->result);
}

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyConvResultShape(
    OpBuilder &b, Location loc, Value input, Value weights,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations, int64_t group,
    function_ref<InFlightDiagnostic()> emitError) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightType = dyn_cast<RankedTensorType>(weights.getType());
  if (!inputType || !weightType) {
    emitError() << "conv input and weights must be ranked tensors";
    return failure();
  }
  FailureOr<ConvShapeInference> inferred = inferConvShapeAndKernel(
      inputType.getShape(), weightType.getShape(), kernelShape, strides, pads,
      dilations, group, emitError);
  if (failed(inferred))
    return failure();
  return reifyWindowResultShape(b, loc, input, weights, /*channelDim=*/0,
                                inferred->result, inferred->resolvedKernel,
                                strides, pads, dilations, /*ceilMode=*/false);
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferPoolShape(ArrayRef<int64_t> inputShape,
                          ArrayRef<int64_t> kernelShape,
                          ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
                          ArrayRef<int64_t> dilations, int64_t ceilMode,
                          function_ref<InFlightDiagnostic()> emitError) {
  if (inputShape.size() < 3 || inputShape.size() > 5) {
    emitError() << "pool input must have spatial rank 1, 2, or 3";
    return failure();
  }
  if (ceilMode != 0 && ceilMode != 1) {
    emitError() << "pool ceil_mode must be 0 or 1";
    return failure();
  }
  FailureOr<SmallVector<int64_t>> spatial = inferSpatialWindowExtents(
      inputShape.drop_front(2), kernelShape, strides, pads, dilations,
      ceilMode == 1, "pool", emitError);
  if (failed(spatial))
    return failure();
  SmallVector<int64_t> result = {inputShape[0], inputShape[1]};
  result.append(spatial->begin(), spatial->end());
  return result;
}

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyPoolResultShape(
    OpBuilder &b, Location loc, Value input, ArrayRef<int64_t> kernelShape,
    ArrayRef<int64_t> strides, ArrayRef<int64_t> pads,
    ArrayRef<int64_t> dilations, int64_t ceilMode,
    function_ref<InFlightDiagnostic()> emitError) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType) {
    emitError() << "pool input must be a ranked tensor";
    return failure();
  }
  FailureOr<SmallVector<int64_t>> inferred =
      inferPoolShape(inputType.getShape(), kernelShape, strides, pads,
                     dilations, ceilMode, emitError);
  if (failed(inferred))
    return failure();

  return reifyWindowResultShape(b, loc, input, input, /*channelDim=*/1,
                                *inferred, kernelShape, strides, pads,
                                dilations, ceilMode == 1);
}

FailureOr<SmallVector<int64_t>> mlir::hip::inferConvTransposeShape(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> outputPadding, int64_t group,
    function_ref<InFlightDiagnostic()> emitError) {
  constexpr int64_t kRank = 4;
  constexpr int64_t kSpatialRank = 2;
  if (inputShape.size() != kRank || weightShape.size() != kRank) {
    emitError() << "conv_transpose requires rank-4 NCHW input and weights";
    return failure();
  }
  if (kernelShape.size() != kSpatialRank || strides.size() != kSpatialRank ||
      pads.size() != 2 * kSpatialRank || dilations.size() != kSpatialRank ||
      outputPadding.size() != kSpatialRank) {
    emitError() << "conv_transpose expects kernel_shape/strides/dilations/"
                   "output_padding of length 2 and pads of length 4";
    return failure();
  }
  if (group <= 0) {
    emitError() << "conv_transpose group must be positive";
    return failure();
  }

  int64_t inputChannels = inputShape[1];
  int64_t weightInputChannels = weightShape[0];
  if (!ShapedType::isDynamic(inputChannels) &&
      !ShapedType::isDynamic(weightInputChannels) &&
      inputChannels != weightInputChannels) {
    emitError() << "conv_transpose input channels " << inputChannels
                << " do not match weights dimension 0 " << weightInputChannels;
    return failure();
  }

  for (int64_t i : llvm::seq<int64_t>(0, kSpatialRank)) {
    if (kernelShape[i] <= 0 || strides[i] <= 0 || dilations[i] <= 0) {
      emitError() << "conv_transpose kernel_shape, strides, and dilations "
                     "must be positive";
      return failure();
    }
    if (pads[i] < 0 || pads[i + kSpatialRank] < 0 || outputPadding[i] < 0 ||
        outputPadding[i] >= std::max(strides[i], dilations[i])) {
      emitError() << "conv_transpose pads must be non-negative and "
                     "output_padding must be in [0, max(stride, dilation))";
      return failure();
    }
  }

  SmallVector<int64_t> result(kRank, ShapedType::kDynamic);
  result[0] = inputShape[0];
  if (!ShapedType::isDynamic(weightShape[1])) {
    FailureOr<int64_t> outputChannels = computeScaledExtent(
        weightShape[1], group, /*offset=*/0, "conv_transpose output channels",
        /*axis=*/-1, emitError);
    if (failed(outputChannels))
      return failure();
    result[1] = *outputChannels;
  }
  for (int64_t i : llvm::seq<int64_t>(0, kSpatialRank)) {
    FailureOr<int64_t> offset = computeConvTransposeOffset(
        kernelShape[i], strides[i], pads[i], pads[i + kSpatialRank],
        dilations[i], outputPadding[i], i, emitError);
    if (failed(offset))
      return failure();
    int64_t inputDim = inputShape[i + 2];
    if (ShapedType::isDynamic(inputDim))
      continue;
    FailureOr<int64_t> outputDim = computeScaledExtent(
        inputDim, strides[i], *offset, "conv_transpose", i, emitError);
    if (failed(outputDim))
      return failure();
    result[i + 2] = *outputDim;
  }
  return result;
}

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyConvTransposeResultShape(
    OpBuilder &b, Location loc, Value input, Value weights,
    ArrayRef<int64_t> kernelShape, ArrayRef<int64_t> strides,
    ArrayRef<int64_t> pads, ArrayRef<int64_t> dilations,
    ArrayRef<int64_t> outputPadding, int64_t group,
    function_ref<InFlightDiagnostic()> emitError) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightsType = dyn_cast<RankedTensorType>(weights.getType());
  if (!inputType || !weightsType) {
    emitError() << "conv_transpose input and weights must be ranked tensors";
    return failure();
  }
  FailureOr<SmallVector<int64_t>> inferred = inferConvTransposeShape(
      inputType.getShape(), weightsType.getShape(), kernelShape, strides, pads,
      dilations, outputPadding, group, emitError);
  if (failed(inferred))
    return failure();

  // Dynamic index arithmetic intentionally keeps the existing mul/add
  // behavior. Validate every constant affine offset before emitting any SSA.
  SmallVector<int64_t> offsets;
  offsets.reserve(2);
  for (int64_t i : llvm::seq<int64_t>(0, 2)) {
    FailureOr<int64_t> offset = computeConvTransposeOffset(
        kernelShape[i], strides[i], pads[i], pads[i + 2], dilations[i],
        outputPadding[i], i, emitError);
    if (failed(offset))
      return failure();
    offsets.push_back(*offset);
  }

  SmallVector<OpFoldResult> inputSizes = tensor::getMixedSizes(b, loc, input);
  SmallVector<OpFoldResult> weightSizes =
      tensor::getMixedSizes(b, loc, weights);
  SmallVector<OpFoldResult> result;
  result.reserve(4);
  result.push_back(inputSizes[0]);
  FailureOr<OpFoldResult> outputChannels =
      detail::scaleAndOffsetDim(b, loc, weightSizes[1], group, /*offset=*/0);
  if (failed(outputChannels))
    return failure();
  result.push_back(*outputChannels);
  for (int64_t i : llvm::seq<int64_t>(0, 2)) {
    FailureOr<OpFoldResult> extent = detail::scaleAndOffsetDim(
        b, loc, inputSizes[i + 2], strides[i], offsets[i]);
    if (failed(extent))
      return failure();
    result.push_back(*extent);
  }
  return result;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferGlobalPoolShape(ArrayRef<int64_t> inputShape,
                                function_ref<InFlightDiagnostic()> emitError) {
  if (inputShape.size() < 3) {
    emitError() << "global_pool input must have rank >= 3";
    return failure();
  }
  SmallVector<int64_t> result(inputShape.begin(), inputShape.end());
  std::fill(result.begin() + 2, result.end(), 1);
  return result;
}

FailureOr<SmallVector<OpFoldResult>> mlir::hip::reifyGlobalPoolResultShape(
    OpBuilder &b, Location loc, Value input,
    function_ref<InFlightDiagnostic()> emitError) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType ||
      failed(inferGlobalPoolShape(inputType.getShape(), emitError)))
    return failure();

  SmallVector<OpFoldResult> result;
  result.reserve(inputType.getRank());
  result.push_back(
      reifyDimOrConstant(b, loc, inputType.getDimSize(0), input, 0));
  result.push_back(
      reifyDimOrConstant(b, loc, inputType.getDimSize(1), input, 1));
  for (int64_t dim : llvm::seq<int64_t>(2, inputType.getRank()))
    result.push_back(b.getIndexAttr(1));
  return result;
}

FailureOr<SmallVector<int64_t>>
mlir::hip::inferResizeShape(ArrayRef<int64_t> inputShape,
                            ArrayRef<int64_t> outputTemplate,
                            function_ref<InFlightDiagnostic()> emitError) {
  if (inputShape.size() < 3 || inputShape.size() > 5 ||
      inputShape.size() != outputTemplate.size()) {
    emitError() << "resize input and output must have matching rank in [3, 5]";
    return failure();
  }

  SmallVector<int64_t> result(inputShape.begin(), inputShape.begin() + 2);
  for (int64_t dim : llvm::seq<int64_t>(2)) {
    int64_t inputExtent = inputShape[dim];
    int64_t templateExtent = outputTemplate[dim];
    if (ShapedType::isDynamic(inputExtent)) {
      if (!ShapedType::isDynamic(templateExtent)) {
        emitError() << "resize output dimension " << dim
                    << " must remain dynamic because input N/C is dynamic";
        return failure();
      }
      continue;
    }
    if (!ShapedType::isDynamic(templateExtent) &&
        inputExtent != templateExtent) {
      emitError() << "resize output dimension " << dim
                  << " must match input N/C extent " << inputExtent;
      return failure();
    }
  }

  for (size_t dim : llvm::seq<size_t>(2, outputTemplate.size())) {
    if (ShapedType::isDynamic(outputTemplate[dim])) {
      emitError() << "resize output spatial dimension " << dim
                  << " must be static because sizes/scales are not carried by "
                     "hip.resize";
      return failure();
    }
    result.push_back(outputTemplate[dim]);
  }
  return result;
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyResizeShape(OpBuilder &b, Location loc, Value input,
                            ArrayRef<int64_t> outputTemplate,
                            function_ref<InFlightDiagnostic()> emitError) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType ||
      failed(inferResizeShape(inputType.getShape(), outputTemplate, emitError)))
    return failure();

  // Validation above is complete before the first tensor.dim is emitted.
  SmallVector<OpFoldResult> result;
  result.reserve(outputTemplate.size());
  result.push_back(
      reifyDimOrConstant(b, loc, inputType.getDimSize(0), input, 0));
  result.push_back(
      reifyDimOrConstant(b, loc, inputType.getDimSize(1), input, 1));
  for (size_t dim : llvm::seq<size_t>(2, outputTemplate.size()))
    result.push_back(b.getIndexAttr(outputTemplate[dim]));
  return result;
}
