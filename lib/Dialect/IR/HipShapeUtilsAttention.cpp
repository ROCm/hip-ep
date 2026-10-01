/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipShapeUtilsAttention.cpp - Attention and normalization shapes ---===//
//
// Category implementation for the public shape helpers declared in
// `hip/Dialect/IR/HipShapeUtils.h`.
//
//===----------------------------------------------------------------------===//

#include "HipShapeUtilsInternal.h"
#include "hip/Dialect/IR/HipShapeUtils.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"

#include <optional>

using namespace mlir;
using namespace mlir::hip;

namespace {

bool areCompatibleStaticExtents(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

} // namespace

FailureOr<SmallVector<SmallVector<int64_t>>>
mlir::hip::inferLayerNormOutputShapes(ArrayRef<int64_t> inputShape,
                                      int64_t axis, unsigned numOutputs) {
  if (numOutputs < 1 || numOutputs > 3 || inputShape.empty())
    return failure();
  int64_t rank = inputShape.size();
  int64_t normalizedAxis = axis < 0 ? axis + rank : axis;
  if (normalizedAxis < 0 || normalizedAxis >= rank)
    return failure();
  SmallVector<int64_t> reductionAxes =
      llvm::to_vector(llvm::seq<int64_t>(normalizedAxis, rank));
  FailureOr<SmallVector<int64_t>> statsShape =
      inferReductionShape(inputShape, reductionAxes, /*keepdims=*/1);
  if (failed(statsShape))
    return failure();

  SmallVector<SmallVector<int64_t>> results;
  results.push_back(SmallVector<int64_t>(inputShape));
  if (numOutputs >= 2)
    results.push_back(*statsShape);
  if (numOutputs >= 3)
    results.push_back(*statsShape);
  return results;
}

FailureOr<ReifiedRankedShapedTypeDims>
mlir::hip::reifyLayerNormOutputShapes(OpBuilder &b, Location loc, Value input,
                                      int64_t axis, unsigned numOutputs) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  if (!inputType || failed(inferLayerNormOutputShapes(inputType.getShape(),
                                                      axis, numOutputs)))
    return failure();
  int64_t rank = inputType.getRank();
  int64_t normalizedAxis = axis < 0 ? axis + rank : axis;
  SmallVector<int64_t> reductionAxes =
      llvm::to_vector(llvm::seq<int64_t>(normalizedAxis, rank));

  FailureOr<SmallVector<OpFoldResult>> outputShape =
      reifyElementwiseSameShape(b, loc, input);
  FailureOr<SmallVector<OpFoldResult>> statsShape =
      reifyReductionResultShape(b, loc, input, reductionAxes, /*keepdims=*/1);
  if (failed(outputShape) || failed(statsShape))
    return failure();
  ReifiedRankedShapedTypeDims results;
  results.push_back(std::move(*outputShape));
  if (numOutputs >= 2)
    results.push_back(*statsShape);
  if (numOutputs >= 3)
    results.push_back(std::move(*statsShape));
  return results;
}

FailureOr<Type> mlir::hip::inferLayerNormStatsType(MLIRContext *ctx,
                                                   int64_t stashType) {
  if (stashType == 0 || stashType == 1)
    return Float32Type::get(ctx);
  if (stashType == 10)
    return Float16Type::get(ctx);
  return failure();
}

FailureOr<SmallVector<SmallVector<int64_t>>>
mlir::hip::inferCausalConvWithStateOutputShapes(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    std::optional<ArrayRef<int64_t>> biasShape,
    std::optional<ArrayRef<int64_t>> pastStateShape, int64_t ndim,
    bool channelsLast, function_ref<InFlightDiagnostic()> emitError) {
  if (ndim != 1) {
    emitError() << "causal_conv_with_state runtime supports only ndim=1";
    return failure();
  }
  if (inputShape.size() != 3 || weightShape.size() != 3) {
    emitError() << "causal_conv_with_state input and weight must have rank 3 "
                   "for ndim=1";
    return failure();
  }

  const size_t channelDim = channelsLast ? 2 : 1;
  if (!areCompatibleStaticExtents(inputShape[channelDim], weightShape[0])) {
    emitError() << "causal_conv_with_state weight channels must match input "
                   "channels";
    return failure();
  }
  if (!ShapedType::isDynamic(weightShape[1]) && weightShape[1] != 1) {
    emitError() << "causal_conv_with_state weight dimension 1 must be 1 for "
                   "depthwise convolution";
    return failure();
  }
  if (!ShapedType::isDynamic(weightShape[2]) && weightShape[2] <= 0) {
    emitError() << "causal_conv_with_state kernel extent must be positive";
    return failure();
  }

  if (biasShape) {
    if (biasShape->size() != 1) {
      emitError() << "causal_conv_with_state bias must have rank 1";
      return failure();
    }
    if (!areCompatibleStaticExtents((*biasShape)[0], inputShape[channelDim])) {
      emitError() << "causal_conv_with_state bias length must match input "
                     "channels";
      return failure();
    }
  }

  int64_t stateLength = ShapedType::isDynamic(weightShape[2])
                            ? ShapedType::kDynamic
                            : weightShape[2] - 1;
  SmallVector<int64_t> stateShape = {inputShape[0], inputShape[channelDim],
                                     stateLength};
  if (pastStateShape) {
    if (pastStateShape->size() != stateShape.size()) {
      emitError() << "causal_conv_with_state past_state must have rank 3";
      return failure();
    }
    for (size_t dim : llvm::seq<size_t>(0, stateShape.size())) {
      if (!areCompatibleStaticExtents((*pastStateShape)[dim],
                                      stateShape[dim])) {
        emitError() << "causal_conv_with_state past_state dimension " << dim
                    << " must match [B, C, K-1]";
        return failure();
      }
    }
  }

  return SmallVector<SmallVector<int64_t>>{SmallVector<int64_t>(inputShape),
                                           std::move(stateShape)};
}

FailureOr<ReifiedRankedShapedTypeDims>
mlir::hip::reifyCausalConvWithStateOutputShapes(
    OpBuilder &b, Location loc, Value input, Value weight, Value bias,
    Value pastState, int64_t ndim, bool channelsLast,
    function_ref<InFlightDiagnostic()> emitError) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightType = dyn_cast<RankedTensorType>(weight.getType());
  auto biasType =
      bias ? dyn_cast<RankedTensorType>(bias.getType()) : RankedTensorType{};
  auto pastStateType = pastState
                           ? dyn_cast<RankedTensorType>(pastState.getType())
                           : RankedTensorType{};
  if (!inputType || !weightType || (bias && !biasType) ||
      (pastState && !pastStateType)) {
    emitError() << "causal_conv_with_state operands must be ranked tensors";
    return failure();
  }

  std::optional<ArrayRef<int64_t>> biasShape;
  if (biasType)
    biasShape = biasType.getShape();
  std::optional<ArrayRef<int64_t>> pastStateShape;
  if (pastStateType)
    pastStateShape = pastStateType.getShape();
  if (failed(inferCausalConvWithStateOutputShapes(
          inputType.getShape(), weightType.getShape(), biasShape,
          pastStateShape, ndim, channelsLast, emitError)))
    return failure();

  SmallVector<OpFoldResult> outputShape = tensor::getMixedSizes(b, loc, input);
  FailureOr<OpFoldResult> stateLength =
      detail::scaleAndOffsetDim(b, loc, tensor::getMixedSize(b, loc, weight, 2),
                                /*scale=*/1, /*offset=*/-1);
  if (failed(stateLength))
    return failure();
  const int64_t channelDim = channelsLast ? 2 : 1;
  SmallVector<OpFoldResult> presentStateShape = {
      tensor::getMixedSize(b, loc, input, 0),
      tensor::getMixedSize(b, loc, input, channelDim), *stateLength};
  return ReifiedRankedShapedTypeDims{std::move(outputShape),
                                     std::move(presentStateShape)};
}

FailureOr<SmallVector<SmallVector<int64_t>>>
mlir::hip::inferSkipRmsNormOutputShapes(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> skipShape,
    ArrayRef<int64_t> gammaShape, std::optional<ArrayRef<int64_t>> biasShape,
    unsigned numOutputs, function_ref<InFlightDiagnostic()> emitError) {
  if (numOutputs < 1 || numOutputs > 2) {
    emitError() << "skip_rms_norm expects 1 or 2 output buffers";
    return failure();
  }
  if (inputShape.empty()) {
    emitError() << "skip_rms_norm input must have rank at least 1";
    return failure();
  }
  if (skipShape.size() != inputShape.size()) {
    emitError() << "skip_rms_norm skip rank must match input rank";
    return failure();
  }

  for (size_t dim : llvm::seq<size_t>(0, inputShape.size())) {
    if (!areCompatibleStaticExtents(skipShape[dim], inputShape[dim])) {
      emitError() << "skip_rms_norm skip dimension " << dim
                  << " must match input";
      return failure();
    }
  }
  if (gammaShape.size() != 1) {
    emitError() << "skip_rms_norm gamma must have rank 1";
    return failure();
  }
  if (!areCompatibleStaticExtents(gammaShape[0], inputShape.back())) {
    emitError() << "skip_rms_norm gamma length must match the input's final "
                   "dimension";
    return failure();
  }
  if (biasShape) {
    if (biasShape->size() != 1) {
      emitError() << "skip_rms_norm bias must have rank 1";
      return failure();
    }
    if (!areCompatibleStaticExtents((*biasShape)[0], inputShape.back())) {
      emitError() << "skip_rms_norm bias length must match the input's final "
                     "dimension";
      return failure();
    }
  }

  SmallVector<SmallVector<int64_t>> results;
  results.assign(numOutputs, SmallVector<int64_t>(inputShape));
  return results;
}

FailureOr<ReifiedRankedShapedTypeDims> mlir::hip::reifySkipRmsNormOutputShapes(
    OpBuilder &b, Location loc, Value input, Value skip, Value gamma,
    Value bias, unsigned numOutputs,
    function_ref<InFlightDiagnostic()> emitError) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto skipType = dyn_cast<RankedTensorType>(skip.getType());
  auto gammaType = dyn_cast<RankedTensorType>(gamma.getType());
  auto biasType =
      bias ? dyn_cast<RankedTensorType>(bias.getType()) : RankedTensorType{};
  if (!inputType || !skipType || !gammaType || (bias && !biasType)) {
    emitError() << "skip_rms_norm operands must be ranked tensors";
    return failure();
  }

  std::optional<ArrayRef<int64_t>> biasShape;
  if (biasType)
    biasShape = biasType.getShape();
  if (failed(inferSkipRmsNormOutputShapes(
          inputType.getShape(), skipType.getShape(), gammaType.getShape(),
          biasShape, numOutputs, emitError)))
    return failure();

  // Reify once and reuse the same extent SSA for the optional residual output.
  SmallVector<OpFoldResult> shape = tensor::getMixedSizes(b, loc, input);
  ReifiedRankedShapedTypeDims results;
  results.assign(numOutputs, shape);
  return results;
}
