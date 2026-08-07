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

bool areCompatibleStaticExtents(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

LogicalResult validateRank3Qkv(StringRef opName, ArrayRef<int64_t> queryShape,
                               ArrayRef<int64_t> keyShape,
                               ArrayRef<int64_t> valueShape,
                               function_ref<InFlightDiagnostic()> emitError) {
  if (queryShape.size() == 3 && keyShape.size() == 3 && valueShape.size() == 3)
    return success();
  emitError() << opName << " query, key, and value must have rank 3";
  return failure();
}

} // namespace

FailureOr<SmallVector<int64_t>> mlir::hip::inferMultiHeadAttentionOutputShape(
    ArrayRef<int64_t> queryShape, ArrayRef<int64_t> keyShape,
    ArrayRef<int64_t> valueShape, int64_t numHeads,
    function_ref<InFlightDiagnostic()> emitError) {
  if (failed(validateRank3Qkv("multi_head_attention", queryShape, keyShape,
                              valueShape, emitError)))
    return failure();
  if (numHeads <= 0) {
    emitError() << "multi_head_attention num_heads must be positive";
    return failure();
  }

  if (!areCompatibleStaticExtents(queryShape[0], keyShape[0]) ||
      !areCompatibleStaticExtents(queryShape[0], valueShape[0])) {
    emitError() << "multi_head_attention Q/K/V batch extents must agree";
    return failure();
  }
  if (!areCompatibleStaticExtents(keyShape[1], valueShape[1])) {
    emitError() << "multi_head_attention K/V sequence extents must agree";
    return failure();
  }
  if (!areCompatibleStaticExtents(queryShape[2], keyShape[2]) ||
      !areCompatibleStaticExtents(queryShape[2], valueShape[2])) {
    emitError() << "multi_head_attention Q/K/V hidden extents must agree";
    return failure();
  }

  auto verifyPositive = [&](StringRef name,
                            ArrayRef<int64_t> shape) -> LogicalResult {
    for (auto [dim, extent] : llvm::enumerate(shape)) {
      if (!ShapedType::isDynamic(extent) && extent <= 0) {
        emitError() << "multi_head_attention " << name << " dimension " << dim
                    << " must be positive";
        return failure();
      }
    }
    return success();
  };
  if (failed(verifyPositive("query", queryShape)) ||
      failed(verifyPositive("key", keyShape)) ||
      failed(verifyPositive("value", valueShape)))
    return failure();
  if (!ShapedType::isDynamic(queryShape[2]) && queryShape[2] % numHeads != 0) {
    emitError() << "multi_head_attention query hidden extent " << queryShape[2]
                << " must be divisible by num_heads " << numHeads;
    return failure();
  }
  return SmallVector<int64_t>(queryShape);
}

FailureOr<SmallVector<OpFoldResult>>
mlir::hip::reifyMultiHeadAttentionOutputShape(
    OpBuilder &b, Location loc, Value query, Value key, Value value,
    int64_t numHeads, function_ref<InFlightDiagnostic()> emitError) {
  auto queryType = dyn_cast<RankedTensorType>(query.getType());
  auto keyType = dyn_cast<RankedTensorType>(key.getType());
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!queryType || !keyType || !valueType ||
      failed(inferMultiHeadAttentionOutputShape(
          queryType.getShape(), keyType.getShape(), valueType.getShape(),
          numHeads, emitError)))
    return failure();

  // Validation above is complete before mixed sizes materialize tensor.dim.
  return tensor::getMixedSizes(b, loc, query);
}

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
mlir::hip::inferLinearAttentionOutputShapes(
    ArrayRef<int64_t> queryShape, ArrayRef<int64_t> keyShape,
    ArrayRef<int64_t> valueShape, int64_t qNumHeads, int64_t kvNumHeads,
    function_ref<InFlightDiagnostic()> emitError) {
  if (failed(validateRank3Qkv("linear_attention", queryShape, keyShape,
                              valueShape, emitError)))
    return failure();
  if (qNumHeads <= 0 || kvNumHeads <= 0) {
    emitError() << "linear_attention head counts must be positive";
    return failure();
  }
  if (qNumHeads % kvNumHeads != 0 && kvNumHeads % qNumHeads != 0) {
    emitError() << "linear_attention q_num_heads and kv_num_heads must divide "
                   "one another, got "
                << qNumHeads << " and " << kvNumHeads;
    return failure();
  }

  for (int64_t dim : {0, 1}) {
    if (!areCompatibleStaticExtents(queryShape[dim], keyShape[dim]) ||
        !areCompatibleStaticExtents(queryShape[dim], valueShape[dim])) {
      emitError() << "linear_attention batch/sequence extent mismatch at "
                     "dimension "
                  << dim;
      return failure();
    }
  }

  auto divideStatic = [&](int64_t extent, int64_t divisor,
                          StringRef name) -> FailureOr<int64_t> {
    if (ShapedType::isDynamic(extent))
      return ShapedType::kDynamic;
    if (extent <= 0 || extent % divisor != 0) {
      emitError() << "linear_attention " << name << " extent " << extent
                  << " must be positive and divisible by " << divisor;
      return failure();
    }
    return extent / divisor;
  };

  FailureOr<int64_t> dk =
      divideStatic(queryShape[2], qNumHeads, "query hidden");
  FailureOr<int64_t> dv =
      divideStatic(valueShape[2], kvNumHeads, "value hidden");
  if (failed(dk) || failed(dv))
    return failure();

  if (!ShapedType::isDynamic(keyShape[2]) && !ShapedType::isDynamic(*dk)) {
    if (keyShape[2] <= 0 || keyShape[2] % *dk != 0) {
      emitError() << "linear_attention key hidden extent " << keyShape[2]
                  << " must be positive and divisible by Dk " << *dk;
      return failure();
    }
    int64_t keyHeads = keyShape[2] / *dk;
    if (keyHeads <= 0 || kvNumHeads % keyHeads != 0) {
      emitError() << "linear_attention derived key head count " << keyHeads
                  << " must divide kv_num_heads " << kvNumHeads;
      return failure();
    }
  }

  int64_t outputHidden = ShapedType::kDynamic;
  if (!ShapedType::isDynamic(*dv)) {
    APInt hidden =
        APInt(128, std::max(qNumHeads, kvNumHeads), /*isSigned=*/true) *
        APInt(128, *dv, /*isSigned=*/true);
    if (!hidden.isSignedIntN(64) || hidden.isNegative()) {
      emitError()
          << "linear_attention inferred output hidden extent is out of range";
      return failure();
    }
    outputHidden = hidden.getSExtValue();
  }
  return SmallVector<SmallVector<int64_t>>{
      {queryShape[0], queryShape[1], outputHidden},
      {queryShape[0], kvNumHeads, *dk, *dv}};
}

FailureOr<ReifiedRankedShapedTypeDims>
mlir::hip::reifyLinearAttentionOutputShapes(
    OpBuilder &b, Location loc, Value query, Value key, Value value,
    int64_t qNumHeads, int64_t kvNumHeads,
    function_ref<InFlightDiagnostic()> emitError) {
  auto queryType = dyn_cast<RankedTensorType>(query.getType());
  auto keyType = dyn_cast<RankedTensorType>(key.getType());
  auto valueType = dyn_cast<RankedTensorType>(value.getType());
  if (!queryType || !keyType || !valueType ||
      failed(inferLinearAttentionOutputShapes(
          queryType.getShape(), keyType.getShape(), valueType.getShape(),
          qNumHeads, kvNumHeads, emitError)))
    return failure();

  OpFoldResult batch = tensor::getMixedSize(b, loc, query, 0);
  OpFoldResult sequence = tensor::getMixedSize(b, loc, query, 1);
  OpFoldResult queryHidden = tensor::getMixedSize(b, loc, query, 2);
  OpFoldResult valueHidden = tensor::getMixedSize(b, loc, value, 2);

  auto divideExtent = [&](OpFoldResult extent,
                          int64_t divisor) -> OpFoldResult {
    if (std::optional<int64_t> constant = getConstantIntValue(extent))
      return b.getIndexAttr(*constant / divisor);
    Value extentValue = getValueOrCreateConstantIndexOp(b, loc, extent);
    Value divisorValue = arith::ConstantIndexOp::create(b, loc, divisor);
    return arith::DivUIOp::create(b, loc, extentValue, divisorValue)
        .getResult();
  };
  OpFoldResult dk = divideExtent(queryHidden, qNumHeads);
  OpFoldResult dv = divideExtent(valueHidden, kvNumHeads);
  FailureOr<OpFoldResult> outputHidden = detail::scaleAndOffsetDim(
      b, loc, dv, std::max(qNumHeads, kvNumHeads), /*offset=*/0);
  if (failed(outputHidden))
    return failure();
  return ReifiedRankedShapedTypeDims{
      {batch, sequence, *outputHidden},
      {batch, b.getIndexAttr(kvNumHeads), dk, dv}};
}

FailureOr<SmallVector<SmallVector<int64_t>>>
mlir::hip::inferCausalConvWithStateOutputShapes(
    ArrayRef<int64_t> inputShape, ArrayRef<int64_t> weightShape,
    std::optional<ArrayRef<int64_t>> biasShape,
    std::optional<ArrayRef<int64_t>> pastStateShape, int64_t ndim,
    function_ref<InFlightDiagnostic()> emitError) {
  if (ndim != 1) {
    emitError() << "causal_conv_with_state runtime supports only ndim=1";
    return failure();
  }
  if (inputShape.size() != 3 || weightShape.size() != 3) {
    emitError() << "causal_conv_with_state input and weight must have rank 3 "
                   "for ndim=1";
    return failure();
  }

  if (!areCompatibleStaticExtents(inputShape[1], weightShape[0])) {
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
    if (!areCompatibleStaticExtents((*biasShape)[0], inputShape[1])) {
      emitError() << "causal_conv_with_state bias length must match input "
                     "channels";
      return failure();
    }
  }

  int64_t stateLength = ShapedType::isDynamic(weightShape[2])
                            ? ShapedType::kDynamic
                            : weightShape[2] - 1;
  SmallVector<int64_t> stateShape = {inputShape[0], inputShape[1], stateLength};
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
    Value pastState, int64_t ndim,
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
          pastStateShape, ndim, emitError)))
    return failure();

  SmallVector<OpFoldResult> outputShape = tensor::getMixedSizes(b, loc, input);
  FailureOr<OpFoldResult> stateLength =
      detail::scaleAndOffsetDim(b, loc, tensor::getMixedSize(b, loc, weight, 2),
                                /*scale=*/1, /*offset=*/-1);
  if (failed(stateLength))
    return failure();
  SmallVector<OpFoldResult> presentStateShape = {
      tensor::getMixedSize(b, loc, input, 0),
      tensor::getMixedSize(b, loc, input, 1), *stateLength};
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
