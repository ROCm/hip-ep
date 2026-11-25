/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- MulMat.cpp - ggml.mul_mat -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

/// The input as long as the weight's rows (the product's shared extent).
/// GGML repeats the weight across the input's batch axes in whole copies
/// (ggml_can_mul_mat), so each of the input's batch extents is a multiple of
/// the weight's. Captured equal and of more than one element, the input
/// holds a single copy, as an elementwise operation's first operand does
/// (Elementwise.cpp): what lets a `ggml.repeat` (Inkling 2's depth-mix
/// queries, one row a token) take its extent from the weight it meets.
void MulMatOp::constrainOperands(ShapeConstraints &constraints) {
  for (unsigned dimension : {0u, 2u, 3u})
    equateCaptured(getInput(), /*slot=*/1, dimension, getWeight(),
                   /*otherSlot=*/0, dimension, constraints);
}

/// weight[K, N, b2, b3] x input[K, M, b2, b3] -> [N, M, b2, b3]
LogicalResult MulMatOp::deriveResultShape(ShapeDerivation &derivation) {
  if (!derivation.hasOperand(0) || !derivation.hasOperand(1))
    return emitOpError() << "expects two operands";
  const Shape &weight = derivation.operandShape(0);
  const Shape &activation = derivation.operandShape(1);
  derivation.setResultShape(
      {weight[1], activation[1], activation[2], activation[3]});
  return success();
}

LogicalResult MulMatOp::verify() {
  auto weightType = getWeight().getType();
  auto inputType = getInput().getType();
  auto resultType = getResult().getType();
  if (weightType.getRank() != 4 || inputType.getRank() != 4 ||
      resultType.getRank() != 4) {
    return emitOpError("requires rank-4 tensors in the raw GGML layout");
  }
  const Type weightElement = weightType.getElementType();
  const Type inputElement = inputType.getElementType();
  if ((!weightElement.isF16() && !weightElement.isBF16() &&
       !weightElement.isF32()) ||
      (!inputElement.isF16() && !inputElement.isF32()) ||
      !resultType.getElementType().isF32()) {
    return emitOpError(
        "requires a bf16/f16/f32 weight, an f16/f32 input, and an f32 result");
  }
  // GGML converts the activation to the weight's dot-product type, so an f16
  // activation is only meaningful against an f16 weight. BF16 parameters use
  // f32 graph activations and perform that conversion inside MUL_MAT. Vision
  // patch embedding relies on the explicit f16 form because im2col already
  // produces f16 patches.
  if (inputElement.isF16() && !weightElement.isF16()) {
    return emitOpError("requires an f16 weight when the input is f16");
  }
  if (getHint() != 0 && getHint() != 1) {
    return emitOpError("requires `hint` to be 0 or 1");
  }
  if (!compatibleDimensions(weightType.getDimSize(3),
                            inputType.getDimSize(3)) ||
      !compatibleDimensions(inputType.getDimSize(0),
                            resultType.getDimSize(0)) ||
      !compatibleDimensions(inputType.getDimSize(1),
                            resultType.getDimSize(1)) ||
      !compatibleDimensions(inputType.getDimSize(2),
                            resultType.getDimSize(2)) ||
      !compatibleDimensions(weightType.getDimSize(2),
                            resultType.getDimSize(3))) {
    return emitOpError("has incompatible matrix and result dimensions");
  }
  for (int64_t dimension = 0; dimension < 2; ++dimension) {
    const int64_t weightExtent = weightType.getDimSize(dimension);
    const int64_t inputExtent = inputType.getDimSize(dimension);
    if (!ShapedType::isDynamic(weightExtent) &&
        !ShapedType::isDynamic(inputExtent) &&
        (weightExtent <= 0 || inputExtent % weightExtent != 0)) {
      return emitOpError() << "requires weight batch dimension " << dimension
                           << " to divide the corresponding input dimension";
    }
  }
  return success();
}
