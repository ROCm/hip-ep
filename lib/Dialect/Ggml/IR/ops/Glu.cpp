/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- Glu.cpp - ggml.glu -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

/// Split in two operands, the two alike.
void GluOp::constrainOperands(ShapeConstraints &constraints) {
  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension)
    equateCaptured(getUp(), /*slot=*/1, dimension, getGate(), /*otherSlot=*/0,
                   dimension, constraints);
}

LogicalResult GluOp::deriveResultShape(ShapeDerivation &derivation) {
  if (!derivation.hasOperand(0))
    return emitOpError() << "expects a gate operand";
  const Shape &gate = derivation.operandShape(0);
  if (getSplit()) {
    derivation.setResultShape(gate);
  } else {
    // One operand carries both halves, so the result is half as wide.
    derivation.setResultShape({derivation.arithmetic().divide(gate[0], 2),
                               gate[1], gate[2], gate[3]});
  }
  return success();
}

LogicalResult GluOp::verify() {
  if (getSplit() != static_cast<bool>(getUp())) {
    return emitOpError(
        "requires `split` to match the presence of the `up` operand");
  }
  auto gateType = getGate().getType();
  auto resultType = getResult().getType();
  if (gateType.getRank() == 0 || gateType.getRank() != resultType.getRank() ||
      gateType.getElementType() != resultType.getElementType()) {
    return emitOpError(
        "requires matching nonzero gate/result rank and element type");
  }
  if (Value up = getUp()) {
    auto upType = cast<RankedTensorType>(up.getType());
    if (upType.getElementType() != gateType.getElementType() ||
        failed(verifyCompatibleShape(gateType, upType)) ||
        failed(verifyCompatibleShape(gateType, resultType))) {
      return emitOpError(
          "requires compatible gate, up, and result types in split form");
    }
    if (getSwapped()) {
      return emitOpError("does not permit `swapped` in split form");
    }
  } else {
    for (int64_t dimension = 0; dimension + 1 < gateType.getRank();
         ++dimension) {
      if (!compatibleDimensions(gateType.getDimSize(dimension),
                                resultType.getDimSize(dimension))) {
        return emitOpError(
            "requires non-split result dimensions to match the gate");
      }
    }
    const int64_t gateWidth = gateType.getDimSize(gateType.getRank() - 1);
    const int64_t resultWidth = resultType.getDimSize(resultType.getRank() - 1);
    if (!ShapedType::isDynamic(gateWidth) &&
        !ShapedType::isDynamic(resultWidth) &&
        (gateWidth <= 0 || gateWidth % 2 != 0 ||
         gateWidth / 2 != resultWidth)) {
      return emitOpError(
          "requires non-split gate width to be twice the result width");
    }
  }
  if (getGluKind() == GluKind::SwigluOai && !getSplit()) {
    return emitOpError("requires split form for SwiGLU-OAI");
  }
  if (!getAlphaAttr().getValue().isFinite() ||
      !getLimitAttr().getValue().isFinite()) {
    return emitOpError("requires finite `alpha` and `limit`");
  }
  return success();
}
