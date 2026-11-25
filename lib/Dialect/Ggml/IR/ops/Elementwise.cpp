/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- Elementwise.cpp - One value per value -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

namespace {

/// GGML repeats the second operand of an elementwise binary operation into
/// the first in whole copies, so each of the first's extents is a multiple of
/// the second's. Where the capture saw the two equal and the second spans
/// more than one element, the first holds a single copy: a count of copies
/// that moved with the request would have had to come out at exactly one for
/// the captured request.
void constrainBroadcast(Value lhs, Value rhs, ShapeConstraints &constraints) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || lhsType.getRank() != kGgmlRank ||
      rhsType.getRank() != kGgmlRank)
    return;
  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
    // The tensor type spells GGML's dimensions outermost first.
    const int64_t lhsExtent = lhsType.getDimSize(kGgmlRank - 1 - dimension);
    const int64_t rhsExtent = rhsType.getDimSize(kGgmlRank - 1 - dimension);
    if (ShapedType::isDynamic(lhsExtent) || lhsExtent != rhsExtent ||
        rhsExtent <= 1)
      continue;
    constraints.equate(/*slot=*/0, dimension, /*otherSlot=*/1, dimension);
  }
}

} // namespace

void AddOp::constrainOperands(ShapeConstraints &constraints) {
  constrainBroadcast(getLhs(), getRhs(), constraints);
}
void MulOp::constrainOperands(ShapeConstraints &constraints) {
  constrainBroadcast(getLhs(), getRhs(), constraints);
}
GGML_SAME_SHAPE_AS(AddOp, 0)

LogicalResult AddOp::verify() {
  return verifyElementwiseBinary(*this, getLhs(), getRhs(), getResult());
}

GGML_SAME_SHAPE_AS(MulOp, 0)

LogicalResult MulOp::verify() {
  return verifyElementwiseBinary(*this, getLhs(), getRhs(), getResult());
}

GGML_SAME_SHAPE_AS(RmsNormOp, 0)

LogicalResult RmsNormOp::verify() {
  if (failed(verifyElementwiseUnary(*this, getInput(), getResult()))) {
    return failure();
  }
  if (!getInput().getType().getElementType().isF32())
    return emitOpError("requires f32 input and result tensors");
  return verifyFiniteNonNegativeFloat(*this, getEpsilonAttr(), "epsilon");
}
