/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SetRows.cpp - ggml.set_rows -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

/// updates[width, rows, ...], indices[rows, ...], destination[width, ...]:
/// an index per row, rows as wide as the destination's.
void SetRowsOp::constrainOperands(ShapeConstraints &constraints) {
  equateCaptured(getIndices(), /*slot=*/1, 0, getUpdates(), /*otherSlot=*/0, 1,
                 constraints);
  equateCaptured(getUpdates(), /*slot=*/0, 0, getDestination(),
                 /*otherSlot=*/2, 0, constraints);
}

// The write lands in the destination and yields it.
GGML_SAME_SHAPE_AS(SetRowsOp, 2)

LogicalResult SetRowsOp::verify() {
  auto updatesType = dyn_cast<RankedTensorType>(getUpdates().getType());
  auto indicesType = dyn_cast<RankedTensorType>(getIndices().getType());
  auto destinationType = dyn_cast<RankedTensorType>(getDestination().getType());
  if (!updatesType || !indicesType || !destinationType) {
    return emitOpError("requires three ranked tensor operands");
  }
  if (!isa<IntegerType>(indicesType.getElementType())) {
    return emitOpError("requires integer `indices` elements");
  }
  if (!haveSameShapeAndElementType(
          cast<RankedTensorType>(getResult().getType()), destinationType)) {
    return emitOpError("requires the result shape and element type to equal "
                       "the destination");
  }
  return success();
}
