/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange NonZeroOp::getDpsInitsMutable() {
  // The two destinations are adjacent operands, which no single accessor
  // spans.
  return MutableOperandRange(getOperation(),
                             getIndicesInitMutable().getOperandNumber(),
                             /*length=*/2);
}

LogicalResult NonZeroOp::verify() {
  auto inputType = cast<ShapedType>(getInput().getType());
  auto indicesType = cast<ShapedType>(getIndicesInit().getType());
  auto countType = cast<ShapedType>(getCountInit().getType());

  if (!indicesType.getElementType().isInteger(64)) {
    return emitOpError("indices element type must be i64");
  }
  if (!countType.getElementType().isInteger(64)) {
    return emitOpError("count element type must be i64");
  }
  if (indicesType.getRank() != 2) {
    return emitOpError("indices must be rank-2: one row per input axis, one "
                       "column per position found");
  }
  // The row count is the input rank, which is known at compile time.
  if (indicesType.isDynamicDim(0)) {
    return emitOpError("indices must have a static row count");
  }
  if (indicesType.getDimSize(0) != inputType.getRank()) {
    return emitOpError("indices must have one row per input axis; input rank "
                       "is ")
           << inputType.getRank() << ", got " << indicesType.getDimSize(0);
  }
  if (countType.getRank() != 1 || countType.getDimSize(0) != 1) {
    return emitOpError("count must be a static single-element vector");
  }
  return success();
}
