/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange GatherOp::getDpsInitsMutable() { return getInitMutable(); }

LogicalResult GatherOp::verify() {
  auto dataType = cast<ShapedType>(getData().getType());
  auto indicesType = cast<ShapedType>(getIndices().getType());
  auto outputType = cast<ShapedType>(getInit().getType());
  int64_t axis = getAxis();

  if (axis < 0 || axis >= dataType.getRank()) {
    return emitOpError("axis must be in [0, data rank); data rank is ")
           << dataType.getRank() << ", got " << axis;
  }
  if (!indicesType.getElementType().isIntOrIndex()) {
    return emitOpError("indices element type must be an integer");
  }
  if (dataType.getElementType() != outputType.getElementType()) {
    return emitOpError("data and output element types must match");
  }

  // The gathered axis is replaced by the whole indices shape.
  SmallVector<int64_t> expectedShape(dataType.getShape().take_front(axis));
  llvm::append_range(expectedShape, indicesType.getShape());
  llvm::append_range(expectedShape, dataType.getShape().drop_front(axis + 1));
  if (failed(verifyCompatibleShape(expectedShape, outputType.getShape()))) {
    return emitOpError("output shape must be the data shape with axis replaced "
                       "by the indices shape");
  }
  return success();
}
