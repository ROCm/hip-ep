/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GetRows.cpp - ggml.get_rows -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

/// data[width, rows, ..] gathered by indices[n, i1, i2, _] -> [width, n, i1,
/// i2]
LogicalResult GetRowsOp::deriveResultShape(ShapeDerivation &derivation) {
  if (!derivation.hasOperand(0) || !derivation.hasOperand(1))
    return emitOpError() << "expects two operands";
  const Shape &data = derivation.operandShape(0);
  const Shape &indices = derivation.operandShape(1);
  derivation.setResultShape({data[0], indices[0], indices[1], indices[2]});
  return success();
}

/// `ggml_get_rows` asserts that the data's outer two extents are the indices',
/// so a gather from a tensor whose extents nothing derives -- a `ggml.repeat`
/// -- takes them from the indices instead.
void GetRowsOp::constrainOperands(ShapeConstraints &constraints) {
  constraints.equate(/*slot=*/0, /*dimension=*/2, /*otherSlot=*/1,
                     /*otherDimension=*/1);
  constraints.equate(/*slot=*/0, /*dimension=*/3, /*otherSlot=*/1,
                     /*otherDimension=*/2);
}

LogicalResult GetRowsOp::verify() {
  auto dataType = getData().getType();
  auto indicesType = getIndices().getType();
  auto resultType = getResult().getType();
  if (dataType.getRank() != 4 || indicesType.getRank() != 4 ||
      resultType.getRank() != 4) {
    return emitOpError("requires rank-4 data, indices, and result tensors");
  }
  if (!indicesType.getElementType().isInteger(32)) {
    return emitOpError("requires i32 indices");
  }

  Type dataElementType = dataType.getElementType();
  if (!dataElementType.isF16() && !dataElementType.isBF16() &&
      !dataElementType.isF32() && !dataElementType.isInteger(32)) {
    return emitOpError("requires bf16, f16, f32, or i32 data");
  }
  Type expectedResultElementType = dataElementType.isInteger(32)
                                       ? dataElementType
                                       : Float32Type::get(getContext());
  if (resultType.getElementType() != expectedResultElementType) {
    return emitOpError("has a result element type inconsistent with GGML");
  }
  if (indicesType.getDimSize(0) != 1 ||
      !compatibleDimensions(dataType.getDimSize(0),
                            indicesType.getDimSize(1)) ||
      !compatibleDimensions(dataType.getDimSize(0), resultType.getDimSize(0)) ||
      !compatibleDimensions(dataType.getDimSize(1),
                            indicesType.getDimSize(2)) ||
      !compatibleDimensions(dataType.getDimSize(1), resultType.getDimSize(1)) ||
      !compatibleDimensions(indicesType.getDimSize(3),
                            resultType.getDimSize(2)) ||
      !compatibleDimensions(dataType.getDimSize(3), resultType.getDimSize(3))) {
    return emitOpError(
        "requires data [B1, B0, rows, width], indices [1, B1, B0, "
        "selected_rows], and result [B1, B0, selected_rows, width]");
  }
  return success();
}
