/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- Permute.cpp - ggml.permute -===//
//
// What the operation is and what shape its result has.
//
//===----------------------------------------------------------------------===//

#include "OpsShared.h"

using namespace mlir;
using namespace mlir::ggml;
using namespace mlir::ggml::shared;

LogicalResult PermuteOp::deriveResultShape(ShapeDerivation &derivation) {
  if (!derivation.hasOperand(0))
    return emitOpError() << "expects an operand";
  SmallVector<int64_t, kGgmlRank> axes;
  for (Attribute entry : getAxes())
    axes.push_back(cast<IntegerAttr>(entry).getInt());
  if (axes.size() != kGgmlRank)
    return emitOpError() << "requires four permutation axes";

  const Shape &input = derivation.operandShape(0);
  const Shape &inputStrides = derivation.operandStrides(0);
  Shape shape(kGgmlRank, derivation.constant(0));
  Shape permuted(kGgmlRank, derivation.constant(0));
  llvm::SmallSet<int64_t, kGgmlRank> seen;
  for (auto [source, destination] : llvm::enumerate(axes)) {
    if (destination < 0 || destination >= kGgmlRank ||
        !seen.insert(destination).second) {
      return emitOpError() << "requires `axes` to be a permutation";
    }
    shape[destination] = input[source];
    // Relabelling axes moves no bytes, so the source strides travel with the
    // extents they belong to.
    permuted[destination] = inputStrides[source];
  }
  derivation.setResultShape(std::move(shape));
  derivation.setResultStrides(std::move(permuted));
  return success();
}

LogicalResult PermuteOp::verify() {
  RankedTensorType inputType = getInput().getType();
  RankedTensorType resultType = getResult().getType();
  if (inputType.getRank() != 4 || resultType.getRank() != 4) {
    return emitOpError("requires rank-4 input and result tensors for the raw "
                       "GGML contract");
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return emitOpError("requires input and result element types to match");
  }

  ArrayAttr axes = getAxes();
  if (axes.size() != 4) {
    return emitOpError("requires exactly 4 `axes` entries");
  }

  std::array<bool, 4> seen = {false, false, false, false};
  for (Attribute attribute : axes) {
    const int64_t axis = cast<IntegerAttr>(attribute).getInt();
    if (axis < 0 || axis >= 4) {
      return emitOpError("requires each `axes` entry to be in [0, 4), got ")
             << axis;
    }
    if (seen[axis]) {
      return emitOpError("requires `axes` to be a permutation of [0, 4); "
                         "duplicate axis ")
             << axis;
    }
    seen[axis] = true;
  }
  // ggml_permute records where each source dimension is placed:
  //   result.ne[axes[source]] = input.ne[source].
  // MLIR reverses the four GGML dimensions in its ranked tensor type.
  for (int64_t ggmlInputDimension = 0; ggmlInputDimension < 4;
       ++ggmlInputDimension) {
    const int64_t inputDimension = 3 - ggmlInputDimension;
    const int64_t resultDimension =
        3 - cast<IntegerAttr>(axes[ggmlInputDimension]).getInt();
    if (!compatibleDimensions(inputType.getDimSize(inputDimension),
                              resultType.getDimSize(resultDimension))) {
      return emitOpError(
          "requires result shape to match the requested permutation");
    }
  }
  return success();
}
