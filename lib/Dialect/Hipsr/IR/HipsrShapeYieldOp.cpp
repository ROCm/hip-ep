/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult ShapeYieldOp::verify() {
  // ODS already checks that `ranks` sums to the operand count, and `shapes` is
  // built by splitting the operands on `ranks`, so shapes.size() ==
  // ranks.size() and each shapes[i].size() == ranks[i] hold by construction.
  // The one thing ODS does not tie down is the element-type count, so check it
  // here: there must be one element type per result.
  size_t numResults = getRanks().size();
  size_t numTypes = getElementTypes().size();
  if (numTypes != numResults)
    return emitOpError() << "has " << numResults << " result(s) but "
                         << numTypes << " element type(s)";
  return success();
}
