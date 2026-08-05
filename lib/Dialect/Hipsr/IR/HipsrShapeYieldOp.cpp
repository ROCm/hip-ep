/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult ShapeYieldOp::verify() {
  auto placeholder = cast<PlaceholderOp>(getOperation()->getParentOp());
  if (getShapes().size() != placeholder.getNumResults()) {
    return emitOpError(
               "must yield one !shape.shape per enclosing placeholder result; "
               "expected ")
           << placeholder.getNumResults() << ", got " << getShapes().size();
  }
  return success();
}
