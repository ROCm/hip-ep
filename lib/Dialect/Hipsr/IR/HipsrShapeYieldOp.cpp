/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult ShapeYieldOp::verify() {
  auto placeholder = cast<PlaceholderOp>(getOperation()->getParentOp());
  if (getShapes().size() != placeholder.getNumResults()) {
    return emitOpError(
               "must yield one extent tensor per enclosing placeholder result; "
               "expected ")
           << placeholder.getNumResults() << ", got " << getShapes().size();
  }

  // Materialize reads only the dynamic dimensions of a result type, so a shape
  // that is too short would otherwise surface as an out-of-range extent read
  // much later.
  for (auto [index, shape] : llvm::enumerate(getShapes())) {
    int64_t extents = cast<RankedTensorType>(shape.getType()).getDimSize(0);
    if (ShapedType::isDynamic(extents)) {
      continue;
    }
    int64_t rank =
        cast<RankedTensorType>(placeholder->getResult(index).getType())
            .getRank();
    if (extents != rank) {
      return emitOpError("shape #")
             << index << " holds " << extents << " extents but result #"
             << index << " has rank " << rank;
    }
  }
  return success();
}
