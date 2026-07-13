/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrInterfaces.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult mlir::hipsr::verifyShapeRegionScoping(Operation *op) {
  Region &shapeRegion = op->getRegion(0);
  if (shapeRegion.empty())
    return success();

  // Values the shape region is allowed to reference from outside itself: the
  // op's own operands (%ctx, inputs, outs, ...).
  DenseSet<Value> allowedValues;
  for (Value operand : op->getOperands())
    allowedValues.insert(operand);

  auto walkResult = shapeRegion.walk([&](Operation *innerOp) -> WalkResult {
    for (Value operand : innerOp->getOperands()) {
      // Defined inside the shape region: always fine.
      if (operand.getParentRegion() == &shapeRegion)
        continue;
      // Block arguments (EndBarrier ops receive their results): fine.
      if (isa<BlockArgument>(operand))
        continue;
      // Otherwise must be one of the op's operands.
      if (!allowedValues.contains(operand))
        return op->emitOpError("shape region uses disallowed outer value");
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

#include "hip/Dialect/Hipsr/IR/HipsrInterfaces.cpp.inc"
