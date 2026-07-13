/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "hip/Dialect/Hipsr/IR/HipsrInfrastructureOps.h"

#include "llvm/ADT/DenseSet.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult mlir::hipsr::verifyShapeRegionStructure(Operation *op) {
  if (op->getNumRegions() == 0)
    return op->emitOpError("expected a shape region (region 0)");

  Region &shapeRegion = op->getRegion(0);
  if (!shapeRegion.hasOneBlock())
    return op->emitOpError("shape region must have exactly one block");

  Block &block = shapeRegion.front();
  if (block.empty() || !block.back().hasTrait<OpTrait::IsTerminator>())
    return op->emitOpError("shape region block must end with a terminator");

  if (!isa<ShapeYieldOp>(block.back()))
    return op->emitOpError(
               "shape region must terminate with hipsr.shape_yield, "
               "got '")
           << block.back().getName()
           << "'; did you forget the "
              "SingleBlockImplicitTerminator<\"ShapeYieldOp\"> trait?";

  return success();
}

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

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.cpp.inc"
