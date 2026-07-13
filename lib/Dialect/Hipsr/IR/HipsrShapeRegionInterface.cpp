/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.h"

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
      // Block arguments (EndBarrier ops receive their results): fine.
      if (isa<BlockArgument>(operand))
        continue;
      // Values (op results or block arguments) defined inside the shape
      // region, directly or in a nested region (e.g. an scf.for used to
      // compute a dim), are always fine. This also covers the shape region's
      // own block arguments, which EndBarrier ops receive their results
      // through; block arguments captured from an enclosing region are not an
      // ancestor of the shape region and fall through to the operand check.
      Region *definingRegion = operand.getParentRegion();
      if (definingRegion && shapeRegion.isAncestor(definingRegion))
        continue;
      // Otherwise the value must be one of the op's own operands.
      if (!allowedValues.contains(operand)) {
        op->emitOpError("shape region references disallowed outer value")
                .attachNote(innerOp->getLoc())
            << "used here by '" << innerOp->getName() << "'";
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });

  return failure(walkResult.wasInterrupted());
}

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.cpp.inc"
