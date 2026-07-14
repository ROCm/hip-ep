/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult mlir::hipsr::verifyShapeRegionStructure(Operation *op) {
  if (op->getNumRegions() == 0)
    return op->emitOpError("expected a shape region (region 0)");

  Region &shapeRegion = op->getRegion(0);
  if (!shapeRegion.hasOneBlock())
    return op->emitOpError("shape region must have exactly one block");

  Block &block = shapeRegion.front();
  if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>())
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

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.cpp.inc"
