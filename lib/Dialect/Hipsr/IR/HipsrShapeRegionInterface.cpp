/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.h"

#include "llvm/ADT/STLExtras.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
// Get the `hipsr.shape_yield` at the end of a (populated) shape region.
ShapeYieldOp getShapeYieldOp(Region &shapeRegion) {
  assert(!shapeRegion.empty() &&
         "shape region must be populated before reading its shape_yield");
  return cast<ShapeYieldOp>(shapeRegion.front().getTerminator());
}
} // namespace

LogicalResult mlir::hipsr::verifyShapeRegionStructure(Operation *op) {
  if (op->getNumRegions() == 0)
    return op->emitOpError("expected a shape region (region 0)");

  Region &shapeRegion = op->getRegion(0);

  // The shape region is optional: an empty region (0 blocks) means the op
  // carries no shape computation. When present it must have exactly one block.
  if (shapeRegion.empty())
    return success();

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
           << "'; a non-empty shape region block must end with "
              "hipsr.shape_yield";

  return success();
}

SmallVector<SmallVector<Value>>
mlir::hipsr::getShapeRegionResultShapes(Region &shapeRegion) {
  ShapeYieldOp yieldOp = getShapeYieldOp(shapeRegion);
  SmallVector<SmallVector<Value>> dims;
  for (OperandRange group : yieldOp.getShapes())
    dims.emplace_back(group.begin(), group.end());
  return dims;
}

SmallVector<RankedTensorType>
mlir::hipsr::getShapeRegionResultTypes(Region &shapeRegion) {
  ShapeYieldOp yieldOp = getShapeYieldOp(shapeRegion);
  SmallVector<RankedTensorType> types;
  for (auto [group, elemType] :
       llvm::zip_equal(yieldOp.getShapes(),
                       yieldOp.getElementTypes().getAsValueRange<TypeAttr>())) {
    // Mark every extent dynamic and keep only the rank and element type. When a
    // dim value is a constant, the tensor.empty canonicalizer folds it back to
    // a static extent later, so nothing is lost.
    SmallVector<int64_t> shape(group.size(), ShapedType::kDynamic);
    types.push_back(RankedTensorType::get(shape, elemType));
  }
  return types;
}

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.cpp.inc"
