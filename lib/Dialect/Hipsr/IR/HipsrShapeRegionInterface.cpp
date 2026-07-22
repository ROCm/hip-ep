/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeYieldOp.h"

#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
ShapeYieldOp getShapeYieldOp(Region &shapeRegion) {
  assert(!shapeRegion.empty() &&
         "shape region must be populated before reading its shape_yield");
  return cast<ShapeYieldOp>(shapeRegion.front().getTerminator());
}

// Take a bare region so one body serves both the shape and capacity regions.
SmallVector<SmallVector<Value>> resultShapesOf(Region &shapeRegion) {
  ShapeYieldOp yieldOp = getShapeYieldOp(shapeRegion);
  SmallVector<SmallVector<Value>> dims;
  for (OperandRange group : yieldOp.getShapes())
    dims.emplace_back(group.begin(), group.end());
  return dims;
}

SmallVector<RankedTensorType> resultTypesOf(Region &shapeRegion) {
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

} // namespace

LogicalResult mlir::hipsr::verifyShapeRegionStructure(Operation *op) {
  // The two builtin traits do the isolation and single-block checks in their
  // own verifiers; here we only require an implementing op declares them.
  if (!op->hasTrait<mlir::OpTrait::IsIsolatedFromAbove>()) {
    return op->emitOpError(
        "ShapeRegionInterface requires the IsolatedFromAbove trait");
  }

  if (!op->hasTrait<mlir::OpTrait::SingleBlock>()) {
    return op->emitOpError(
        "ShapeRegionInterface requires the SingleBlock trait");
  }

  if (op->getNumRegions() == 0) {
    return op->emitOpError("expected a shape region (region 0)");
  }

  Region &shapeRegion = op->getRegion(0);

  // The region is optional: empty means no shape computation yet. Otherwise
  // SingleBlock guarantees one block, so verify only its contents below.
  if (shapeRegion.empty()) {
    return success();
  }

  Block &block = shapeRegion.front();
  if (block.empty() || !block.back().hasTrait<mlir::OpTrait::IsTerminator>()) {
    return op->emitOpError("shape region block must end with a terminator");
  }

  if (!isa<ShapeYieldOp>(block.back())) {
    return op->emitOpError("shape region must terminate with "
                           "hipsr.shape_yield, got '")
           << block.back().getName() << "'";
  }

  // Entry-block args must mirror the DPS inputs one-for-one; that is how the
  // isolated region reaches the inputs without capturing the op's operands.
  if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
    SmallVector<OpOperand *> inputs = dps.getDpsInputOperands();
    if (block.getNumArguments() != inputs.size()) {
      return op->emitOpError("shape region block must have one argument per "
                             "DPS input (expected ")
             << inputs.size() << ", got " << block.getNumArguments() << ")";
    }
    for (auto [i, in] : llvm::enumerate(inputs)) {
      Type argType = block.getArgument(i).getType();
      Type inType = in->get().getType();
      if (argType != inType) {
        return op->emitOpError("shape region block argument ")
               << i << " type " << argType << " does not match DPS input type "
               << inType;
      }
    }
  }

  return success();
}

Region &mlir::hipsr::getShapeRegion(ShapeRegionInterface op) {
  return op->getRegion(0);
}

Region &mlir::hipsr::getCapacityShapeRegion(ShapeRegionInterface op) {
  // Region 1 exists only on EndBarrier ops; asking any other op for it is a
  // caller bug, so fail with the op name instead of reading a missing region.
  if (op->getNumRegions() <= 1) {
    std::string msg;
    llvm::raw_string_ostream(msg)
        << op->getName() << " has no capacity shape region (region 1)";
    llvm::report_fatal_error(llvm::StringRef(msg));
  }
  return op->getRegion(1);
}

SmallVector<SmallVector<Value>>
mlir::hipsr::getShapeRegionResultShapes(ShapeRegionInterface op) {
  return resultShapesOf(getShapeRegion(op));
}

SmallVector<RankedTensorType>
mlir::hipsr::getShapeRegionResultTypes(ShapeRegionInterface op) {
  return resultTypesOf(getShapeRegion(op));
}

SmallVector<SmallVector<Value>>
mlir::hipsr::getCapacityShapeRegionResultShapes(ShapeRegionInterface op) {
  return resultShapesOf(getCapacityShapeRegion(op));
}

SmallVector<RankedTensorType>
mlir::hipsr::getCapacityShapeRegionResultTypes(ShapeRegionInterface op) {
  return resultTypesOf(getCapacityShapeRegion(op));
}

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.cpp.inc"
