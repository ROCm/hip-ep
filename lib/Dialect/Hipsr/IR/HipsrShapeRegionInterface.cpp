/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrStartBarrierInterface.h"

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

SmallVector<SmallVector<Value>> resultShapesOf(Region &shapeRegion) {
  ShapeYieldOp yieldOp = getShapeYieldOp(shapeRegion);
  SmallVector<SmallVector<Value>> dims;
  for (OperandRange group : yieldOp.getShapes()) {
    dims.emplace_back(group.begin(), group.end());
  }
  return dims;
}

SmallVector<RankedTensorType> resultTypesOf(Region &shapeRegion) {
  ShapeYieldOp yieldOp = getShapeYieldOp(shapeRegion);
  SmallVector<RankedTensorType> types;
  for (auto [group, elemType] :
       llvm::zip_equal(yieldOp.getShapes(),
                       yieldOp.getElementTypes().getAsValueRange<TypeAttr>())) {
    SmallVector<int64_t> shape(group.size(), ShapedType::kDynamic);
    types.push_back(RankedTensorType::get(shape, elemType));
  }
  return types;
}

} // namespace

LogicalResult mlir::hipsr::verifyShapeRegionStructure(Operation *op) {
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

  if (isa<DestinationStyleOpInterface>(op)) {
    SmallVector<Value> expected =
        getShapeRegionArgOperands(cast<ShapeRegionInterface>(op));
    if (block.getNumArguments() != expected.size()) {
      return op->emitOpError("shape region block must have one argument per "
                             "shape-region operand (expected ")
             << expected.size() << ", got " << block.getNumArguments() << ")";
    }
    for (auto [i, operand] : llvm::enumerate(expected)) {
      Type argType = block.getArgument(i).getType();
      Type operandType = operand.getType();
      if (argType != operandType) {
        return op->emitOpError("shape region block argument ")
               << i << " type " << argType << " does not match expected type "
               << operandType;
      }
    }
  }

  return success();
}

SmallVector<Value>
mlir::hipsr::getShapeRegionArgOperands(ShapeRegionInterface op) {
  SmallVector<Value> args;
  auto dps = dyn_cast<DestinationStyleOpInterface>(op.getOperation());
  if (!dps) {
    return args;
  }

  SmallVector<Value> ins = dps.getDpsInputs();
  Operation *raw = op.getOperation();
  if (isa<StartBarrierInterface>(raw) && !ins.empty()) {
    args.push_back(ins.front()); // ctx
  }
  if (!ins.empty()) {
    llvm::append_range(args, llvm::drop_begin(ins)); // data inputs
  }
  if (isa<EndBarrierInterface>(raw)) {
    llvm::append_range(args, dps.getDpsInits()); // outs
  }
  return args;
}

Region &mlir::hipsr::getShapeRegion(ShapeRegionInterface op) {
  return op->getRegion(0);
}

Region &mlir::hipsr::getCapacityShapeRegion(ShapeRegionInterface op) {
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
