/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrCastOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrCastOp.cpp.inc"

// DestinationStyleOpInterface: the single init operand is the DPS out.
MutableOperandRange CastOp::getDpsInitsMutable() { return getInitMutable(); }

// Single source of truth for the output shape: identical to the input shape.
// Uses the shape dialect so it works uniformly for both tensor and memref
// inputs (Hipsr_TensorOrDeviceMemRef); static dimensions fold automatically.
void CastOp::populateShapeRegion(OpBuilder &builder, Region &shapeRegion) {
  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&shapeRegion);
  builder.setInsertionPointToStart(body);

  Location loc = getLoc();
  auto shapedTy = cast<ShapedType>(getInput().getType());
  Value shape = builder.create<shape::ShapeOfOp>(loc, getInput());
  SmallVector<Value> dims;
  dims.reserve(shapedTy.getRank());
  for (int64_t i = 0; i < shapedTy.getRank(); ++i) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    dims.push_back(builder.create<shape::GetExtentOp>(loc, shape, idx));
  }
  // Single result: one dim group and its (output) element type.
  Type elemTy = cast<ShapedType>(getResult(0).getType()).getElementType();
  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{ValueRange(dims)},
                               TypeRange{elemTy});
}
