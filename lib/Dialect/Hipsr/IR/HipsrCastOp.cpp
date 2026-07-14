/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrCastOp.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrCastOp.cpp.inc"

// DestinationStyleOpInterface: the single init operand is the DPS out.
MutableOperandRange CastOp::getDpsInitsMutable() { return getInitMutable(); }

// Single source of truth for the output shape: identical to the input shape.
void CastOp::populateShapeRegion(OpBuilder &builder, Region &shapeRegion) {
  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&shapeRegion);
  builder.setInsertionPointToStart(body);

  Location loc = getLoc();
  auto shapedTy = cast<ShapedType>(getInput().getType());
  SmallVector<Value> dims;
  dims.reserve(shapedTy.getRank());
  for (int64_t i = 0; i < shapedTy.getRank(); ++i)
    dims.push_back(builder.create<tensor::DimOp>(loc, getInput(), i));
  builder.create<ShapeYieldOp>(loc, dims);
}
