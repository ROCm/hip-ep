/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrMatMulOp.cpp.inc"

// DestinationStyleOpInterface: the single init operand is the DPS out.
MutableOperandRange MatMulOp::getDpsInitsMutable() { return getInitMutable(); }

// Single source of truth for the output shape [M, N]:
//   M = dim(lhs, 0), N = dim(rhs, 1).
void MatMulOp::populateShapeRegion(OpBuilder &builder, Region &shapeRegion) {
  OpBuilder::InsertionGuard guard(builder);
  Block *body = builder.createBlock(&shapeRegion);
  builder.setInsertionPointToStart(body);

  Location loc = getLoc();
  Value m = builder.create<tensor::DimOp>(loc, getLhs(), 0);
  Value n = builder.create<tensor::DimOp>(loc, getRhs(), 1);
  builder.create<ShapeYieldOp>(loc, ValueRange{m, n});
}
