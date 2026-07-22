/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrCastOp.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionHelpers.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hipsr;

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrCastOp.cpp.inc"

namespace {
// Block args follow the operands: 0 is ctx (unused), 1 the input.
struct CastShapeArgs : ShapeRegionArgs<CastOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getInput() const { return ins(1); }
};
} // namespace

MutableOperandRange CastOp::getDpsInitsMutable() { return getInitMutable(); }

// Yields the input shape with the output element type. Uses the shape dialect
// so one path covers both tensor and memref inputs and static dims fold away.
void CastOp::populateShapeRegion(OpBuilder &builder, Block &shapeBlock) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = getLoc();
  Value input = CastShapeArgs{shapeBlock}.getInput();
  auto shapedTy = cast<ShapedType>(input.getType());
  Value shape = builder.create<shape::ShapeOfOp>(loc, input);
  SmallVector<Value> dims;
  dims.reserve(shapedTy.getRank());
  for (int64_t i : llvm::seq<int64_t>(0, shapedTy.getRank())) {
    Value idx = builder.create<arith::ConstantIndexOp>(loc, i);
    dims.push_back(builder.create<shape::GetExtentOp>(loc, shape, idx));
  }
  Type elemTy = cast<ShapedType>(getInit().getType()).getElementType();
  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{ValueRange(dims)},
                               TypeRange{elemTy});
}
