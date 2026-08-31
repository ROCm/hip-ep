/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct NonZeroPlaceholderShapeArgs : PlaceholderShapeRegionArgs {
  Value getInput() const { return in(0); }
};
} // namespace

namespace mlir {
namespace hipsr {

// The indices hold one row per input axis and, in the worst case, one column
// per input element. The count is one number.
LogicalResult populateNonZeroShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                         NonZeroOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = op.getLoc();
  NonZeroPlaceholderShapeArgs args{shapeBlock};
  Value inputShape = args.getInput();

  int64_t rank = cast<ShapedType>(op.getInput().getType()).getRank();
  Value rows = arith::ConstantIndexOp::create(builder, loc, rank);
  Value capacity = shape::NumElementsOp::create(builder, loc, inputShape);
  Value indicesShape =
      createExtentTensor(builder, loc, ValueRange{rows, capacity});
  Value one = arith::ConstantIndexOp::create(builder, loc, 1);
  Value countShape = createExtentTensor(builder, loc, ValueRange{one});
  ShapeYieldOp::create(builder, loc, ValueRange{indicesShape, countShape});
  return success();
}

} // namespace hipsr
} // namespace mlir

MutableOperandRange NonZeroOp::getDpsInitsMutable() {
  // The two inits are adjacent operands, and no generated accessor spans both.
  return MutableOperandRange(getOperation(),
                             getIndicesInitMutable().getOperandNumber(),
                             /*length=*/2);
}

LogicalResult NonZeroOp::verify() {
  auto inputType = cast<ShapedType>(getInput().getType());
  auto indicesType = cast<ShapedType>(getIndicesInit().getType());
  auto countType = cast<ShapedType>(getCountInit().getType());

  if (!indicesType.getElementType().isInteger(64)) {
    return emitOpError("indices element type must be i64");
  }
  if (!countType.getElementType().isInteger(64)) {
    return emitOpError("count element type must be i64");
  }
  if (indicesType.getRank() != 2) {
    return emitOpError("indices must be rank-2: one row per input axis, one "
                       "column per position found");
  }
  // A dynamic row count fails this too: the input rank is known here.
  if (indicesType.getDimSize(0) != inputType.getRank()) {
    return emitOpError("indices must have one row per input axis; input rank "
                       "is ")
           << inputType.getRank();
  }
  if (countType.getRank() != 1 || countType.getDimSize(0) != 1) {
    return emitOpError("count must be a static single-element vector");
  }
  return success();
}
