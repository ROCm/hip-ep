/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct ExpandShapeArgs : ShapeRegionArgs<ExpandOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getInput() const { return in(0); }
  Value getRequestedShape() const { return in(1); }
};
} // namespace

MutableOperandRange ExpandOp::getDpsInitsMutable() { return getInitMutable(); }

LogicalResult ExpandOp::verify() {
  if (getNumResults() > 1) {
    return emitOpError("expected at most one tensor result");
  }

  auto inputType = cast<ShapedType>(getInput().getType());
  auto shapeType = cast<ShapedType>(getShape().getType());
  auto outputType = cast<ShapedType>(getInit().getType());

  if (shapeType.getRank() != 1) {
    return emitOpError("shape must be rank-1");
  }
  if (!shapeType.getElementType().isInteger(64)) {
    return emitOpError("shape element type must be i64");
  }
  if (shapeType.isDynamicDim(0)) {
    return emitOpError("shape length must be static");
  }
  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }

  int64_t expectedRank = std::max(inputType.getRank(), shapeType.getDimSize(0));
  if (outputType.getRank() != expectedRank) {
    return emitOpError("output rank must equal max(input rank, shape length); "
                       "expected ")
           << expectedRank << ", got " << outputType.getRank();
  }
  return success();
}

bool ExpandOp::isStartBarrier() {
  // Direct dense producers prove every requested extent without runtime or
  // external-storage access. Other producers remain barriers.
  DenseIntElementsAttr denseAttr;
  if (matchPattern(getShape(), m_Constant(&denseAttr))) {
    return false;
  }

  if (auto constant = getShape().getDefiningOp<ConstantOp>()) {
    return !isa_and_nonnull<DenseIntElementsAttr>(constant.getValueAttr());
  }
  return true;
}

// Before:
//   hipsr.expand(%ctx) ins(%input, %shape) outs(%init)
// After:
//   hipsr.expand ... shape_region {
//     %broadcast = shape.broadcast %input_shape, %requested_shape
//     hipsr.shape_yield (%dims) : [element_type]
//   }
void ExpandOp::populateShapeRegion(OpBuilder &builder, Block &shapeBlock) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = getLoc();
  MLIRContext *ctx = builder.getContext();
  ExpandShapeArgs args{shapeBlock};
  Value input = args.getInput();
  Value requestedShapeArg = args.getRequestedShape();
  auto inputType = cast<ShapedType>(input.getType());
  auto requestedShapeType = cast<ShapedType>(requestedShapeArg.getType());
  int64_t resultRank =
      std::max(inputType.getRank(), requestedShapeType.getDimSize(0));

  Value inputShape = builder.create<shape::ShapeOfOp>(loc, input);
  SmallVector<Value> requestedExtents;
  requestedExtents.reserve(requestedShapeType.getDimSize(0));
  for (int64_t i : llvm::seq<int64_t>(0, requestedShapeType.getDimSize(0))) {
    Value index = builder.create<arith::ConstantIndexOp>(loc, i);
    Value extent;
    if (isa<RankedTensorType>(requestedShapeArg.getType())) {
      extent = builder.create<tensor::ExtractOp>(loc, requestedShapeArg,
                                                 ValueRange{index});
    } else {
      extent = builder.create<memref::LoadOp>(loc, requestedShapeArg,
                                              ValueRange{index});
    }
    requestedExtents.push_back(builder.create<arith::IndexCastOp>(
        loc, builder.getIndexType(), extent));
  }

  auto shapeType = shape::ShapeType::get(ctx);
  Value requestedShape = builder.create<shape::FromExtentsOp>(
      loc, shapeType, ValueRange{requestedExtents});
  Value witness = builder.create<shape::CstrBroadcastableOp>(loc, inputShape,
                                                             requestedShape);
  auto assuming = builder.create<shape::AssumingOp>(
      loc, witness, [&](OpBuilder &b, Location) -> SmallVector<Value, 2> {
        Value broadcastShape = b.create<shape::BroadcastOp>(
            loc, shapeType, inputShape, requestedShape, StringAttr{});
        SmallVector<Value, 2> resultExtents;
        resultExtents.reserve(resultRank);
        for (int64_t i : llvm::seq<int64_t>(0, resultRank)) {
          Value extent = b.create<shape::GetExtentOp>(loc, broadcastShape, i);
          resultExtents.push_back(b.create<shape::SizeToIndexOp>(loc, extent));
        }
        return resultExtents;
      });

  builder.create<ShapeYieldOp>(loc, ArrayRef<ValueRange>{assuming.getResults()},
                               TypeRange{inputType.getElementType()});
}
