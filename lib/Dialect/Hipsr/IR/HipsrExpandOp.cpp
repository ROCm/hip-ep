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
  // A directly defined dense constant makes every requested extent known at
  // compile time. All other shape sources require a runtime barrier.
  DenseIntElementsAttr denseAttr;
  if (matchPattern(getShape(), m_Constant(&denseAttr))) {
    return false;
  }

  if (auto constant = getShape().getDefiningOp<ConstantOp>()) {
    return !isa_and_nonnull<DenseIntElementsAttr>(constant.getValueAttr());
  }
  return true;
}

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

  Value inputShape = shape::ShapeOfOp::create(builder, loc, input);
  SmallVector<Value> requestedExtents;
  requestedExtents.reserve(requestedShapeType.getDimSize(0));
  for (int64_t i : llvm::seq<int64_t>(0, requestedShapeType.getDimSize(0))) {
    Value index = arith::ConstantIndexOp::create(builder, loc, i);
    Value extent;
    if (isa<RankedTensorType>(requestedShapeArg.getType())) {
      extent = tensor::ExtractOp::create(builder, loc, requestedShapeArg,
                                         ValueRange{index});
    } else {
      extent = memref::LoadOp::create(builder, loc, requestedShapeArg,
                                      ValueRange{index});
    }
    requestedExtents.push_back(arith::IndexCastOp::create(
        builder, loc, builder.getIndexType(), extent));
  }

  auto shapeType = shape::ShapeType::get(ctx);
  Value requestedShape = shape::FromExtentsOp::create(
      builder, loc, shapeType, ValueRange{requestedExtents});

  // ONNX Expand uses right-aligned multidirectional broadcasting.
  Value witness = shape::CstrBroadcastableOp::create(builder, loc, inputShape,
                                                     requestedShape);
  auto assuming = shape::AssumingOp::create(
      builder, loc, witness,
      [&](OpBuilder &b, Location) -> SmallVector<Value, 2> {
        Value broadcastShape = shape::BroadcastOp::create(
            b, loc, shapeType, inputShape, requestedShape, StringAttr{});
        SmallVector<Value, 2> resultExtents;
        resultExtents.reserve(resultRank);
        for (int64_t i : llvm::seq<int64_t>(0, resultRank)) {
          Value extent = shape::GetExtentOp::create(b, loc, broadcastShape, i);
          resultExtents.push_back(shape::SizeToIndexOp::create(b, loc, extent));
        }
        return resultExtents;
      });

  ShapeYieldOp::create(builder, loc,
                       ArrayRef<ValueRange>{assuming.getResults()},
                       TypeRange{inputType.getElementType()});
}
