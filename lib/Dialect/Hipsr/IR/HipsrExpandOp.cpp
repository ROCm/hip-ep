/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct CanonicalizeConstantShape : OpRewritePattern<ExpandOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getShape() || op.getShapeAttrAttr()) {
      return failure();
    }

    DenseIntElementsAttr denseShape;
    if (!matchPattern(op.getShape(), m_Constant(&denseShape))) {
      return failure();
    }

    auto shapeType = cast<ShapedType>(op.getShape().getType());
    if (denseShape.getNumElements() != shapeType.getDimSize(0)) {
      return failure();
    }

    SmallVector<int64_t> extents;
    extents.reserve(denseShape.getNumElements());
    for (APInt extent : denseShape.getValues<APInt>()) {
      extents.push_back(extent.getSExtValue());
    }

    rewriter.modifyOpInPlace(op, [&] {
      op.getShapeMutable().clear();
      op.setShapeAttrAttr(rewriter.getDenseI64ArrayAttr(extents));
    });
    return success();
  }
};
} // namespace

MutableOperandRange ExpandOp::getDpsInitsMutable() { return getInitMutable(); }

namespace {

struct ExpandShapeArgs : ShapeRegionArgs<ExpandOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getInput() const { return in(0); }
  Value getRequestedShape() const { return in(1); }
};

PlaceholderType getExpandPlaceholderType(ExpandOp op) {
  return op.getShapeAttrAttr() ? PlaceholderType::Normal
                               : PlaceholderType::Barrier;
}

void populateExpandShapeRegion(OpBuilder &builder, Block &block, ExpandOp op,
                               PlaceholderType placeholderType) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);

  ExpandShapeArgs args(placeholderType, block);
  Location loc = op.getLoc();
  MLIRContext *ctx = builder.getContext();
  Value inputShape;
  SmallVector<Value> requestedExtents;
  if (placeholderType == PlaceholderType::Normal) {
    DenseI64ArrayAttr shapeAttr = op.getShapeAttrAttr();
    inputShape = args.getInput();
    requestedExtents.reserve(shapeAttr.size());
    for (int64_t extent : shapeAttr.asArrayRef()) {
      requestedExtents.push_back(
          builder.create<arith::ConstantIndexOp>(loc, extent));
    }
  } else {
    Value input = args.getInput();
    Value requestedShape = args.getRequestedShape();
    inputShape = builder.create<shape::ShapeOfOp>(loc, input);

    auto requestedShapeType = cast<RankedTensorType>(requestedShape.getType());
    int64_t requestedRank = requestedShapeType.getDimSize(0);
    requestedExtents.reserve(requestedRank);
    for (int64_t i : llvm::seq<int64_t>(0, requestedRank)) {
      Value index = builder.create<arith::ConstantIndexOp>(loc, i);
      Value extent = builder.create<tensor::ExtractOp>(loc, requestedShape,
                                                       ValueRange{index});
      requestedExtents.push_back(builder.create<arith::IndexCastOp>(
          loc, builder.getIndexType(), extent));
    }
  }

  auto shapeType = shape::ShapeType::get(ctx);
  Value requestedShape = builder.create<shape::FromExtentsOp>(
      loc, shapeType, ValueRange{requestedExtents});
  Value witness = builder.create<shape::CstrBroadcastableOp>(loc, inputShape,
                                                             requestedShape);
  auto assuming = builder.create<shape::AssumingOp>(
      loc, witness,
      [&](OpBuilder &nestedBuilder,
          Location nestedLoc) -> SmallVector<Value, 2> {
        Value broadcastShape = nestedBuilder.create<shape::BroadcastOp>(
            nestedLoc, shapeType, inputShape, requestedShape, StringAttr{});
        return {broadcastShape};
      });

  builder.create<ShapeYieldOp>(loc, assuming.getResult(0));
}

} // namespace

void mlir::hipsr::populateExpandShapeRegionPatterns(
    ShapeRegionPopulationPatternSet &patterns) {
  patterns.add<ExpandOp, getExpandPlaceholderType, populateExpandShapeRegion>();
}

void ExpandOp::getCanonicalizationPatterns(RewritePatternSet &patterns,
                                           MLIRContext *context) {
  patterns.add<CanonicalizeConstantShape>(context);
}

LogicalResult ExpandOp::verify() {
  bool hasShapeOperand = static_cast<bool>(getShape());
  bool hasShapeAttr = static_cast<bool>(getShapeAttrAttr());
  if (hasShapeOperand && hasShapeAttr) {
    return emitOpError(
        "cannot have both shape operand and shape_attr attribute");
  }
  if (!hasShapeOperand && !hasShapeAttr) {
    return emitOpError(
        "must have either shape operand or shape_attr attribute");
  }

  auto inputType = cast<ShapedType>(getInput().getType());
  auto outputType = cast<ShapedType>(getInit().getType());

  int64_t shapeLength;
  if (hasShapeOperand) {
    auto shapeType = cast<ShapedType>(getShape().getType());
    if (shapeType.getRank() != 1) {
      return emitOpError("shape must be rank-1");
    }
    if (!shapeType.getElementType().isInteger(64)) {
      return emitOpError("shape element type must be i64");
    }
    if (shapeType.isDynamicDim(0)) {
      return emitOpError("shape length must be static");
    }
    shapeLength = shapeType.getDimSize(0);
  } else {
    shapeLength = static_cast<int64_t>(getShapeAttrAttr().asArrayRef().size());
  }

  if (inputType.getElementType() != outputType.getElementType()) {
    return emitOpError("input and output element types must match");
  }

  int64_t expectedRank = std::max(inputType.getRank(), shapeLength);
  if (outputType.getRank() != expectedRank) {
    return emitOpError("output rank must equal max(input rank, shape length); "
                       "expected ")
           << expectedRank << ", got " << outputType.getRank();
  }
  return success();
}
