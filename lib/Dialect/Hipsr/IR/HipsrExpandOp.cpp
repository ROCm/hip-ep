/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Matchers.h"

#include "llvm/ADT/Sequence.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::hipsr;

namespace {
struct ExpandShapeArgs : PlaceholderShapeRegionArgs {
  explicit ExpandShapeArgs(Block &block) : PlaceholderShapeRegionArgs(block) {}
  using PlaceholderShapeRegionArgs::getPlaceholderType;
  Value getInput() const { return in(0); }
  Value getRequestedShape() const { return in(1); }
};

struct CanonicalizeConstantShape : OpRewritePattern<ExpandOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(ExpandOp op,
                                PatternRewriter &rewriter) const override {
    if (!op.getShape() || op.getShapeAttrAttr() ||
        !op.getShapeRegion().empty()) {
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

bool ExpandOp::isStartBarrier() { return getShapeAttrAttr() == nullptr; }

void ExpandOp::populateShapeRegion(OpBuilder &, Block &) {}

namespace mlir {
namespace hipsr {

LogicalResult populateExpandShapeRegion(OpBuilder &builder, Block &shapeBlock,
                                        ExpandOp op) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&shapeBlock);

  Location loc = op.getLoc();
  ExpandShapeArgs args{shapeBlock};
  Value inputShape;
  SmallVector<Value> requestedExtents;
  if (args.getPlaceholderType() == PlaceholderType::Normal) {
    inputShape = args.getInput();
    DenseI64ArrayAttr shapeAttr = op.getShapeAttrAttr();
    requestedExtents.reserve(shapeAttr.asArrayRef().size());
    for (int64_t extent : shapeAttr.asArrayRef()) {
      requestedExtents.push_back(
          arith::ConstantIndexOp::create(builder, loc, extent));
    }
  } else {
    inputShape = shape::ShapeOfOp::create(builder, loc, args.getInput());
    Value requestedShapeArg = args.getRequestedShape();
    auto requestedShapeType = cast<ShapedType>(requestedShapeArg.getType());
    int64_t requestedRank = requestedShapeType.getDimSize(0);
    requestedExtents.reserve(requestedRank);
    for (int64_t i : llvm::seq<int64_t>(0, requestedRank)) {
      Value index = arith::ConstantIndexOp::create(builder, loc, i);
      Value extent = tensor::ExtractOp::create(builder, loc, requestedShapeArg,
                                               ValueRange{index});
      requestedExtents.push_back(arith::IndexCastOp::create(
          builder, loc, builder.getIndexType(), extent));
    }
  }

  auto shapeType = shape::ShapeType::get(builder.getContext());
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
        return SmallVector<Value, 2>{broadcastShape};
      });

  ShapeYield2Op::create(builder, loc, assuming.getResults());
  return success();
}

} // namespace hipsr
} // namespace mlir
