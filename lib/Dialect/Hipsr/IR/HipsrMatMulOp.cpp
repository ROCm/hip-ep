/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulation.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Shape/IR/Shape.h"

using namespace mlir;
using namespace mlir::hipsr;

MutableOperandRange MatMulOp::getDpsInitsMutable() { return getInitMutable(); }

namespace {

struct MatMulShapeArgs : ShapeRegionArgs<MatMulOp> {
  using ShapeRegionArgs::ShapeRegionArgs;
  Value getA() const { return in(0); }
  Value getB() const { return in(1); }
};

PlaceholderType getMatMulPlaceholderType(MatMulOp) {
  return PlaceholderType::Normal;
}

void populateMatMulShapeRegion(OpBuilder &builder, Block &block, MatMulOp op,
                               PlaceholderType placeholderType) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&block);

  MatMulShapeArgs args(placeholderType, block);
  Location loc = op.getLoc();
  MLIRContext *ctx = builder.getContext();
  Value aShape = args.getA();
  Value bShape = args.getB();
  auto shapeType = shape::ShapeType::get(ctx);
  auto witnessType = shape::WitnessType::get(ctx);
  Value one = builder.create<arith::ConstantIndexOp>(loc, 1);
  Value minusTwo = builder.create<arith::ConstantIndexOp>(loc, -2);

  auto getRank = [&](Value shape) {
    Value rank =
        builder.create<shape::RankOp>(loc, shape::SizeType::get(ctx), shape);
    return builder.create<shape::SizeToIndexOp>(loc, rank);
  };
  Value aIs1D = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              getRank(aShape), one);
  Value bIs1D = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq,
                                              getRank(bShape), one);

  Value emptyShape =
      builder.create<shape::FromExtentsOp>(loc, shapeType, ValueRange{});
  Value oneShape =
      builder.create<shape::FromExtentsOp>(loc, shapeType, ValueRange{one});

  // Promote vectors to matrices so the remaining algebra is rank-agnostic.
  Value promotedA =
      builder.create<shape::ConcatOp>(loc, shapeType, oneShape, aShape);
  Value promotedB =
      builder.create<shape::ConcatOp>(loc, shapeType, bShape, oneShape);
  Value normalizedA =
      builder.create<arith::SelectOp>(loc, aIs1D, promotedA, aShape);
  Value normalizedB =
      builder.create<arith::SelectOp>(loc, bIs1D, promotedB, bShape);

  auto aParts = builder.create<shape::SplitAtOp>(
      loc, TypeRange{shapeType, shapeType}, normalizedA, minusTwo);
  auto bParts = builder.create<shape::SplitAtOp>(
      loc, TypeRange{shapeType, shapeType}, normalizedB, minusTwo);
  auto aMatrix = builder.create<shape::SplitAtOp>(
      loc, TypeRange{shapeType, shapeType}, aParts.getTail(), one);
  auto bMatrix = builder.create<shape::SplitAtOp>(
      loc, TypeRange{shapeType, shapeType}, bParts.getTail(), one);

  Value kWitness = builder.create<shape::CstrEqOp>(
      loc, witnessType, ValueRange{aMatrix.getTail(), bMatrix.getHead()});
  Value batchWitness = builder.create<shape::CstrBroadcastableOp>(
      loc, aParts.getHead(), bParts.getHead());
  Value witness = builder.create<shape::AssumingAllOp>(
      loc, witnessType, ValueRange{kWitness, batchWitness});

  auto assuming = builder.create<shape::AssumingOp>(
      loc, witness,
      [&](OpBuilder &nestedBuilder,
          Location nestedLoc) -> SmallVector<Value, 2> {
        Value batchShape = nestedBuilder.create<shape::BroadcastOp>(
            nestedLoc, shapeType,
            ValueRange{aParts.getHead(), bParts.getHead()},
            /*error=*/nullptr);
        Value mShape = nestedBuilder.create<arith::SelectOp>(
            nestedLoc, aIs1D, emptyShape, aMatrix.getHead());
        Value nShape = nestedBuilder.create<arith::SelectOp>(
            nestedLoc, bIs1D, emptyShape, bMatrix.getTail());
        Value matrixShape = nestedBuilder.create<shape::ConcatOp>(
            nestedLoc, shapeType, mShape, nShape);
        Value resultShape = nestedBuilder.create<shape::ConcatOp>(
            nestedLoc, shapeType, batchShape, matrixShape);
        return {resultShape};
      });

  builder.create<ShapeYieldOp>(loc, assuming.getResult(0));
}

} // namespace

void mlir::hipsr::populateMatMulShapeRegionPatterns(
    ShapeRegionPopulationPatternSet &patterns) {
  patterns.add<MatMulOp, getMatMulPlaceholderType, populateMatMulShapeRegion>();
}

// A and B must be at least 1-D because matmul needs a contraction dimension.
LogicalResult MatMulOp::verify() {
  if (cast<ShapedType>(getA().getType()).getRank() < 1) {
    return emitOpError("operand A must be at least 1-D");
  }
  if (cast<ShapedType>(getB().getType()).getRank() < 1) {
    return emitOpError("operand B must be at least 1-D");
  }
  return success();
}
