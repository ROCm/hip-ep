/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "rounded_scaled_add_fusion.hpp"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace hip::fusion_transform {
namespace {
using namespace mlir;

RankedTensorType plain(Value value, bool half) {
  if (!value)
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() > 4 || type.getEncoding() ||
      (half ? !type.getElementType().isF16() : !type.getElementType().isF32()))
    return {};
  return type;
}

bool onlyDataUser(Value value, Operation *consumer) {
  unsigned count = 0;
  for (OpOperand &use : value.getUses()) {
    if (isa<tensor::DimOp>(use.getOwner()))
      continue;
    if (use.getOwner() != consumer)
      return false;
    ++count;
  }
  return count == 1;
}

bool roundToHalf(mlir::hip::CastOp op, Value ctx) {
  if (!op || op.getCtx() != ctx || op.getTo() != 10)
    return false;
  auto input = plain(op.getInput(), false),
       output = plain(op.getOutput(), true);
  return input && output && input.getShape() == output.getShape() &&
         op->getResult(0).getType() == output;
}

void redirectDims(Value from, Value to, PatternRewriter &rewriter) {
  rewriter.replaceUsesWithIf(from, to, [](OpOperand &use) {
    return isa<tensor::DimOp>(use.getOwner());
  });
}

struct FuseRoundedScaledAdd : OpRewritePattern<mlir::hip::AddOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::hip::AddOp add,
                                PatternRewriter &rewriter) const override {
    auto lhs = add.getLhs().getDefiningOp<mlir::hip::CastOp>();
    auto rhs = add.getRhs().getDefiningOp<mlir::hip::CastOp>();
    auto output = plain(add.getOutput(), true);
    if (!roundToHalf(lhs, add.getCtx()) || !roundToHalf(rhs, add.getCtx()) ||
        lhs == rhs || !output || add->getResult(0).getType() != output ||
        !onlyDataUser(lhs->getResult(0), add) ||
        !onlyDataUser(rhs->getResult(0), add))
      return failure();

    auto base = lhs, productCast = rhs;
    tensor::CollapseShapeOp flatten;
    auto findProduct = [&](mlir::hip::CastOp op) {
      Value value = op.getInput();
      flatten = value.getDefiningOp<tensor::CollapseShapeOp>();
      if (flatten)
        value = flatten.getSrc();
      return value.getDefiningOp<mlir::hip::MulOp>();
    };
    auto mul = findProduct(productCast);
    bool productOnLhs = false;
    if (!mul) {
      base = rhs;
      productCast = lhs;
      mul = findProduct(productCast);
      productOnLhs = true;
    }
    if (!mul || mul.getCtx() != add.getCtx() || !plain(mul.getLhs(), false) ||
        !plain(mul.getRhs(), false) || !plain(mul.getOutput(), false) ||
        mul->getResult(0).getType() != mul.getOutput().getType() ||
        !onlyDataUser(mul->getResult(0), flatten ? flatten.getOperation()
                                                 : productCast.getOperation()))
      return failure();
    SmallVector<ReassociationIndices> groups{{0}, {1}, {2, 3}};
    int64_t heads = 0, width = 0;
    if (flatten) {
      auto expanded = plain(flatten.getSrc(), false);
      auto flat = plain(flatten.getResult(), false);
      auto baseType = plain(base.getInput(), false);
      if (!expanded || expanded.getRank() != 4 || !flat ||
          flat.getRank() != 3 || baseType.getRank() != 3 ||
          output.getRank() != 3 ||
          flatten.getReassociationIndices() != groups ||
          !onlyDataUser(flatten.getResult(), productCast))
        return failure();
      heads = expanded.getDimSize(2);
      width = expanded.getDimSize(3);
      int64_t hidden = flat.getDimSize(2);
      if (heads <= 0 || width <= 0 || hidden <= 0 || hidden % width ||
          hidden / width != heads || baseType.getDimSize(2) != hidden ||
          output.getDimSize(2) != hidden)
        return failure();
    }

    // All guards precede IR changes. Keep shape queries on the old
    // destinations.
    redirectDims(lhs->getResult(0), lhs.getOutput(), rewriter);
    redirectDims(rhs->getResult(0), rhs.getOutput(), rewriter);
    redirectDims(mul->getResult(0), mul.getOutput(), rewriter);
    if (flatten) {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPoint(flatten);
      auto shapeOnly = tensor::CollapseShapeOp::create(
          rewriter, flatten.getLoc(), flatten.getResult().getType(),
          mul.getOutput(), groups);
      redirectDims(flatten.getResult(), shapeOnly, rewriter);
    }
    auto expand = [&](Value value) -> Value {
      if (!flatten)
        return value;
      auto type = cast<RankedTensorType>(value.getType());
      auto shape = tensor::getMixedSizes(rewriter, add.getLoc(), value);
      shape.pop_back();
      shape.push_back(rewriter.getIndexAttr(heads));
      shape.push_back(rewriter.getIndexAttr(width));
      auto expandedType = RankedTensorType::get(
          {type.getDimSize(0), type.getDimSize(1), heads, width},
          type.getElementType());
      return tensor::ExpandShapeOp::create(rewriter, add.getLoc(), expandedType,
                                           value, groups, shape);
    };
    Value expandedBase = expand(base.getInput());
    Value destination = expand(add.getOutput());
    auto fused = mlir::hip::RoundedScaledAddOp::create(
        rewriter, add.getLoc(), TypeRange{destination.getType()}, add.getCtx(),
        expandedBase, mul.getLhs(), mul.getRhs(), destination, productOnLhs);
    Value result = fused->getResult(0);
    if (flatten)
      result = tensor::CollapseShapeOp::create(rewriter, add.getLoc(), output,
                                               result, groups);
    rewriter.replaceOp(add, result);
    rewriter.eraseOp(lhs);
    rewriter.eraseOp(rhs);
    if (flatten)
      rewriter.eraseOp(flatten);
    rewriter.eraseOp(mul);
    return success();
  }
};
} // namespace
void populateRoundedScaledAddPattern(mlir::RewritePatternSet &patterns) {
  patterns.add<FuseRoundedScaledAdd>(patterns.getContext());
}
} // namespace hip::fusion_transform
