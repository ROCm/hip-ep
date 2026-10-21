/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "swiglu_f32_fusion.hpp"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

namespace hip::fusion_transform {
namespace {
using namespace mlir;

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

bool plainTensor(Value value, unsigned width) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getRank() > 4 || type.getEncoding())
    return false;
  return width == 16 ? type.getElementType().isF16()
                     : type.getElementType().isF32();
}

bool shapePreservingCast(mlir::hip::CastOp op, unsigned from, unsigned to) {
  // Cast lowering selects arithmetic from the element types, not `to`.
  return op && plainTensor(op.getInput(), from) &&
         plainTensor(op.getOutput(), to) &&
         cast<RankedTensorType>(op.getInput().getType()).getShape() ==
             cast<RankedTensorType>(op.getOutput().getType()).getShape();
}

struct SwigluF32Fusion : OpRewritePattern<mlir::hip::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::hip::CastOp final,
                                PatternRewriter &rewriter) const override {
    if (!shapePreservingCast(final, 32, 16))
      return failure();
    auto mul = final.getInput().getDefiningOp<mlir::hip::MulOp>();
    if (!mul || !onlyDataUser(mul->getResult(0), final))
      return failure();
    auto swish = mul.getRhs().getDefiningOp<mlir::hip::SwishOp>();
    Value upValue = mul.getLhs();
    if (!swish) {
      swish = mul.getLhs().getDefiningOp<mlir::hip::SwishOp>();
      upValue = mul.getRhs();
    }
    if (!swish || swish.getAlpha().convertToDouble() != 1.0 ||
        !onlyDataUser(swish->getResult(0), mul))
      return failure();
    auto gate = swish.getInput().getDefiningOp<mlir::hip::CastOp>();
    auto up = upValue.getDefiningOp<mlir::hip::CastOp>();
    if (!shapePreservingCast(gate, 16, 32) ||
        !shapePreservingCast(up, 16, 32) || gate == up ||
        !onlyDataUser(gate->getResult(0), swish) ||
        !onlyDataUser(up->getResult(0), mul) ||
        !plainTensor(swish.getOutput(), 32) ||
        swish.getInput().getType() != swish.getOutput().getType())
      return failure();
    Value ctx = final.getCtx();
    if (gate.getCtx() != ctx || up.getCtx() != ctx || swish.getCtx() != ctx ||
        mul.getCtx() != ctx)
      return failure();

    // Each replacement precedes the old producer. Shape queries cannot
    // depend on the fused result because they can size its destination.
    auto retargetDims = [&](Value from, Value to) {
      rewriter.replaceUsesWithIf(from, to, [](OpOperand &use) {
        return isa<tensor::DimOp>(use.getOwner());
      });
    };
    retargetDims(gate->getResult(0), gate.getInput());
    retargetDims(up->getResult(0), up.getInput());
    retargetDims(swish->getResult(0), gate.getInput());
    retargetDims(mul->getResult(0), mul.getOutput());
    auto fused = mlir::hip::SwigluF32Op::create(
        rewriter, final.getLoc(), final->getResultTypes(), ctx, gate.getInput(),
        up.getInput(), final.getOutput());
    rewriter.replaceOp(final, fused->getResults());
    rewriter.eraseOp(mul);
    rewriter.eraseOp(swish);
    rewriter.eraseOp(up);
    rewriter.eraseOp(gate);
    return success();
  }
};
} // namespace
void populateSwigluF32Pattern(mlir::RewritePatternSet &patterns) {
  patterns.add<SwigluF32Fusion>(patterns.getContext());
}
} // namespace hip::fusion_transform
