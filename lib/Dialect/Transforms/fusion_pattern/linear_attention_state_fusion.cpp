// Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
// Licensed under the MIT License.
#include "linear_attention_state_fusion.hpp"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "llvm/ADT/STLExtras.h"

namespace hip::fusion_transform {
namespace {
using namespace mlir;

RankedTensorType plainTensor(Value value, unsigned bits, int rank) {
  if (!value)
    return {};
  auto type = dyn_cast<RankedTensorType>(value.getType());
  if (!type || type.getEncoding() || type.getRank() != rank ||
      !type.getElementType().isFloat() ||
      type.getElementType().getIntOrFloatBitWidth() != bits)
    return {};
  return type;
}

bool onlyDataUser(Value value, Operation *dataUser) {
  return llvm::all_of(value.getUsers(), [&](Operation *user) {
    return user == dataUser || isa<tensor::DimOp>(user);
  });
}

void redirectDims(Value from, Value to, PatternRewriter &rewriter) {
  for (Operation *user : llvm::make_early_inc_range(from.getUsers()))
    if (auto dim = dyn_cast<tensor::DimOp>(user))
      rewriter.modifyOpInPlace(dim, [&] { dim.getSourceMutable().assign(to); });
}

struct FuseStateCasts : OpRewritePattern<mlir::hip::CastOp> {
  using OpRewritePattern::OpRewritePattern;
  LogicalResult matchAndRewrite(mlir::hip::CastOp castOut,
                                PatternRewriter &rewriter) const override {
    auto la = castOut.getInput().getDefiningOp<mlir::hip::LinearAttentionOp>();
    if (!la || la->getNumResults() != 2 ||
        castOut.getInput() != la->getResult(1) ||
        !onlyDataUser(la->getResult(1), castOut) || !la.getPastState())
      return failure();
    auto castIn = la.getPastState().getDefiningOp<mlir::hip::CastOp>();
    if (!castIn || !onlyDataUser(la.getPastState(), la) ||
        castIn.getCtx() != la.getCtx() || castOut.getCtx() != la.getCtx() ||
        castIn.getTo() != 1 || castOut.getTo() != 10)
      return failure();

    auto past = plainTensor(castIn.getInput(), 16, 4);
    auto present = plainTensor(castOut->getResult(0), 16, 4);
    auto state = plainTensor(la.getPastState(), 32, 4);
    if (!past || !present || !state || past != present ||
        past.getElementType() != rewriter.getF16Type() ||
        state.getShape() != past.getShape() ||
        la.getPresentState().getType() != state ||
        la->getResult(1).getType() != state)
      return failure();
    for (Value v : {la.getQuery(), la.getKey(), la.getValue(), la.getOutput(),
                    Value(la->getResult(0)), la.getDecay(), la.getBeta()})
      if (v && !plainTensor(v, 32, 3))
        return failure();

    // Both removed casts have one data user. Shape queries do not read data.
    // Build the new destination here so it dominates the new operation.
    rewriter.setInsertionPoint(la);
    Value empty = tensor::EmptyOp::create(
        rewriter, la.getLoc(),
        tensor::getMixedSizes(rewriter, la.getLoc(), la.getPresentState()),
        rewriter.getF16Type());
    auto fused = cast<mlir::hip::LinearAttentionOp>(rewriter.clone(*la));
    fused.getPastStateMutable().assign(castIn.getInput());
    fused.getPresentStateMutable().assign(empty);
    fused->getResult(1).setType(present);
    // State rounds once per invocation, not once per prefill token.
    fused->setAttr("hip.fp16_state_boundary", rewriter.getUnitAttr());
    redirectDims(la.getPastState(), castIn.getInput(), rewriter);
    redirectDims(la->getResult(1), fused->getResult(1), rewriter);
    rewriter.replaceAllUsesWith(la->getResult(0), fused->getResult(0));
    rewriter.replaceOp(castOut, fused->getResult(1));
    rewriter.eraseOp(la);
    return success();
  }
};
} // namespace
void populateLinearAttentionStatePattern(mlir::RewritePatternSet &patterns) {
  patterns.add<FuseStateCasts>(patterns.getContext());
}
} // namespace hip::fusion_transform
