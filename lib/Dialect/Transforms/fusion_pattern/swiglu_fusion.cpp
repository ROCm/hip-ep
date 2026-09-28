/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- swiglu_fusion.cpp - hip.sigmoid + hip.mul chain to hip.swiglu ------===//
//
// Any frontend that lowers a gated MLP activation to the primitive HIP ops
// gets the fused kernel. The match lives here, not in an ONNX conversion,
// so a later frontend does not repeat it.
//
// Both multiplies commute. The outer product's destination is reused, so
// the fused op keeps the shape the chain already computed.
//
//   Before:
//     %s = hip.sigmoid(%ctx) ins(%gate : t) outs(%e0 : t) : t
//     %a = hip.mul(%ctx) ins(%gate, %s : t, t) outs(%e1 : t) -> t
//     %y = hip.mul(%ctx) ins(%a, %up : t, t) outs(%e2 : t) -> t
//   After:
//     %y = hip.swiglu(%ctx) ins(%gate, %up : t, t) outs(%e2 : t) : t
//
//===----------------------------------------------------------------------===//

#include "swiglu_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"

namespace hip {
namespace fusion_transform {
namespace {

struct SwigluFusion : public mlir::OpRewritePattern<mlir::hip::MulOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::MulOp outer,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::hip::MulOp silu;
    mlir::Value up;
    if (auto candidate = outer.getLhs().getDefiningOp<mlir::hip::MulOp>()) {
      silu = candidate;
      up = outer.getRhs();
    } else if (auto candidate =
                   outer.getRhs().getDefiningOp<mlir::hip::MulOp>()) {
      silu = candidate;
      up = outer.getLhs();
    } else {
      return rewriter.notifyMatchFailure(outer, "no inner hip.mul");
    }

    mlir::hip::SigmoidOp sigmoid;
    mlir::Value gate;
    auto matchSiLU = [&](mlir::Value sigmoidResult, mlir::Value other) {
      auto candidate = sigmoidResult.getDefiningOp<mlir::hip::SigmoidOp>();
      if (!candidate || other != candidate.getX())
        return false;
      sigmoid = candidate;
      gate = candidate.getX();
      return true;
    };
    if (!matchSiLU(silu.getLhs(), silu.getRhs()) &&
        !matchSiLU(silu.getRhs(), silu.getLhs()))
      return rewriter.notifyMatchFailure(
          outer, "inner mul is not gate * sigmoid(gate)");

    if (!sigmoid.getCtx() || sigmoid.getCtx() != outer.getCtx() ||
        silu.getCtx() != outer.getCtx())
      return rewriter.notifyMatchFailure(outer,
                                         "SwiGLU ops must share a context");

    if (!sigmoid->getResult(0).hasOneUse() || !silu->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(
          outer, "SwiGLU intermediates must have one use");

    auto asTensor = [](mlir::Value value) {
      return mlir::dyn_cast<mlir::RankedTensorType>(value.getType());
    };
    auto gateType = asTensor(gate);
    auto upType = asTensor(up);
    auto sigmoidType = asTensor(sigmoid->getResult(0));
    auto siluType = asTensor(silu->getResult(0));
    auto resultType = asTensor(outer->getResult(0));
    if (!gateType || gateType != upType || gateType != sigmoidType ||
        gateType != siluType || gateType != resultType)
      return rewriter.notifyMatchFailure(
          outer, "SwiGLU does not support broadcasting");

    mlir::Type elemType = gateType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16() &&
        !elemType.isF64())
      return rewriter.notifyMatchFailure(
          outer, "SwiGLU supports only f16, f32, bf16, and f64");

    mlir::Value siluInit = silu.getOutput();
    mlir::Value sigmoidInit = sigmoid.getY();
    mlir::hip::SwigluOp swiglu = mlir::hip::SwigluOp::create(
        rewriter, outer.getLoc(), resultType, outer.getCtx(), gate, up,
        outer.getOutput());
    rewriter.replaceOp(outer, swiglu.getResult(0));
    rewriter.eraseOp(silu);
    rewriter.eraseOp(sigmoid);
    auto eraseIfDead = [&](mlir::Value value) {
      mlir::Operation *def = value.getDefiningOp();
      if (def && def->use_empty())
        rewriter.eraseOp(def);
    };
    eraseIfDead(siluInit);
    eraseIfDead(sigmoidInit);
    return mlir::success();
  }
};

} // namespace

void populateSwigluFusionPattern(mlir::RewritePatternSet &patterns,
                                 mlir::PatternBenefit benefit) {
  patterns.add<SwigluFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
