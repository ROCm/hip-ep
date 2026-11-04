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
//     %d0 = tensor.dim %a, %c0 : t
//     %y = hip.mul(%ctx) ins(%a, %up : t, t) outs(%e2 : t) -> t
//   After:
//     %d0 = tensor.dim %gate, %c0 : t
//     %y = hip.swiglu(%ctx) ins(%gate, %up : t, t) outs(%e2 : t) : t
//
// A dynamic export sizes the outer init with tensor.dim of the inner
// product. That query does not read activation values, and the pattern
// already requires the inner product's type to equal the gate's, so the
// query is retargeted to the gate and does not keep the intermediate alive.
//
// Identical types do not prove a dynamic axis has one runtime extent:
// tensor<?x?x14336xf16> matches tensor<?x?x14336xf16> when one side is
// [4, S, 14336] and the other is [1, S, 14336], and hip.mul would
// broadcast. Each dynamic axis must trace to the same extent (see
// extent_trace.hpp). A block argument is its own extent.
//
//===----------------------------------------------------------------------===//

#include "swiglu_fusion.hpp"

#include "extent_trace.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace hip {
namespace fusion_transform {
namespace {

// True when every use is either `consumer` or a tensor.dim. tensor.dim only
// reads the type, which this pattern requires to match the gate.
bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

void retargetShapeQueries(mlir::PatternRewriter &rewriter, mlir::Value from,
                          mlir::Value to) {
  rewriter.replaceUsesWithIf(from, to, [](mlir::OpOperand &operand) {
    return mlir::isa<mlir::tensor::DimOp>(operand.getOwner());
  });
}

// True when each dynamic axis has a common, proven runtime extent. Static
// axes are settled by type equality; equal type spelling alone proves no
// equality between independent dynamic arguments.
bool sameRuntimeShape(mlir::Value lhs, mlir::Value rhs) {
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhsType || lhsType != rhsType)
    return false;
  for (int64_t dim : llvm::seq<int64_t>(0, lhsType.getRank())) {
    if (!lhsType.isDynamicDim(static_cast<unsigned>(dim)))
      continue;
    if (!sameRuntimeExtent(lhs, dim, rhs, dim))
      return false;
  }
  return true;
}

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

    if (!hasSingleValueUse(sigmoid->getResult(0), silu) ||
        !hasSingleValueUse(silu->getResult(0), outer))
      return rewriter.notifyMatchFailure(
          outer, "SwiGLU intermediates must have one value use");

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

    if (!sameRuntimeShape(gate, up))
      return rewriter.notifyMatchFailure(
          outer, "dynamic dimensions are not known to match");

    mlir::Operation *siluInitDef = silu.getOutput().getDefiningOp();
    mlir::Operation *sigmoidInitDef = sigmoid.getY().getDefiningOp();
    mlir::Value siluResult = silu->getResult(0);
    mlir::Value sigmoidResult = sigmoid->getResult(0);
    retargetShapeQueries(rewriter, siluResult, gate);
    retargetShapeQueries(rewriter, sigmoidResult, gate);
    mlir::hip::SwigluOp swiglu = mlir::hip::SwigluOp::create(
        rewriter, outer.getLoc(), resultType, outer.getCtx(), gate, up,
        outer.getOutput());
    rewriter.replaceOp(outer, swiglu.getResult(0));
    rewriter.eraseOp(silu);
    rewriter.eraseOp(sigmoid);
    // Drop a destination that exists only to feed an erased op. A producer
    // with side effects, such as a call that records, stays in the graph.
    // Both destinations may come from one producer, which must be erased once.
    auto eraseIfDead = [&](mlir::Operation *def) {
      if (def && mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    };
    eraseIfDead(siluInitDef);
    if (sigmoidInitDef != siluInitDef)
      eraseIfDead(sigmoidInitDef);
    return mlir::success();
  }
};

// Cast-to-f32(gate/up) -> f32 SwiGLU -> cast-to-f16. The f32 operation
// establishes the intermediate rounding contract; the promoted kernel keeps
// that contract while reading and writing only the original half buffers.
struct PromotedSwigluFusion : public mlir::OpRewritePattern<mlir::hip::CastOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::CastOp outputCast,
                  mlir::PatternRewriter &rewriter) const override {
    auto swiglu = outputCast.getInput().getDefiningOp<mlir::hip::SwigluOp>();
    if (!swiglu || swiglu.getPromoteActivation())
      return mlir::failure();
    auto gateCast = swiglu.getGate().getDefiningOp<mlir::hip::CastOp>();
    auto upCast = swiglu.getUp().getDefiningOp<mlir::hip::CastOp>();
    if (!gateCast || !upCast || gateCast == upCast)
      return mlir::failure();
    auto gateType =
        mlir::dyn_cast<mlir::RankedTensorType>(gateCast.getInput().getType());
    auto upType =
        mlir::dyn_cast<mlir::RankedTensorType>(upCast.getInput().getType());
    auto outputType = mlir::dyn_cast<mlir::RankedTensorType>(
        outputCast.getResult(0).getType());
    auto floatType =
        mlir::dyn_cast<mlir::RankedTensorType>(swiglu.getResult(0).getType());
    if (!gateType || !floatType || !gateType.getElementType().isF16() ||
        !floatType.getElementType().isF32() || gateType != upType ||
        gateType != outputType || floatType.getShape() != gateType.getShape() ||
        swiglu.getGate().getType() != floatType ||
        swiglu.getUp().getType() != floatType)
      return mlir::failure();
    if (outputCast.getCtx() != swiglu.getCtx() ||
        gateCast.getCtx() != swiglu.getCtx() ||
        upCast.getCtx() != swiglu.getCtx() ||
        !hasSingleValueUse(swiglu.getResult(0), outputCast) ||
        !hasSingleValueUse(gateCast.getResult(0), swiglu) ||
        !hasSingleValueUse(upCast.getResult(0), swiglu))
      return mlir::failure();

    // SwiGLU itself has a no-broadcast shape contract. The standard fusion
    // above proves it before creating the f32 operation.
    llvm::SmallVector<mlir::Operation *> initDefs{
        swiglu.getOutput().getDefiningOp(),
        gateCast.getOutput().getDefiningOp(),
        upCast.getOutput().getDefiningOp()};
    retargetShapeQueries(rewriter, swiglu.getResult(0), gateCast.getInput());
    retargetShapeQueries(rewriter, gateCast.getResult(0), gateCast.getInput());
    retargetShapeQueries(rewriter, upCast.getResult(0), upCast.getInput());
    auto promoted = mlir::hip::SwigluOp::create(
        rewriter, outputCast.getLoc(), outputType, outputCast.getCtx(),
        gateCast.getInput(), upCast.getInput(), outputCast.getOutput(), true);
    rewriter.replaceOp(outputCast, promoted.getResult(0));
    rewriter.eraseOp(swiglu);
    rewriter.eraseOp(gateCast);
    rewriter.eraseOp(upCast);
    llvm::SmallPtrSet<mlir::Operation *, 4> erased;
    for (mlir::Operation *def : initDefs)
      if (def && erased.insert(def).second && mlir::isOpTriviallyDead(def))
        rewriter.eraseOp(def);
    return mlir::success();
  }
};

} // namespace

void populateSwigluFusionPattern(mlir::RewritePatternSet &patterns,
                                 mlir::PatternBenefit benefit) {
  patterns.add<SwigluFusion, PromotedSwigluFusion>(patterns.getContext(),
                                                   benefit);
}

} // namespace fusion_transform
} // namespace hip
