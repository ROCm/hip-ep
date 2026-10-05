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
// broadcast. Each dynamic axis must trace to the same extent. A block
// argument is its own extent. hip.matmul and hip.qmatmul contribute batch
// and M from the activation (N from the weight); hip.matmul_nbits
// contributes every axis but the last from the activation and the last
// from its N attribute. That is the same source choice as
// reifyMatmulLikeShape and MatMulNBitsOp::reifyResultShapes.
//
//===----------------------------------------------------------------------===//

#include "swiglu_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/Sequence.h"

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

// One dynamic axis after projection producers have been followed.
struct Extent {
  bool isStatic = false;
  int64_t staticSize = 0;
  mlir::Value value;
  int64_t dim = 0;

  bool operator==(const Extent &other) const {
    if (isStatic || other.isStatic)
      return isStatic && other.isStatic && staticSize == other.staticSize;
    return value == other.value && dim == other.dim;
  }
};

// Follow one matmul-like result axis to the operand axis that supplies it.
// A static result is returned directly. std::nullopt means this op does not
// reveal the axis, so the caller keeps the op result as the extent.
std::optional<Extent> stepProjection(mlir::Operation *op, int64_t outDim) {
  if (op->getNumResults() == 0)
    return std::nullopt;
  auto mapMatmul = [&](mlir::Value a, mlir::Value b, int64_t transA,
                       int64_t transB,
                       int64_t resultRank) -> std::optional<Extent> {
    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
    auto bType = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
    if (!aType || !bType || aType.getRank() < 2 || bType.getRank() < 2)
      return std::nullopt;
    int64_t aRank = aType.getRank();
    int64_t bRank = bType.getRank();
    int64_t batchRank = std::max(aRank - 2, bRank - 2);
    int64_t outRank = batchRank + 2;
    if (outDim < 0 || outDim >= outRank || outRank != resultRank)
      return std::nullopt;

    if (outDim + 2 == outRank) {
      int64_t aDim = transA ? aRank - 1 : aRank - 2;
      return Extent{false, 0, a, aDim};
    }
    if (outDim + 1 == outRank) {
      int64_t bDim = transB ? bRank - 2 : bRank - 1;
      return Extent{false, 0, b, bDim};
    }

    int64_t aPad = batchRank - (aRank - 2);
    int64_t bPad = batchRank - (bRank - 2);
    int64_t aExtent =
        outDim < aPad
            ? 1
            : aType.getDimSize(static_cast<unsigned>(outDim - aPad));
    int64_t bExtent =
        outDim < bPad
            ? 1
            : bType.getDimSize(static_cast<unsigned>(outDim - bPad));
    bool aCanonical = outDim >= aPad && aExtent != 1;
    bool bCanonical = outDim >= bPad && bExtent != 1;
    bool pickA = aCanonical || (!bCanonical && outDim >= aPad);
    if (pickA && outDim >= aPad)
      return Extent{false, 0, a, outDim - aPad};
    if (!pickA && outDim >= bPad)
      return Extent{false, 0, b, outDim - bPad};
    return std::nullopt;
  };

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return std::nullopt;
  int64_t resultRank = resultType.getRank();

  if (auto matmul = mlir::dyn_cast<mlir::hip::MatmulOp>(op))
    return mapMatmul(matmul.getA(), matmul.getB(), matmul.getTransA(),
                     matmul.getTransB(), resultRank);
  if (auto qmatmul = mlir::dyn_cast<mlir::hip::QMatMulOp>(op))
    return mapMatmul(qmatmul.getA(), qmatmul.getB(), qmatmul.getTransA(),
                     qmatmul.getTransB(), resultRank);
  if (auto nbits = mlir::dyn_cast<mlir::hip::MatMulNBitsOp>(op)) {
    auto aType = mlir::dyn_cast<mlir::RankedTensorType>(nbits.getA().getType());
    if (!aType || aType.getRank() != resultRank || outDim < 0 ||
        outDim >= resultRank)
      return std::nullopt;
    // The trailing axis is the N attribute, not A's contraction axis.
    if (outDim + 1 == resultRank)
      return Extent{true, static_cast<int64_t>(nbits.getN()), {}, 0};
    return Extent{false, 0, nbits.getA(), outDim};
  }
  return std::nullopt;
}

// The walk follows tensor.cast and one projection. hip-infer-shapes inserts
// a cast on the projection result and may leave one on its activation, which
// is three steps; the fourth covers a second cast on either side. Anything
// longer is not this pattern, so the walk declines instead of scanning it.
constexpr int kMaxExtentHops = 4;

std::optional<Extent> traceExtent(mlir::Value value, int64_t dim) {
  mlir::Value current = value;
  int64_t axis = dim;
  for (int hop = 0; hop < kMaxExtentHops; ++hop) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(current.getType());
    if (!type || axis < 0 || axis >= type.getRank())
      return std::nullopt;
    if (!type.isDynamicDim(static_cast<unsigned>(axis)))
      return Extent{true, type.getDimSize(static_cast<unsigned>(axis)), {}, 0};

    mlir::Operation *def = current.getDefiningOp();
    if (!def)
      return Extent{false, 0, current, axis};
    if (auto castOp = mlir::dyn_cast<mlir::tensor::CastOp>(def)) {
      current = castOp.getSource();
      continue;
    }
    if (std::optional<Extent> stepped = stepProjection(def, axis)) {
      if (stepped->isStatic)
        return *stepped;
      current = stepped->value;
      axis = stepped->dim;
      continue;
    }
    return Extent{false, 0, current, axis};
  }
  return std::nullopt;
}

// True when every dynamic axis of both tensors is the same runtime extent.
// Static axes are already settled by type equality.
bool sameRuntimeShape(mlir::Value lhs, mlir::Value rhs) {
  auto lhsType = mlir::dyn_cast<mlir::RankedTensorType>(lhs.getType());
  auto rhsType = mlir::dyn_cast<mlir::RankedTensorType>(rhs.getType());
  if (!lhsType || lhsType != rhsType)
    return false;
  for (int64_t dim : llvm::seq<int64_t>(0, lhsType.getRank())) {
    if (!lhsType.isDynamicDim(static_cast<unsigned>(dim)))
      continue;
    std::optional<Extent> lhsExtent = traceExtent(lhs, dim);
    std::optional<Extent> rhsExtent = traceExtent(rhs, dim);
    if (!lhsExtent || !rhsExtent || !(*lhsExtent == *rhsExtent))
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

    mlir::Value siluInit = silu.getOutput();
    mlir::Value sigmoidInit = sigmoid.getY();
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
    auto eraseIfDead = [&](mlir::Value value) {
      mlir::Operation *def = value.getDefiningOp();
      if (def && mlir::isOpTriviallyDead(def))
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
