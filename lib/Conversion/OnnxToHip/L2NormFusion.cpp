/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- L2NormFusion.cpp - Fold the q/k L2-normalization chain into hip ---===//
//
// Collapses the exporter-emitted q/k L2-normalization decomposition in the
// Qwen3.5 linear-attention layers:
//
//   sq   = onnx.Mul(x, x)                       // square (NOT Pow)
//   ss   = onnx.ReduceSum(sq, axes=[-1], keepdims=1)
//   den  = onnx.Add(ss, eps)
//   r    = onnx.Reciprocal(onnx.Sqrt(den))
//   out  = onnx.Mul(x, r)
//
// into a single `hip.l2_norm`. Rooted on the (rare) onnx.Reciprocal. Lives in
// the pre-lowering loop (next to FastGeluFusion) so it matches the primitive
// onnx.* ops before convertComputeOps and while the eps Constant is inline.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cstdlib>
#include <optional>
#include <string>

#define DEBUG_TYPE "l2-norm-fusion"

STATISTIC(NumL2NormFused, "q/k L2-normalization chains folded into hip.l2_norm");

namespace mlir {
namespace hip {

namespace {

static std::optional<double> getScalarFloatConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "onnx.Constant")
    return std::nullopt;
  auto denseAttr =
      mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
  if (!denseAttr || denseAttr.getNumElements() != 1)
    return std::nullopt;
  mlir::Type et = denseAttr.getElementType();
  if (et.isF32())
    return static_cast<double>(*denseAttr.getValues<float>().begin());
  if (et.isF64())
    return *denseAttr.getValues<double>().begin();
  if (et.isF16() || et.isBF16())
    return (*denseAttr.getValues<llvm::APFloat>().begin()).convertToDouble();
  // Pow exponent sometimes ships as an integer constant (e.g. 2 : i64).
  if (et.isIntOrIndex())
    return static_cast<double>(
        (*denseAttr.getValues<llvm::APInt>().begin()).getSExtValue());
  return std::nullopt;
}

static std::optional<int64_t> getSingleReduceAxis(mlir::Operation *reduceOp) {
  if (reduceOp->getNumOperands() == 2) {
    auto def = reduceOp->getOperand(1).getDefiningOp();
    if (def && def->getName().getStringRef() == "onnx.Constant") {
      auto dense = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
          def->getAttr("value"));
      if (dense && dense.getElementType().isInteger(64) &&
          dense.getNumElements() == 1)
        return (*dense.getValues<llvm::APInt>().begin()).getSExtValue();
    }
    return std::nullopt;
  }
  if (auto axesAttr = reduceOp->getAttrOfType<mlir::ArrayAttr>("axes")) {
    if (axesAttr.size() == 1)
      if (auto ia = mlir::dyn_cast<mlir::IntegerAttr>(axesAttr[0]))
        return ia.getValue().getSExtValue();
  }
  return std::nullopt;
}

struct L2NormChainToHip : public mlir::RewritePattern {
  L2NormChainToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Reciprocal", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *recipOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (recipOp->getNumOperands() != 1 || recipOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(recipOp, "recip.arity");

    mlir::Operation *sqrtOp = recipOp->getOperand(0).getDefiningOp();
    if (!sqrtOp || sqrtOp->getName().getStringRef() != "onnx.Sqrt" ||
        sqrtOp->getNumOperands() != 1)
      return rewriter.notifyMatchFailure(recipOp, "sqrt.in");

    mlir::Operation *addOp = sqrtOp->getOperand(0).getDefiningOp();
    if (!addOp || addOp->getName().getStringRef() != "onnx.Add" ||
        addOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(recipOp, "add.in");

    mlir::Operation *reduceOp = nullptr;
    std::optional<double> eps;
    for (int i = 0; i < 2; ++i) {
      mlir::Value cand = addOp->getOperand(i);
      mlir::Value other = addOp->getOperand(1 - i);
      mlir::Operation *def = cand.getDefiningOp();
      if (def && def->getName().getStringRef() == "onnx.ReduceSum") {
        if (auto e = getScalarFloatConstant(other)) {
          reduceOp = def;
          eps = e;
          break;
        }
      }
    }
    if (!reduceOp || !eps)
      return rewriter.notifyMatchFailure(recipOp, "add.reducesum_eps");

    std::optional<int64_t> axis = getSingleReduceAxis(reduceOp);
    if (!axis)
      return rewriter.notifyMatchFailure(recipOp, "reduce.axis_unknown");
    auto sqType = mlir::dyn_cast<mlir::RankedTensorType>(
        reduceOp->getOperand(0).getType());
    if (!sqType)
      return rewriter.notifyMatchFailure(recipOp, "reduce.in_not_ranked");
    int64_t rank = sqType.getRank();
    int64_t normAxis = *axis < 0 ? *axis + rank : *axis;
    if (normAxis != rank - 1)
      return rewriter.notifyMatchFailure(recipOp, "reduce.not_last_axis");
    if (auto kd = reduceOp->getAttrOfType<mlir::IntegerAttr>("keepdims"))
      if (kd.getValue().getSExtValue() != 1)
        return rewriter.notifyMatchFailure(recipOp, "reduce.not_keepdims");

    // Square is either Mul(x, x) (rtn export) or Pow(x, 2) (fp16-ve export).
    mlir::Operation *sqOp = reduceOp->getOperand(0).getDefiningOp();
    if (!sqOp || sqOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(recipOp, "square.arity");
    llvm::StringRef sqName = sqOp->getName().getStringRef();
    mlir::Value x;
    if (sqName == "onnx.Mul") {
      if (sqOp->getOperand(0) != sqOp->getOperand(1))
        return rewriter.notifyMatchFailure(recipOp, "square.mul_operands_differ");
      x = sqOp->getOperand(0);
    } else if (sqName == "onnx.Pow") {
      auto exp = getScalarFloatConstant(sqOp->getOperand(1));
      if (!exp || *exp != 2.0)
        return rewriter.notifyMatchFailure(recipOp, "square.pow_not_2");
      x = sqOp->getOperand(0);
    } else {
      return rewriter.notifyMatchFailure(recipOp, "square.not_mul_or_pow");
    }

    mlir::Value recipRes = recipOp->getResult(0);
    if (!recipRes.hasOneUse())
      return rewriter.notifyMatchFailure(recipOp, "recip.not_one_use");
    mlir::Operation *finalMul = *recipRes.getUsers().begin();
    if (finalMul->getName().getStringRef() != "onnx.Mul" ||
        finalMul->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(recipOp, "final.not_mul");
    mlir::Value mulOther = (finalMul->getOperand(0) == recipRes)
                               ? finalMul->getOperand(1)
                               : finalMul->getOperand(0);
    if (mulOther != x)
      return rewriter.notifyMatchFailure(recipOp, "final.operand_neq_x");

    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(recipOp, "x.not_ranked");
    auto outType = mlir::dyn_cast<mlir::RankedTensorType>(
        finalMul->getResult(0).getType());
    if (!outType || outType.getShape() != xType.getShape() ||
        outType.getElementType() != xType.getElementType())
      return rewriter.notifyMatchFailure(recipOp, "out_type_mismatch");

    auto ctxOrFailure = getContextArg(finalMul, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(recipOp, "no_context_arg");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = finalMul->getLoc();
    rewriter.setInsertionPoint(finalMul);
    mlir::Value init = createEmptyTensor(rewriter, loc, outType, x);
    auto axisAttr = rewriter.getI64IntegerAttr(-1);
    auto epsAttr = rewriter.getF32FloatAttr(static_cast<float>(*eps));

    auto hipOp = mlir::hip::L2NormOp::create(rewriter, loc, outType, context, x,
                                             init, axisAttr, epsAttr);
    rewriter.replaceOp(finalMul, hipOp->getResult(0));

    auto eraseIfDead = [&rewriter](mlir::Operation *op) {
      if (op && op->use_empty())
        rewriter.eraseOp(op);
    };
    eraseIfDead(recipOp);
    eraseIfDead(sqrtOp);
    eraseIfDead(addOp);
    eraseIfDead(reduceOp);
    eraseIfDead(sqOp);

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] fused L2-norm at " << loc
                            << " (eps=" << *eps << ")\n");
    ++NumL2NormFused;
    return mlir::success();
  }
};

/// Direct match of `onnx.LpNormalization` (p=2, last axis) -> hip.l2_norm.
/// The fp16-ve Qwen3.5 export emits LpNormalization (48x) for q/k norm instead
/// of the decomposed chain. LpNormalization has no epsilon (output = x /
/// L2(x)), so eps=0 -- matching the LpNormalizationConversion decomposition's
/// plain-division behaviour. Benefit 3 so this fires before
/// LpNormalizationDecompose (benefit 2).
struct LpNormalizationToHip : public mlir::RewritePattern {
  LpNormalizationToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.LpNormalization", /*benefit=*/3, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "lpnorm.arity");

    int64_t p = 2;
    if (auto pAttr = op->getAttrOfType<mlir::IntegerAttr>("p"))
      p = pAttr.getValue().getSExtValue();
    if (p != 2)
      return rewriter.notifyMatchFailure(op, "lpnorm.p_not_2");

    mlir::Value x = op->getOperand(0);
    auto xType = mlir::dyn_cast<mlir::RankedTensorType>(x.getType());
    if (!xType)
      return rewriter.notifyMatchFailure(op, "lpnorm.not_ranked");
    int64_t rank = xType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "lpnorm.scalar");

    int64_t axis = -1;
    if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = axisAttr.getValue().getSExtValue();
    int64_t normAxis = axis < 0 ? axis + rank : axis;
    if (normAxis != rank - 1)
      return rewriter.notifyMatchFailure(op, "lpnorm.not_last_axis");

    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!outType || outType.getShape() != xType.getShape() ||
        outType.getElementType() != xType.getElementType())
      return rewriter.notifyMatchFailure(op, "lpnorm.out_type_mismatch");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "lpnorm.no_context_arg");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value init = createEmptyTensor(rewriter, loc, outType, x);
    auto axisAttr = rewriter.getI64IntegerAttr(-1);
    auto epsAttr = rewriter.getF32FloatAttr(0.0f);
    auto hipOp = mlir::hip::L2NormOp::create(rewriter, loc, outType, context, x,
                                             init, axisAttr, epsAttr);
    rewriter.replaceOp(op, hipOp->getResult(0));
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] fused LpNormalization at "
                            << loc << "\n");
    ++NumL2NormFused;
    return mlir::success();
  }
};

} // namespace

void populateL2NormFusionPatterns(mlir::RewritePatternSet &patterns,
                                  mlir::MLIRContext *ctx) {
  if (const char *env = std::getenv("HIPDNN_EP_L2NORM_FUSE"))
    if (std::string(env) == "0")
      return;
  patterns.add<L2NormChainToHip, LpNormalizationToHip>(ctx);
}

} // namespace hip
} // namespace mlir
