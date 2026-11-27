/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RmsNormFusion.cpp - Fold RMSNorm primitives into onnx.RMSNormalization
//
// Diffusers / PyTorch RMSNorm exports as an elementwise chain:
//
//   y = (x / sqrt(mean(x^2) + eps)) * scale
//
// with optional fp32 Casts around the math (stash_type = 1). Lowered as written
// that is several bandwidth-bound kernels per site. Fold the private chain into
// one `onnx.RMSNormalization`, which NormConversion already lowers to
// `hip.rms_norm`.
//
// Matched spellings (rooted on the final scale Mul):
//
//   Pow(x, 2) | Mul(x, x) -> ReduceMean(last axis, keepdims=1)
//     -> Add(eps) -> Sqrt -> (Div(1, .) | Reciprocal)
//     -> Mul(x, inv) -> Mul(., scale)
//
// or Div(x, Sqrt(...)) in place of Mul(x, inv). Cast / CastLike around x for
// the square and normalize legs are peeked through so SD3.5's fp16->fp32->fp16
// stash form still matches. LayerNorm's centered variance (Pow(Sub(...))) is
// rejected so this does not steal LayerNorm fusions.
//
// Benefit is higher than ProjectorOps' PowToMul so the chain fuses before
// Pow(x,2) is rewritten to Mul(x,x); Mul(x,x) is still accepted.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "rms-norm-fusion"

STATISTIC(NumRmsNormFused,
          "Number of RMSNorm primitive chains folded to onnx.RMSNormalization");

namespace mlir {
namespace hip {

namespace {

constexpr double kConstantTolerance = 1e-3;

bool isOnnxOp(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

bool isCastLike(mlir::Operation *op) {
  return isOnnxOp(op, "onnx.Cast") || isOnnxOp(op, "onnx.CastLike");
}

mlir::Value peekCast(mlir::Value v, mlir::Operation **castOp = nullptr) {
  mlir::Operation *def = v.getDefiningOp();
  if (isCastLike(def) && def->getNumOperands() >= 1) {
    if (castOp)
      *castOp = def;
    return def->getOperand(0);
  }
  if (castOp)
    *castOp = nullptr;
  return v;
}

std::optional<double> getScalarConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isOnnxOp(def, "onnx.Constant")) {
    auto attr =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
    if (!attr || attr.getNumElements() != 1 ||
        !mlir::isa<mlir::FloatType>(attr.getElementType()))
      return std::nullopt;
    return (*attr.getValues<llvm::APFloat>().begin()).convertToDouble();
  }
  if (isCastLike(def) && def->getNumOperands() >= 1)
    return getScalarConstant(def->getOperand(0));
  return std::nullopt;
}

bool isNear(std::optional<double> v, double expected) {
  return v && std::abs(*v - expected) < kConstantTolerance;
}

bool hasOnlyUser(mlir::Value v, mlir::Operation *user) {
  return v.hasOneUse() && *v.getUsers().begin() == user;
}

/// True when ReduceMean reduces only the last axis with keepdims=1.
bool reduceMeanIsLastAxisKeepdims(mlir::Operation *rm,
                                  mlir::RankedTensorType squareType) {
  if (!isOnnxOp(rm, "onnx.ReduceMean") || !squareType ||
      squareType.getRank() < 1)
    return false;

  auto keepdims = rm->getAttrOfType<mlir::IntegerAttr>("keepdims");
  if (keepdims && keepdims.getInt() == 0)
    return false;

  const int64_t rank = squareType.getRank();
  auto normalizeAxis = [&](int64_t axis) -> int64_t {
    if (axis < 0)
      axis += rank;
    return axis;
  };

  if (auto dense = rm->getAttrOfType<mlir::DenseI64ArrayAttr>("axes")) {
    if (dense.size() != 1)
      return false;
    return normalizeAxis(dense[0]) == rank - 1;
  }
  if (auto axesAttr = rm->getAttrOfType<mlir::DenseIntElementsAttr>("axes")) {
    if (axesAttr.getNumElements() != 1)
      return false;
    return normalizeAxis(
               (*axesAttr.getValues<llvm::APInt>().begin()).getSExtValue()) ==
           rank - 1;
  }
  if (auto arr = rm->getAttrOfType<mlir::ArrayAttr>("axes")) {
    if (arr.size() != 1)
      return false;
    auto ia = mlir::dyn_cast<mlir::IntegerAttr>(arr[0]);
    if (!ia)
      return false;
    return normalizeAxis(ia.getInt()) == rank - 1;
  }
  // Opset-18 axes operand.
  if (rm->getNumOperands() >= 2) {
    mlir::Operation *axesDef = rm->getOperand(1).getDefiningOp();
    if (!isOnnxOp(axesDef, "onnx.Constant"))
      return false;
    auto axes = axesDef->getAttrOfType<mlir::DenseIntElementsAttr>("value");
    if (!axes || axes.getNumElements() != 1)
      return false;
    return normalizeAxis(
               (*axes.getValues<llvm::APInt>().begin()).getSExtValue()) ==
           rank - 1;
  }
  return false;
}

/// Match x^2 as Pow(x, 2) or Mul(x, x). Rejects Pow(Sub(...)) (LayerNorm).
bool matchSquare(mlir::Value square, mlir::Value &xSource,
                 mlir::Operation *&squareOp) {
  squareOp = square.getDefiningOp();
  if (isOnnxOp(squareOp, "onnx.Pow") && squareOp->getNumOperands() == 2) {
    if (!isNear(getScalarConstant(squareOp->getOperand(1)), 2.0))
      return false;
    mlir::Value base = squareOp->getOperand(0);
    if (isOnnxOp(base.getDefiningOp(), "onnx.Sub"))
      return false;
    xSource = peekCast(base);
    if (isOnnxOp(xSource.getDefiningOp(), "onnx.Sub"))
      return false;
    return true;
  }
  if (isOnnxOp(squareOp, "onnx.Mul") && squareOp->getNumOperands() == 2 &&
      squareOp->getOperand(0) == squareOp->getOperand(1)) {
    mlir::Value base = squareOp->getOperand(0);
    if (isOnnxOp(base.getDefiningOp(), "onnx.Sub"))
      return false;
    xSource = peekCast(base);
    if (isOnnxOp(xSource.getDefiningOp(), "onnx.Sub"))
      return false;
    return true;
  }
  return false;
}

/// Match inv = 1/sqrt as Div(1, sqrt), Reciprocal(sqrt), or the
/// BroadcastDivToMulReciprocal form Mul(1, Reciprocal(sqrt)).
/// On success, `invOp` is the op that produces `inv` (feeds the normalize
/// Mul), and `sqrtConsumer` is the op that must be Sqrt's only user.
bool matchInvRms(mlir::Value inv, mlir::Value &sqrtVal, mlir::Operation *&invOp,
                 mlir::Operation *&sqrtConsumer) {
  invOp = inv.getDefiningOp();
  if (isOnnxOp(invOp, "onnx.Reciprocal") && invOp->getNumOperands() == 1) {
    sqrtVal = invOp->getOperand(0);
    sqrtConsumer = invOp;
    return isOnnxOp(sqrtVal.getDefiningOp(), "onnx.Sqrt");
  }
  if (isOnnxOp(invOp, "onnx.Div") && invOp->getNumOperands() == 2 &&
      isNear(getScalarConstant(invOp->getOperand(0)), 1.0)) {
    sqrtVal = invOp->getOperand(1);
    sqrtConsumer = invOp;
    return isOnnxOp(sqrtVal.getDefiningOp(), "onnx.Sqrt");
  }
  // After BroadcastDivToMulReciprocal: Div(1, sqrt) -> Mul(1,
  // Reciprocal(sqrt)).
  if (isOnnxOp(invOp, "onnx.Mul") && invOp->getNumOperands() == 2) {
    for (unsigned oneIdx : {0u, 1u}) {
      if (!isNear(getScalarConstant(invOp->getOperand(oneIdx)), 1.0))
        continue;
      mlir::Value recVal = invOp->getOperand(1u - oneIdx);
      mlir::Operation *recOp = recVal.getDefiningOp();
      if (!isOnnxOp(recOp, "onnx.Reciprocal") || recOp->getNumOperands() != 1)
        continue;
      if (!hasOnlyUser(recOp->getResult(0), invOp))
        continue;
      sqrtVal = recOp->getOperand(0);
      if (!isOnnxOp(sqrtVal.getDefiningOp(), "onnx.Sqrt"))
        continue;
      sqrtConsumer = recOp;
      return true;
    }
  }
  return false;
}

bool isLastDimScale(mlir::Value scale, mlir::RankedTensorType xType) {
  auto scaleType = mlir::dyn_cast<mlir::RankedTensorType>(scale.getType());
  if (!scaleType || scaleType.getRank() != 1 || !xType || xType.getRank() < 1)
    return false;
  if (xType.isDynamicDim(xType.getRank() - 1) || scaleType.isDynamicDim(0))
    return false;
  return scaleType.getDimSize(0) == xType.getDimSize(xType.getRank() - 1) &&
         scaleType.getElementType() == xType.getElementType();
}

struct RmsNormFusion : public mlir::RewritePattern {
  RmsNormFusion(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/3, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *mul,
                  mlir::PatternRewriter &rewriter) const override {
    if (mul->getNumOperands() != 2 || mul->getNumResults() != 1)
      return rewriter.notifyMatchFailure(mul, "mul.arity");

    auto outType =
        mlir::dyn_cast<mlir::RankedTensorType>(mul->getResult(0).getType());
    if (!outType || outType.getRank() < 1)
      return rewriter.notifyMatchFailure(mul, "mul.out_unranked");

    for (unsigned scaleIdx : {0u, 1u}) {
      mlir::Value scale = mul->getOperand(scaleIdx);
      mlir::Value normalizedSide = mul->getOperand(1u - scaleIdx);

      mlir::Operation *outCast = nullptr;
      mlir::Value normalized = peekCast(normalizedSide, &outCast);
      if (outCast && !hasOnlyUser(outCast->getResult(0), mul))
        continue;

      mlir::Operation *normOp = normalized.getDefiningOp();
      if (!normOp)
        continue;

      mlir::Value xSource;
      mlir::Value sqrtVal;
      mlir::Operation *sqrtOp = nullptr;
      mlir::Operation *consumerOfSqrt = nullptr;

      // Path A: Mul(Cast?(x), inv_rms) with inv = Div(1,sqrt) |
      // Reciprocal(sqrt) or Mul(1, Reciprocal(sqrt)) after
      // BroadcastDivToMulReciprocal.
      if (isOnnxOp(normOp, "onnx.Mul") && normOp->getNumOperands() == 2) {
        bool matched = false;
        for (unsigned xIdx : {0u, 1u}) {
          mlir::Operation *maybeXCast = nullptr;
          mlir::Value xCand = peekCast(normOp->getOperand(xIdx), &maybeXCast);
          mlir::Value invCand = normOp->getOperand(1u - xIdx);
          mlir::Value sqrtCand;
          mlir::Operation *invCandOp = nullptr;
          mlir::Operation *sqrtUse = nullptr;
          if (!matchInvRms(invCand, sqrtCand, invCandOp, sqrtUse))
            continue;
          if (maybeXCast && !hasOnlyUser(maybeXCast->getResult(0), normOp))
            continue;
          if (!hasOnlyUser(invCandOp->getResult(0), normOp))
            continue;
          xSource = xCand;
          sqrtVal = sqrtCand;
          sqrtOp = sqrtVal.getDefiningOp();
          consumerOfSqrt = sqrtUse;
          matched = true;
          break;
        }
        if (!matched)
          continue;
        if (!hasOnlyUser(normOp->getResult(0), outCast ? outCast : mul))
          continue;
      }
      // Path B: Div(Cast?(x), sqrt)
      else if (isOnnxOp(normOp, "onnx.Div") && normOp->getNumOperands() == 2) {
        mlir::Operation *xCast = nullptr;
        xSource = peekCast(normOp->getOperand(0), &xCast);
        sqrtVal = normOp->getOperand(1);
        sqrtOp = sqrtVal.getDefiningOp();
        if (!isOnnxOp(sqrtOp, "onnx.Sqrt"))
          continue;
        if (xCast && !hasOnlyUser(xCast->getResult(0), normOp))
          continue;
        if (!hasOnlyUser(normOp->getResult(0), outCast ? outCast : mul))
          continue;
        consumerOfSqrt = normOp;
      } else {
        continue;
      }

      auto xType = mlir::dyn_cast<mlir::RankedTensorType>(xSource.getType());
      if (!isLastDimScale(scale, xType))
        continue;
      if (outType.getElementType() != xType.getElementType() ||
          outType.getShape() != xType.getShape())
        continue;

      if (!sqrtOp || !hasOnlyUser(sqrtOp->getResult(0), consumerOfSqrt))
        continue;

      mlir::Value addVal = sqrtOp->getOperand(0);
      mlir::Operation *addOp = addVal.getDefiningOp();
      if (!isOnnxOp(addOp, "onnx.Add") || addOp->getNumOperands() != 2)
        continue;
      std::optional<double> eps;
      mlir::Value meanVal;
      if (auto e0 = getScalarConstant(addOp->getOperand(1))) {
        eps = e0;
        meanVal = addOp->getOperand(0);
      } else if (auto e1 = getScalarConstant(addOp->getOperand(0))) {
        eps = e1;
        meanVal = addOp->getOperand(1);
      } else {
        continue;
      }
      if (!eps || *eps <= 0.0 || *eps > 1.0e-4)
        continue;
      if (!hasOnlyUser(addOp->getResult(0), sqrtOp))
        continue;

      mlir::Operation *rmOp = meanVal.getDefiningOp();
      auto meanInType = rmOp ? mlir::dyn_cast<mlir::RankedTensorType>(
                                   rmOp->getOperand(0).getType())
                             : mlir::RankedTensorType();
      if (!reduceMeanIsLastAxisKeepdims(rmOp, meanInType))
        continue;
      if (!hasOnlyUser(rmOp->getResult(0), addOp))
        continue;

      mlir::Value squareX;
      mlir::Operation *squareOp = nullptr;
      if (!matchSquare(rmOp->getOperand(0), squareX, squareOp))
        continue;
      if (squareX != xSource)
        continue;
      if (!hasOnlyUser(squareOp->getResult(0), rmOp))
        continue;

      // Private Cast into the square leg (SD3.5 stash form).
      mlir::Operation *squareCast = squareOp->getOperand(0).getDefiningOp();
      if (isCastLike(squareCast) &&
          !hasOnlyUser(squareCast->getResult(0), squareOp))
        continue;

      mlir::Location loc = mul->getLoc();
      auto sintType = mlir::IntegerType::get(rewriter.getContext(), 64,
                                             mlir::IntegerType::Signed);
      mlir::OperationState state(loc, "onnx.RMSNormalization");
      state.addOperands({xSource, scale});
      state.addTypes(outType);
      state.addAttribute("epsilon",
                         rewriter.getF32FloatAttr(static_cast<float>(*eps)));
      state.addAttribute("axis", mlir::IntegerAttr::get(sintType, -1));
      state.addAttribute("stash_type", mlir::IntegerAttr::get(sintType, 1));
      mlir::Value fused = rewriter.create(state)->getResult(0);
      rewriter.replaceOp(mul, fused);
      ++NumRmsNormFused;
      LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] fused RMSNorm chain at "
                              << loc << " eps=" << *eps << "\n");
      return mlir::success();
    }
    return rewriter.notifyMatchFailure(mul, "not_rms_norm_scale_mul");
  }
};

} // namespace

void populateRmsNormFusionPatterns(mlir::RewritePatternSet &patterns,
                                   mlir::MLIRContext *ctx) {
  patterns.add<RmsNormFusion>(ctx);
}

} // namespace hip
} // namespace mlir
