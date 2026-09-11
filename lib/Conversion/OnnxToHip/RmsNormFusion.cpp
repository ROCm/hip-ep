/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RmsNormFusion.cpp - Fold a decomposed RMS-normalize chain ----------===//
//
// Some exports emit an unweighted RMS normalization
//
//   y = x / sqrt(mean(x^2) + eps)
//
// as primitive ops rather than as `SimplifiedLayerNormalization`. Unlike the
// weighted block norms, this form carries no learned gamma, so exporters have
// no fused op to reach for and spell it out instead. Attention q/k
// normalization is the usual source: the reduction is over the head dimension,
// so every launch in the chain works on `heads x head_dim` elements and the
// per-row results are `heads` elements wide. At single-token decode that is a
// handful of values per kernel, and the chain costs six to eight launches per
// layer to normalize a few kilobytes.
//
// Exporters commonly wrap the chain in a f16 -> f32 -> f16 cast pair to keep
// the sum of squares off fp16. `hip.rms_norm` already accumulates in fp32
// regardless of the I/O type, so when the chain is bracketed that way the
// casts carry no numerical meaning and are folded in as well.
//
//   Before (rooted on onnx.ReduceMean, leaves -> root):
//     %xf   = onnx.Cast(%x)          {to = f32}  : tensor<1x1x8x256xf16> -> f32
//     %sq   = onnx.Mul(%xf, %xf)                 : tensor<1x1x8x256xf32>
//     %mean = onnx.ReduceMean(%sq)   {axes = [-1], keepdims = 1}
//                                                : tensor<1x1x8x1xf32>
//     %eps  = onnx.Constant dense<9.99999997E-7> : tensor<1xf32>
//     %add  = onnx.Add(%mean, %eps)              : tensor<1x1x8x1xf32>
//     %rms  = onnx.Sqrt(%add)                    : tensor<1x1x8x1xf32>
//     %inv  = onnx.Reciprocal(%rms)              : tensor<1x1x8x1xf32>
//     %nf   = onnx.Mul(%xf, %inv)                : tensor<1x1x8x256xf32>
//     %y    = onnx.Cast(%nf)         {to = f16}  : tensor<1x1x8x256xf16>
//
//   After:
//     %s = onnx.Constant dense<1.0> : tensor<256xf16>
//     %y = onnx.Custom(%x, %s)
//            {function_name = "SimplifiedLayerNormalization",
//             epsilon = 9.99999997E-7 : f32, axis = -1 : si64,
//             stash_type = 1 : si64}
//          : (tensor<1x1x8x256xf16>, tensor<256xf16>) -> tensor<1x1x8x256xf16>
//
// The all-ones scale is what turns the weighted fused op into the unweighted
// identity; NormConversion lowers the result to `hip.rms_norm`.
//
// Matching only accepts the canonical spelling: `Mul(x, x)` for the square,
// `onnx.Sqrt`, and `Mul(x, Reciprocal(r))` for the divide. Exports that use
// `Pow(x, 2)` / `Pow(r, 0.5)` or a broadcasting `Div` still fuse, because
// PowDecompose and BroadcastDivToMulReciprocal canonicalize those forms in the
// same pre-lowering fixed-point loop and this pattern is rooted on the
// ReduceMean they leave untouched, so it gets another attempt once they
// settle. A non-broadcasting `Div` is matched directly since nothing rewrites
// it.
//
// Rooted on `onnx.ReduceMean` (the one op that appears exactly once in the
// chain) and run BEFORE `lowerOnnxConstants`, so the epsilon literal is still
// reachable through the generic `onnx.Constant` producer.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cmath>

#define DEBUG_TYPE "rms-norm-fusion"

STATISTIC(NumRmsNormFused,
          "Number of decomposed RMS-normalize chains folded back to a fused "
          "SimplifiedLayerNormalization");

namespace mlir {
namespace hip {

namespace {

/// Evaluate a single fp scalar defined by an `onnx.Constant`, peeking through
/// the dtype-only wrappers exporters leave around epsilon literals.
static std::optional<double> getScalarFloatConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  llvm::StringRef opName = def->getName().getStringRef();

  if (opName == "onnx.Constant") {
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
    return std::nullopt;
  }

  if ((opName == "onnx.Cast" || opName == "onnx.CastLike") &&
      def->getNumOperands() >= 1)
    return getScalarFloatConstant(def->getOperand(0));

  return std::nullopt;
}

static bool isOp(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

/// Single-use producer of `v` when it is `name`, else null. The chain is
/// only worth folding when each intermediate feeds nothing else; a shared
/// intermediate would survive the rewrite and leave the launches in place.
static mlir::Operation *soleProducer(mlir::Value v, llvm::StringRef name) {
  mlir::Operation *def = v.getDefiningOp();
  if (!isOp(def, name) || !def->getResult(0).hasOneUse())
    return nullptr;
  return def;
}

/// Sole consumer of `op`'s result when it is `name`, else null.
static mlir::Operation *soleConsumer(mlir::Operation *op,
                                     llvm::StringRef name) {
  if (!op || !op->getResult(0).hasOneUse())
    return nullptr;
  mlir::Operation *user = *op->getResult(0).getUsers().begin();
  return isOp(user, name) ? user : nullptr;
}

/// The axes an `onnx.ReduceMean` reduces, whether spelled as the pre-opset-18
/// `axes` attribute or the opset-18 constant operand. Fails on a non-constant
/// operand, which cannot be checked statically.
static std::optional<llvm::SmallVector<int64_t>>
getReduceAxes(mlir::Operation *op) {
  llvm::SmallVector<int64_t> axes;
  if (op->getNumOperands() > 1) {
    auto def = op->getOperand(1).getDefiningOp();
    if (!isOp(def, "onnx.Constant"))
      return std::nullopt;
    auto dense =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
    if (!dense)
      return std::nullopt;
    for (const llvm::APInt &a : dense.getValues<llvm::APInt>())
      axes.push_back(a.getSExtValue());
    return axes;
  }
  auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes");
  if (!axesAttr)
    return std::nullopt;
  for (mlir::Attribute a : axesAttr)
    axes.push_back(mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  return axes;
}

struct DecomposedRmsNormToFused : public mlir::RewritePattern {
  DecomposedRmsNormToFused(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMean", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *meanOp,
                  mlir::PatternRewriter &rewriter) const override {
    if (meanOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(meanOp, "mean.arity");

    // ── the reduction must be a keepdims mean over the trailing axis ────
    if (auto keepdims = meanOp->getAttrOfType<mlir::IntegerAttr>("keepdims"))
      if (keepdims.getSInt() != 1)
        return rewriter.notifyMatchFailure(meanOp, "mean.not_keepdims");

    auto sqType =
        mlir::dyn_cast<mlir::RankedTensorType>(meanOp->getOperand(0).getType());
    if (!sqType)
      return rewriter.notifyMatchFailure(meanOp, "mean.input_not_ranked");
    int64_t rank = sqType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(meanOp, "mean.scalar_input");

    auto axes = getReduceAxes(meanOp);
    if (!axes)
      return rewriter.notifyMatchFailure(meanOp, "mean.axes_not_static");
    if (axes->size() != 1)
      return rewriter.notifyMatchFailure(meanOp, "mean.multi_axis");
    int64_t axis = (*axes)[0];
    int64_t normAxis = axis < 0 ? axis + rank : axis;
    if (normAxis != rank - 1)
      return rewriter.notifyMatchFailure(meanOp, "mean.axis_not_trailing");

    // rms_norm derives its row count from input_elements / scale_elements, so
    // the normalized extent has to be a compile-time constant.
    int64_t n = sqType.getShape()[normAxis];
    if (n == mlir::ShapedType::kDynamic || n <= 0)
      return rewriter.notifyMatchFailure(meanOp, "mean.dynamic_extent");

    // ── back-walk: the reduced value is x squared ───────────────────────
    mlir::Operation *sqOp = soleProducer(meanOp->getOperand(0), "onnx.Mul");
    if (!sqOp || sqOp->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(meanOp, "square.not_mul");
    mlir::Value x = sqOp->getOperand(0);
    if (sqOp->getOperand(1) != x)
      return rewriter.notifyMatchFailure(meanOp, "square.operands_differ");

    // ── forward-walk: (+eps)? -> sqrt -> divide ─────────────────────────
    // Epsilon is optional: exporters that rely on the input never being all
    // zero emit Sqrt directly on the mean.
    double epsilon = 0.0;
    mlir::Operation *addOp = soleConsumer(meanOp, "onnx.Add");
    mlir::Operation *preSqrt = meanOp;
    if (addOp) {
      if (addOp->getNumOperands() != 2)
        return rewriter.notifyMatchFailure(meanOp, "eps.arity");
      mlir::Value other = addOp->getOperand(0) == meanOp->getResult(0)
                              ? addOp->getOperand(1)
                              : addOp->getOperand(0);
      auto epsVal = getScalarFloatConstant(other);
      if (!epsVal)
        return rewriter.notifyMatchFailure(meanOp, "eps.not_scalar_const");
      epsilon = *epsVal;
      preSqrt = addOp;
    }

    mlir::Operation *sqrtOp = soleConsumer(preSqrt, "onnx.Sqrt");
    if (!sqrtOp)
      return rewriter.notifyMatchFailure(meanOp, "sqrt.missing");

    // Either Mul(x, Reciprocal(rms)) or a bare Div(x, rms).
    mlir::Operation *divideOp = nullptr;
    mlir::Operation *recipOp = soleConsumer(sqrtOp, "onnx.Reciprocal");
    if (recipOp) {
      divideOp = soleConsumer(recipOp, "onnx.Mul");
      if (!divideOp || divideOp->getNumOperands() != 2)
        return rewriter.notifyMatchFailure(meanOp, "divide.not_mul");
      mlir::Value data = divideOp->getOperand(0) == recipOp->getResult(0)
                             ? divideOp->getOperand(1)
                             : divideOp->getOperand(0);
      if (data != x)
        return rewriter.notifyMatchFailure(meanOp, "divide.data_mismatch");
    } else {
      divideOp = soleConsumer(sqrtOp, "onnx.Div");
      if (!divideOp || divideOp->getNumOperands() != 2)
        return rewriter.notifyMatchFailure(meanOp, "divide.not_div");
      // Div is not commutative: x must be the numerator.
      if (divideOp->getOperand(0) != x ||
          divideOp->getOperand(1) != sqrtOp->getResult(0))
        return rewriter.notifyMatchFailure(meanOp, "divide.operand_order");
    }

    // ── optionally absorb the fp32 cast bracket ─────────────────────────
    // rms_norm accumulates in fp32 for f16 I/O, so normalizing the pre-cast
    // value computes the same thing the cast pair was there to guarantee.
    mlir::Value fusedInput = x;
    mlir::Operation *replacedOp = divideOp;
    mlir::Operation *inCastOp = x.getDefiningOp();
    mlir::Operation *outCastOp = soleConsumer(divideOp, "onnx.Cast");
    if (isOp(inCastOp, "onnx.Cast") && outCastOp) {
      mlir::Value preCast = inCastOp->getOperand(0);
      auto preType = mlir::dyn_cast<mlir::RankedTensorType>(preCast.getType());
      auto outType = mlir::dyn_cast<mlir::RankedTensorType>(
          outCastOp->getResult(0).getType());
      if (preType && outType &&
          preType.getElementType() == outType.getElementType() &&
          preType.getShape() == outType.getShape()) {
        fusedInput = preCast;
        replacedOp = outCastOp;
      }
    }

    auto inputType = mlir::cast<mlir::RankedTensorType>(fusedInput.getType());
    auto elemType = inputType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16())
      return rewriter.notifyMatchFailure(meanOp, "input.elem_type_not_float");

    // ── rewrite ─────────────────────────────────────────────────────────
    mlir::Location loc = replacedOp->getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();

    llvm::APFloat one(1.0f);
    bool losesInfo = false;
    one.convert(mlir::cast<mlir::FloatType>(elemType).getFloatSemantics(),
                llvm::APFloat::rmNearestTiesToEven, &losesInfo);
    auto scaleType = mlir::RankedTensorType::get({n}, elemType);
    llvm::SmallVector<llvm::APFloat> scaleData(static_cast<size_t>(n), one);
    mlir::OperationState scaleState(loc, "onnx.Constant");
    scaleState.addTypes(scaleType);
    scaleState.addAttribute("value",
                            mlir::DenseElementsAttr::get(scaleType, scaleData));
    rewriter.setInsertionPoint(replacedOp);
    mlir::Value scale = rewriter.create(scaleState)->getResult(0);

    // NormConversion reads axis / stash_type with getSInt(), so both must be
    // SIGNED i64 attrs.
    auto sintType = mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed);
    mlir::OperationState customState(loc, "onnx.Custom");
    customState.addOperands({fusedInput, scale});
    customState.addTypes(replacedOp->getResult(0).getType());
    customState.addAttribute(
        "function_name",
        rewriter.getStringAttr("SimplifiedLayerNormalization"));
    customState.addAttribute("domain_name",
                             rewriter.getStringAttr("com.microsoft"));
    customState.addAttribute(
        "epsilon", rewriter.getF32FloatAttr(static_cast<float>(epsilon)));
    customState.addAttribute("axis", mlir::IntegerAttr::get(sintType, -1));
    customState.addAttribute("stash_type", mlir::IntegerAttr::get(sintType, 1));
    if (auto outputs =
            replacedOp->getAttrOfType<mlir::ArrayAttr>("node.outputs"))
      customState.addAttribute("node.outputs", outputs);
    if (auto nodeName =
            replacedOp->getAttrOfType<mlir::StringAttr>("onnx_node_name"))
      customState.addAttribute("onnx_node_name", nodeName);
    mlir::Operation *fused = rewriter.create(customState);

    rewriter.replaceOp(replacedOp, fused->getResult(0));

    // Each matched intermediate had exactly one use, so erasing root-ward
    // cascades. Shared leaves (the epsilon constant, and the input cast when
    // it still feeds something else) are left to the global `onnx.*
    // use_empty` DCE walk at the end of ConvertOnnxToHipPass.
    auto eraseIfDead = [&rewriter](mlir::Operation *op) {
      if (op && op->use_empty())
        rewriter.eraseOp(op);
    };
    if (replacedOp == outCastOp)
      eraseIfDead(divideOp);
    eraseIfDead(recipOp);
    eraseIfDead(sqrtOp);
    eraseIfDead(addOp);
    eraseIfDead(meanOp);
    eraseIfDead(sqOp);
    eraseIfDead(inCastOp);

    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] fused chain at " << loc << " N=" << n
               << " epsilon=" << epsilon
               << (replacedOp == outCastOp ? " (casts absorbed)" : "") << "\n");
    ++NumRmsNormFused;
    return mlir::success();
  }
};

} // namespace

void populateRmsNormFusionPatterns(mlir::RewritePatternSet &patterns,
                                   mlir::MLIRContext *ctx) {
  patterns.add<DecomposedRmsNormToFused>(ctx);
}

} // namespace hip
} // namespace mlir
