/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SkipNormCastFold.cpp - Fold Casts around SkipSimplifiedLayerNorm ---===//
//
// Pre-lowering pattern for fp16 models whose residual stream is kept in fp32
// (gpt-oss): every SkipSimplifiedLayerNormalization is wrapped as
//
//   %s  = onnx.Cast(%skip16)  : f16 -> f32
//   %g  = onnx.Cast(%gamma16) : f16 -> f32
//   %y, ..., %sum = onnx.Custom(%res32, %s, %g)
//                     {function_name = "SkipSimplifiedLayerNormalization"}
//   %y16 = onnx.Cast(%y)      : f32 -> f16
//
// which costs three Cast launches per norm on top of the norm itself, one of
// them re-casting the constant gamma on every call. Both input Casts are
// exact and the output Cast is the rounding the norm kernel would apply
// anyway, so the norm is rebuilt to read %skip16 / %gamma16 directly and
// produce %y16, keeping %res32 and %sum in fp32:
//
//   %y16, ..., %sum = onnx.Custom(%res32, %skip16, %gamma16) {...}
//
// hip.skip_rms_norm carries the per-tensor types to the
// wrap_skip_simplified_layer_norm_mixed runtime entry. The fold is
// all-or-nothing (both input Casts, a sole output Cast, no bias) so the
// runtime only needs the one mixed dtype combination.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "skip-norm-cast-fold"

STATISTIC(NumSkipNormCastFolds,
          "Number of Cast -> SkipSimplifiedLayerNorm(f32) -> Cast folds");

namespace mlir {
namespace hip {

namespace {

static bool hasElementType(mlir::Value v, bool f16) {
  auto ty = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return ty &&
         (f16 ? ty.getElementType().isF16() : ty.getElementType().isF32());
}

/// The f16 source of an exact `onnx.Cast` f16 -> f32 defining \p v, or null.
static mlir::Value castF16Source(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def || def->getName().getStringRef() != "onnx.Cast" ||
      def->getNumOperands() != 1 || !hasElementType(v, /*f16=*/false) ||
      !hasElementType(def->getOperand(0), /*f16=*/true))
    return nullptr;
  return def->getOperand(0);
}

struct FoldCastsIntoSkipSimplifiedLayerNorm : public mlir::RewritePattern {
  FoldCastsIntoSkipSimplifiedLayerNorm(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto fn = op->getAttrOfType<mlir::StringAttr>("function_name");
    auto domain = op->getAttrOfType<mlir::StringAttr>("domain_name");
    if (!fn || fn.getValue() != "SkipSimplifiedLayerNormalization" || !domain ||
        domain.getValue() != "com.microsoft")
      return rewriter.notifyMatchFailure(op, "not SkipSimplifiedLayerNorm");

    const unsigned numOps = op->getNumOperands();
    if (numOps < 3 || numOps > 4 ||
        (numOps == 4 &&
         !mlir::isa<mlir::NoneType>(op->getOperand(3).getType())))
      return rewriter.notifyMatchFailure(op, "bias present");
    if (!hasElementType(op->getOperand(0), /*f16=*/false))
      return rewriter.notifyMatchFailure(op, "input not f32");

    mlir::Value skip16 = castF16Source(op->getOperand(1));
    mlir::Value gamma16 = castF16Source(op->getOperand(2));
    if (!skip16 || !gamma16)
      return rewriter.notifyMatchFailure(op, "skip/gamma not Cast f16->f32");

    mlir::Value y = op->getResult(0);
    if (!y.hasOneUse())
      return rewriter.notifyMatchFailure(op, "output not solely cast");
    mlir::Operation *outCast = *y.getUsers().begin();
    if (outCast->getName().getStringRef() != "onnx.Cast" ||
        !hasElementType(outCast->getResult(0), /*f16=*/true))
      return rewriter.notifyMatchFailure(op, "output not cast to f16");

    // Results 1-2 (mean, inv_std_var) are training-only and lowered to
    // placeholders; result 3 is the residual and stays f32.
    if (op->getNumResults() > 3 &&
        !mlir::isa<mlir::NoneType>(op->getResult(3).getType()) &&
        !hasElementType(op->getResult(3), /*f16=*/false))
      return rewriter.notifyMatchFailure(op, "residual sum not f32");

    SmallVector<mlir::Type> resultTypes(op->getResultTypes());
    resultTypes[0] = outCast->getResult(0).getType();

    mlir::OperationState state(op->getLoc(), "onnx.Custom");
    state.addOperands({op->getOperand(0), skip16, gamma16});
    state.addTypes(resultTypes);
    state.addAttributes(op->getAttrs());
    mlir::Operation *folded = rewriter.create(state);

    mlir::Operation *skipCast = op->getOperand(1).getDefiningOp();
    mlir::Operation *gammaCast = op->getOperand(2).getDefiningOp();
    rewriter.replaceOp(outCast, folded->getResult(0));
    for (unsigned i = 1; i < op->getNumResults(); ++i)
      rewriter.replaceAllUsesWith(op->getResult(i), folded->getResult(i));
    rewriter.eraseOp(op);
    if (skipCast->use_empty())
      rewriter.eraseOp(skipCast);
    if (gammaCast != skipCast && gammaCast->use_empty())
      rewriter.eraseOp(gammaCast);
    ++NumSkipNormCastFolds;
    return mlir::success();
  }
};

} // namespace

void populateSkipNormCastFoldPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<FoldCastsIntoSkipSimplifiedLayerNorm>(ctx);
}

} // namespace hip
} // namespace mlir
