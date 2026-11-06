/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SoftplusCastFold.cpp - Fold the Casts around an fp32 Softplus ------===//
//
// Pre-lowering pattern for fp16 models that evaluate Softplus in fp32:
//
//   %x32 = onnx.Cast(%x16)      : f16 -> f32
//   %y32 = onnx.Softplus(%x32)
//   %y16 = onnx.Cast(%y32)      : f32 -> f16
//
// The fp16 softplus kernel widens each element to f32, evaluates the same f32
// softplus and rounds once on store, which is exactly the three-op chain
// above, so the chain becomes one onnx.Softplus on %x16. The input Cast stays
// if it has other users.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Statistic.h"

#define DEBUG_TYPE "softplus-cast-fold"

STATISTIC(NumSoftplusCastFolds,
          "Number of Cast -> Softplus(f32) -> Cast folds");

namespace mlir {
namespace hip {

namespace {

static bool hasElementType(mlir::Value v, bool f16) {
  auto ty = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return ty &&
         (f16 ? ty.getElementType().isF16() : ty.getElementType().isF32());
}

struct FoldCastsIntoSoftplus : public mlir::RewritePattern {
  FoldCastsIntoSoftplus(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Softplus", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "unexpected arity");
    mlir::Value x = op->getOperand(0);
    mlir::Value y = op->getResult(0);
    if (!hasElementType(x, /*f16=*/false) || !hasElementType(y, /*f16=*/false))
      return rewriter.notifyMatchFailure(op, "not f32");

    mlir::Operation *inCast = x.getDefiningOp();
    if (!inCast || inCast->getName().getStringRef() != "onnx.Cast" ||
        inCast->getNumOperands() != 1 ||
        !hasElementType(inCast->getOperand(0), /*f16=*/true))
      return rewriter.notifyMatchFailure(op, "input not Cast f16->f32");

    if (!y.hasOneUse())
      return rewriter.notifyMatchFailure(op, "output not solely cast");
    mlir::Operation *outCast = *y.getUsers().begin();
    if (outCast->getName().getStringRef() != "onnx.Cast" ||
        !hasElementType(outCast->getResult(0), /*f16=*/true))
      return rewriter.notifyMatchFailure(op, "output not cast to f16");

    mlir::OperationState state(op->getLoc(), "onnx.Softplus");
    state.addOperands({inCast->getOperand(0)});
    state.addTypes({outCast->getResult(0).getType()});
    state.addAttributes(op->getAttrs());
    mlir::Operation *folded = rewriter.create(state);

    rewriter.replaceOp(outCast, folded->getResult(0));
    rewriter.eraseOp(op);
    if (inCast->use_empty())
      rewriter.eraseOp(inCast);
    ++NumSoftplusCastFolds;
    return mlir::success();
  }
};

} // namespace

void populateSoftplusCastFoldPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<FoldCastsIntoSoftplus>(ctx);
}

} // namespace hip
} // namespace mlir
