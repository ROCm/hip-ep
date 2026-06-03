/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Exp -> hip.exp
/// Unary element-wise exponential: Y = exp(X). Float types.
//
// Before:
//   %y = "onnx.Exp"(%x) : (tensor<1024x1x64x64xf16>) ->
//   tensor<1024x1x64x64xf16>
//
// After:
//   %init = tensor.empty() : tensor<1024x1x64x64xf16>
//   %y = hip.exp(%ctx) ins(%x : tensor<1024x1x64x64xf16>)
//                     outs(%init : tensor<1024x1x64x64xf16>)
struct ExpToHip : public mlir::RewritePattern {
  ExpToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Exp", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    if (!mlir::isa<mlir::RankedTensorType>(input.getType()))
      return rewriter.notifyMatchFailure(
          op, "onnx.Exp lowering expects a ranked tensor input");
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = mlir::hip::ExpOp::create(rewriter, loc, resultType, context,
                                          input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateExpConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<ExpToHip>(ctx);
}

} // namespace hip
} // namespace mlir
