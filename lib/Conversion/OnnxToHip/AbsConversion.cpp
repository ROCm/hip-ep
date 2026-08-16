/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Abs -> hip.abs
/// Unary element-wise absolute value: Y = abs(X).
//
// Before:
//   %y = "onnx.Abs"(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
//
// After:
//   %init = tensor.empty() : tensor<3x4xf32>
//   %y = hip.abs(%ctx) ins(%x : tensor<3x4xf32>)
//                     outs(%init : tensor<3x4xf32>)
struct AbsToHip : public mlir::RewritePattern {
  AbsToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Abs", /*benefit=*/1, ctx) {}

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
          op, "onnx.Abs lowering expects a ranked tensor input");
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto init = createSameShapeEmptyTensor(rewriter, loc, resultType, input);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "Abs result type must match the input shape");
    auto hipOp = mlir::hip::AbsOp::create(rewriter, loc, resultType, context,
                                          input, *init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateAbsConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<AbsToHip>(ctx);
}

} // namespace hip
} // namespace mlir
