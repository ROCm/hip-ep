/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Log -> hip.log
/// Unary element-wise natural logarithm: Y = log(X). Float types.
//
// Before:
//   %y = "onnx.Log"(%x) : (tensor<3x4xf32>) -> tensor<3x4xf32>
//
// After:
//   %init = tensor.empty() : tensor<3x4xf32>
//   %y = hip.log(%ctx) ins(%x : tensor<3x4xf32>)
//                     outs(%init : tensor<3x4xf32>)
struct LogToHip : public mlir::RewritePattern {
  LogToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Log", /*benefit=*/1, ctx) {}

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
          op, "onnx.Log lowering expects a ranked tensor input");
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = mlir::hip::LogOp::create(rewriter, loc, resultType, context,
                                          input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateLogConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<LogToHip>(ctx);
}

} // namespace hip
} // namespace mlir
