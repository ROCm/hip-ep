/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct SoftplusToHip : public mlir::RewritePattern {
  SoftplusToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Softplus", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SoftplusToHip::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  // Get context argument (required for all HIP ops)
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // ONNX Softplus: input X -> output Y
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Create DPS init tensor (zero-cost empty tensor)
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Create HIP softplus operation
  auto hipOp = mlir::hip::SoftplusOp::create(rewriter, loc, resultType, context,
                                             input, init);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateSoftplusConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<SoftplusToHip>(ctx);
}

} // namespace hip
} // namespace mlir
