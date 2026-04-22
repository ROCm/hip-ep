/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Range -> hip.range
struct RangeToHip : public mlir::RewritePattern {
  RangeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Range", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
RangeToHip::matchAndRewrite(mlir::Operation *op,
                            mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Extract operands: start, limit, delta (all scalar tensors)
  mlir::Value start = op->getOperand(0);
  mlir::Value limit = op->getOperand(1);
  mlir::Value delta = op->getOperand(2);

  // Get result type - should be a 1-D tensor with dynamic dimension
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Create empty tensor with dynamic dimension for output
  // For Range, the output size is computed at runtime by the GPU kernel
  // based on the formula: max(ceil((limit - start) / delta), 0)
  // We need to provide a placeholder size value for the dynamic dimension
  // Use 0 as a placeholder - the actual size will be computed by the runtime
  llvm::SmallVector<mlir::Value> dynSizes;
  auto indexType = rewriter.getIndexType();
  mlir::Value placeholderSize = rewriter.create<mlir::arith::ConstantIndexOp>(loc, 0);
  dynSizes.push_back(placeholderSize);

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  // Create hip.range operation
  auto hipOp = mlir::hip::RangeOp::create(rewriter, loc, resultType, context,
                                          start, limit, delta, init);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void mlir::hip::populateRangeConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<RangeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
