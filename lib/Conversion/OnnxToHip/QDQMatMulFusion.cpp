/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- QDQMatMulFusion.cpp ------------------------------------------------===//
//
// Fusion pattern: QuantizeLinear -> MatMul -> DequantizeLinear => hip.qmatmul
//
// Demonstrates pattern matching and fusion using the same infrastructure
// as other hip-ep patterns.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Pattern to fuse QDQ MatMul: onnx.QuantizeLinear -> onnx.MatMul -> onnx.DequantizeLinear
/// into hip.qmatmul
struct QDQMatMulFusionPattern : public mlir::RewritePattern {
  QDQMatMulFusionPattern(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DequantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *dqOp,
                  mlir::PatternRewriter &rewriter) const override {
    // Match DequantizeLinear
    if (dqOp->getNumOperands() < 2 || dqOp->getNumResults() != 1)
      return mlir::failure();

    mlir::Value matmulResult = dqOp->getOperand(0);
    mlir::Value outputScale = dqOp->getOperand(1);

    // Match MatMul
    auto matmulOp = matmulResult.getDefiningOp();
    if (!matmulOp || matmulOp->getName().getStringRef() != "onnx.MatMul")
      return mlir::failure();

    if (matmulOp->getNumOperands() < 2)
      return mlir::failure();

    mlir::Value quantizedLhs = matmulOp->getOperand(0);
    mlir::Value rhs = matmulOp->getOperand(1);

    // Match QuantizeLinear on LHS
    auto qOp = quantizedLhs.getDefiningOp();
    if (!qOp || qOp->getName().getStringRef() != "onnx.QuantizeLinear")
      return mlir::failure();

    if (qOp->getNumOperands() < 2)
      return mlir::failure();

    mlir::Value lhsInput = qOp->getOperand(0);
    mlir::Value lhsScale = qOp->getOperand(1);

    // Get context
    auto ctxOrFailure = getContextArg(dqOp, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = dqOp->getLoc();
    mlir::Type outputType = dqOp->getResult(0).getType();

    // Extract scale values (simplified - just use placeholders for demo)
    // In real implementation, would extract from onnx.Constant ops
    float lhsScaleValue = 0.1f;
    float rhsScaleValue = 1.0f;
    float outScaleValue = 0.2f;

    // Create output tensor
    auto tensorType = mlir::cast<mlir::RankedTensorType>(outputType);
    mlir::Value emptyTensor = rewriter.create<mlir::tensor::EmptyOp>(
        loc, tensorType.getShape(), tensorType.getElementType());

    // Create fused qmatmul
    auto qmatmulOp = rewriter.create<hip::QMatMulOp>(
        loc,
        tensorType,
        context,
        lhsInput,
        rhs,
        emptyTensor,
        rewriter.getF32FloatAttr(lhsScaleValue),
        rewriter.getF32FloatAttr(rhsScaleValue),
        rewriter.getF32FloatAttr(outScaleValue));

    // Replace dequantize
    rewriter.replaceOp(dqOp, qmatmulOp.getResult(0));

    return mlir::success();
  }
};

} // namespace

void populateQDQMatMulFusionPatterns(RewritePatternSet &patterns) {
  patterns.add<QDQMatMulFusionPattern>(patterns.getContext());
}

} // namespace hip
} // namespace mlir
