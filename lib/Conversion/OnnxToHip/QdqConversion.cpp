/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {
static mlir::Value getOptionalOperand(mlir::Operation *op, size_t idx) {
  if (idx >= op->getNumOperands())
    return nullptr;
  mlir::Value val = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(val.getType()))
    return nullptr;
  return val;
}

struct QuantizeLinearDecompose : public mlir::RewritePattern {
  QuantizeLinearDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.QuantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // check operands and results. should be 2/3 input and 1 output.
    bool operandsValid = op->getNumOperands() == 2 || op->getNumOperands() == 3;
    if (!operandsValid || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "expected 2/3 inputs and 1 output");
    mlir::Value input = op->getOperand(0);
    mlir::Value scale = op->getOperand(1);
    mlir::Value zero_point = getOptionalOperand(op, 2);
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "expected ranked output");
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

    auto loc = op->getLoc();
    llvm::SmallVector<mlir::Value> operands = {context, input, scale};
    if (zero_point)
      operands.push_back(zero_point);
    auto quantizeLinearOp =
        mlir::hip::QuantizeLinearOp::create(rewriter, loc, operands);
    rewriter.replaceOp(op, quantizeLinearOp.getResult());
    return success();
  }
};
struct DequantizeLinearDecompose : public mlir::RewritePattern {
  DequantizeLinearDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DequantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {

    return success();
  }
};

} // namespace

void populateQuantizeLinearConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<QuantizeLinearDecompose>(ctx);
}

void populateDequantizeLinearConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<DequantizeLinearDecompose>(ctx);
}
} // namespace hip
} // namespace mlir
