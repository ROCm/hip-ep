/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.GreaterOrEqual -> onnx.Not(onnx.Less(A, B))
///
/// GreaterOrEqual(A, B) is the elementwise complement of Less(A, B):
///   A >= B  <=>  !(A < B)
/// Both onnx.Less and onnx.Not already have ONNX->HIP conversion patterns,
/// so the emitted ops are picked up in the same greedy rewrite round. No new
/// hip.* op, lowering or runtime implementation is required.
struct GreaterOrEqualDecompose : public mlir::RewritePattern {
  GreaterOrEqualDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GreaterOrEqual", /*benefit=*/1, ctx) {}

  static mlir::Value createOnnxOp(mlir::PatternRewriter &rewriter,
                                  mlir::Location loc, llvm::StringRef opName,
                                  llvm::ArrayRef<mlir::Value> operands,
                                  mlir::Type resultType) {
    mlir::OperationState state(loc, opName);
    state.addOperands(operands);
    state.addTypes(resultType);
    return rewriter.create(state)->getResult(0);
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.GreaterOrEqual decompose expects a ranked tensor result");

    // %less has the same (boolean, broadcasted) type as the final result.
    mlir::Value less =
        createOnnxOp(rewriter, loc, "onnx.Less", {a, b}, resultType);
    mlir::Value result =
        createOnnxOp(rewriter, loc, "onnx.Not", {less}, resultType);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

} // namespace

void populateGreaterOrEqualConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<GreaterOrEqualDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
