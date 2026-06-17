/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.LessOrEqual -> onnx.Not(onnx.Less(B, A))
///
/// LessOrEqual(A, B) is the elementwise complement of Greater(A, B), and
/// Greater(A, B) == Less(B, A):
///   A <= B  <=>  !(A > B)  <=>  !(B < A)
/// Both onnx.Less and onnx.Not already have ONNX->HIP conversion patterns,
/// so the emitted ops are picked up in the same greedy rewrite round. No new
/// hip.* op, lowering or runtime implementation is required.
struct LessOrEqualDecompose : public mlir::RewritePattern {
  LessOrEqualDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.LessOrEqual", /*benefit=*/1, ctx) {}

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
          op, "onnx.LessOrEqual decompose expects a ranked tensor result");

    // Less(B, A) has the same (boolean, broadcasted) type as the final result.
    mlir::Value less =
        createOnnxOp(rewriter, loc, "onnx.Less", {b, a}, resultType);
    mlir::Value result =
        createOnnxOp(rewriter, loc, "onnx.Not", {less}, resultType);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

} // namespace

void populateLessOrEqualConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<LessOrEqualDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
