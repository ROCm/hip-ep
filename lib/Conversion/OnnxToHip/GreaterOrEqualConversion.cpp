/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.GreaterOrEqual -> hip.not(hip.less(A, B))
///
/// GreaterOrEqual(A, B) is the elementwise complement of Less(A, B):
///   A >= B  <=>  !(A < B)
/// The decomposition emits the hip.* ops directly (rather than onnx.Less /
/// onnx.Not). convert-onnx-to-hip runs its greedy rewrite with
/// GreedyRewriteStrictness::ExistingOps, so freshly created onnx.* ops would
/// NOT be revisited by the onnx.Less / onnx.Not patterns and would survive to
/// bufferization. Building hip.less / hip.not here keeps the lowering
/// self-contained. No new hip.* op, lowering or runtime is required (hip.less
/// and hip.not already exist).
struct GreaterOrEqualDecompose : public mlir::RewritePattern {
  GreaterOrEqualDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GreaterOrEqual", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.GreaterOrEqual decompose expects a ranked tensor result");

    // less = (A < B); same (boolean, broadcasted) type as the final result.
    mlir::FailureOr<mlir::Value> lessInit =
        createBroadcastEmptyTensor(rewriter, loc, resultType, {a, b});
    if (mlir::failed(lessInit))
      return rewriter.notifyMatchFailure(
          op, "GreaterOrEqual: no ranked operand spans dynamic result dim");
    mlir::Value less =
        mlir::hip::LessOp::create(rewriter, loc, context, a, b, *lessInit)
            ->getResult(0);

    // result = !less
    mlir::Value notInit = createEmptyTensor(rewriter, loc, resultType, less);
    auto notOp =
        mlir::hip::NotOp::create(rewriter, loc, context, less, notInit);
    rewriter.replaceOp(op, notOp->getResult(0));
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
