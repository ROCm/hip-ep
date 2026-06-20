/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.LessOrEqual -> hip.not(hip.less(B, A))
///
/// LessOrEqual(A, B) is the elementwise complement of Greater(A, B), and
/// Greater(A, B) == Less(B, A):
///   A <= B  <=>  !(A > B)  <=>  !(B < A)
/// The decomposition emits the hip.* ops directly (rather than onnx.Less /
/// onnx.Not). convert-onnx-to-hip runs its greedy rewrite with
/// GreedyRewriteStrictness::ExistingOps, so freshly created onnx.* ops would
/// NOT be revisited by the onnx.Less / onnx.Not patterns and would survive to
/// bufferization. Building hip.less / hip.not here keeps the lowering
/// self-contained. No new hip.* op, lowering or runtime is required (hip.less
/// and hip.not already exist).
///
/// KNOWN LIMITATION (NaN): the `A <= B  <=>  !(B < A)` identity holds only for
/// a total order. With NaN operands it deviates from ONNX/IEEE-754: ONNX
/// requires `x <= NaN` to be false, but `hip.less` (C++ `<`) returns false for
/// any NaN comparison, so `!(B < A)` yields true. This is exact for integer
/// inputs (the dominant index/mask use) and for NaN-free float models; float
/// models that can produce NaN are not supported by this decomposition. A
/// NaN-correct lowering would need e.g. `(A < B) || (A == B)` (extra ops) or a
/// dedicated `<=` kernel.
struct LessOrEqualDecompose : public mlir::RewritePattern {
  LessOrEqualDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.LessOrEqual", /*benefit=*/1, ctx) {}

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
          op, "onnx.LessOrEqual decompose expects a ranked tensor result");

    // less = (B < A); same (boolean, broadcasted) type as the final result.
    mlir::FailureOr<mlir::Value> lessInit =
        createBroadcastEmptyTensor(rewriter, loc, resultType, {b, a});
    if (mlir::failed(lessInit))
      return rewriter.notifyMatchFailure(
          op, "LessOrEqual: cannot infer dynamic result dimensions from "
              "operands (both inputs must be ranked tensors)");
    mlir::Value less =
        mlir::hip::LessOp::create(rewriter, loc, context, b, a, *lessInit)
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

void populateLessOrEqualConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<LessOrEqualDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
