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
///
/// KNOWN LIMITATION (NaN): the `A >= B  <=>  !(A < B)` identity holds only for
/// a total order. With NaN operands it deviates from ONNX/IEEE-754: ONNX
/// requires `NaN >= x` to be false, but `hip.less` (C++ `<`) returns false for
/// any NaN comparison, so `!(A < B)` yields true. This is exact for integer
/// inputs (the dominant index/mask use) and for NaN-free float models; float
/// models that can produce NaN are not supported by this decomposition. A
/// NaN-correct lowering would need e.g. `(B < A) || (A == B)` (extra ops) or a
/// dedicated `>=` kernel.
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
        createOnnxBroadcastEmptyTensor(rewriter, loc, resultType, {a, b}, op);
    if (mlir::failed(lessInit))
      return rewriter.notifyMatchFailure(
          op, "GreaterOrEqual: cannot infer dynamic result dimensions from "
              "operands (both inputs must be ranked tensors)");
    mlir::Value less =
        mlir::hip::LessOp::create(rewriter, loc, context, a, b, *lessInit)
            ->getResult(0);

    // result = !less
    auto notInit = createSameShapeEmptyTensor(rewriter, loc, resultType, less);
    if (mlir::failed(notInit))
      return rewriter.notifyMatchFailure(
          op, "GreaterOrEqual negation shape must match comparison shape");
    auto notOp =
        mlir::hip::NotOp::create(rewriter, loc, context, less, *notInit);
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
