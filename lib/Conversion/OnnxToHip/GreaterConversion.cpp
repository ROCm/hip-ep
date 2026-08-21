/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Greater -> hip.less(B, A)
///
/// Greater(A, B) is elementwise A > B, equivalent to Less(B, A):
///   A > B  <=>  B < A
/// The decomposition emits hip.less directly (rather than onnx.Less).
/// convert-onnx-to-hip runs its greedy rewrite with
/// GreedyRewriteStrictness::ExistingOps, so freshly created onnx.* ops would
/// NOT be revisited by the onnx.Less pattern and would survive to
/// bufferization. Building hip.less here keeps the lowering self-contained.
/// No new hip.* op, lowering or runtime is required (hip.less already exists).
///
/// KNOWN LIMITATION (NaN): the `A > B  <=>  B < A` identity holds only for
/// a total order. With NaN operands it deviates from ONNX/IEEE-754: ONNX
/// requires `NaN > x` to be false, but `hip.less` (C++ `<`) returns false for
/// any NaN comparison, so `B < A` with NaN in A can yield false when ONNX
/// expects false for `A > B` with NaN in A — the dominant integer/mask use is
/// exact; NaN-bearing float models are not supported by this decomposition.
struct GreaterDecompose : public mlir::RewritePattern {
  GreaterDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Greater", /*benefit=*/1, ctx) {}

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
          op, "onnx.Greater decompose expects a ranked tensor result");

    mlir::FailureOr<mlir::Value> initOrFailure =
        createBroadcastEmptyTensor(rewriter, loc, resultType, {b, a});
    if (mlir::failed(initOrFailure))
      return rewriter.notifyMatchFailure(
          op, "Greater: cannot infer dynamic result dimensions from "
              "operands (both inputs must be ranked tensors)");

    return replaceWithBroadcastBinaryHipOp<mlir::hip::LessOp>(
        rewriter, op, "Greater", context, b, a, *initOrFailure, resultType);
  }
};

} // namespace

void populateGreaterConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<GreaterDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
