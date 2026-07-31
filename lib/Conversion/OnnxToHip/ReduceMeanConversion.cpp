/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceMean -> hip.reduce_mean
///
/// Direct, dim-tolerant conversion: the division by the reduced-element count
/// happens inside the runtime kernel, so (unlike the retired
/// ReduceMeanToReduceSumDiv decomposition) no static reduce axis is required.
/// Result dims are refined later by --hip-infer-shapes through the upstream
/// MLIR ReifyRankedShapedTypeOpInterface / InferTypeOpInterface that
/// Hip_DpsOp_Reduction implements, so no ONNX-level shape refinement is needed.
///
/// `axes` is always materialized as an operand for the DPS op: opset >= 18
/// already supplies it as an operand; for older opsets (or the all-axes
/// default) it is built here from the `axes` attribute into an i64 constant.
///
/// Before (opset < 18, axes as attribute):
///   %y = "onnx.ReduceMean"(%x) {axes = [1], keepdims = 1}
///          : (tensor<?x4096xf16>) -> tensor<?x1xf16>
/// After:
///   %axes = arith.constant dense<[1]> : tensor<1xi64>
///   %init = tensor.empty(%n) : tensor<?x1xf16>
///   %y    = hip.reduce_mean(%ctx) ins(%x, %axes) outs(%init) {keepdims = 1}
struct ReduceMeanToHip : public mlir::RewritePattern {
  ReduceMeanToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMean", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceMeanToHip::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);

  // Extract noop_with_empty_axes attribute (defaults to 0 in ONNX)
  int64_t noopWithEmptyAxes = 0;
  if (auto noopAttr =
          op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes")) {
    noopWithEmptyAxes = noopAttr.getSInt();
  }

  // Extract keepdims attribute (defaults to 1 in ONNX)
  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims")) {
    keepdims = keepdimsAttr.getSInt();
  }

  // Reduced axes, resolved from the attribute or a compile-time-constant
  // operand.
  llvm::SmallVector<int64_t> axesVec;
  bool axesStaticallyKnown =
      resolveReductionAxes(op, data, noopWithEmptyAxes, axesVec);

  // Resolve the result type (infer if the importer left it unranked).
  auto resultTypeOr =
      inferReduceResultType(op, data, axesVec, axesStaticallyKnown, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceMean: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;
  mlir::FailureOr<mlir::Value> init = createReductionEmptyTensor(
      rewriter, loc, resultType, data, axesVec, axesStaticallyKnown, keepdims);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "ReduceMean result type is incompatible with the reduction shape");

  // axes is always required in HIP dialect; create empty tensor<0xi64> when not
  // provided
  mlir::Value axesOperand;
  if (op->getNumOperands() > 1) {
    // Axes provided as operand (opset 18+)
    axesOperand = op->getOperand(1);
  } else {
    // Create constant tensor for axes
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  // Create hip.reduce_mean operation (axes always provided, may be empty)
  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceMeanOp::create(rewriter, loc, context, data,
                                               axesOperand, *init, keepdimsAttr,
                                               noopWithEmptyAxesAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceMeanConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<ReduceMeanToHip>(ctx);
}

} // namespace hip
} // namespace mlir
