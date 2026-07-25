/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceSum -> hip.reduce_sum
struct ReduceSumToHip : public mlir::RewritePattern {
  ReduceSumToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceSum", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceSumToHip::matchAndRewrite(mlir::Operation *op,
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

  // Statically-known reduced axes (only when axes is NOT a runtime operand).
  // Used to materialize the axes constant and to infer an unranked result.
  bool axesStaticallyKnown = op->getNumOperands() <= 1;
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  llvm::SmallVector<int64_t> axesVec;
  if (axesStaticallyKnown) {
    if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
      for (auto a : axesAttr)
        axesVec.push_back(
            mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
    } else if (noopWithEmptyAxes == 0 && inputType) {
      // Default: reduce all axes (when noop_with_empty_axes is 0)
      for (int64_t i : llvm::seq<int64_t>(inputType.getRank()))
        axesVec.push_back(i);
    }
    // noop_with_empty_axes == 1 with no axes -> axesVec stays empty (identity).
  }

  // Resolve the result type (infer if the importer left it unranked).
  auto resultTypeOr =
      inferReduceResultType(op, data, axesVec, axesStaticallyKnown, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceSum: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, data);

  // axes is always required in HIP dialect; create empty tensor<0xi64> when not
  // provided
  mlir::Value axesOperand;
  if (op->getNumOperands() > 1) {
    // Axes provided as operand (opset 13+)
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

  // Create hip.reduce_sum operation (axes always provided, may be empty)
  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp =
      mlir::hip::ReduceSumOp::create(rewriter, loc, context, data, axesOperand,
                                     init, keepdimsAttr, noopWithEmptyAxesAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceSumConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ReduceSumToHip>(ctx);
}

} // namespace hip
} // namespace mlir
