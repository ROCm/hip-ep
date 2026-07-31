/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceMin -> hip.reduce_min
/// Reuses the same axes/keepdims/noop_with_empty_axes handling as ReduceMax.
struct ReduceMinToHip : public mlir::RewritePattern {
  ReduceMinToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMin", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceMinToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);

  int64_t noopWithEmptyAxes = 0;
  if (auto noopAttr =
          op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes")) {
    noopWithEmptyAxes = noopAttr.getSInt();
  }

  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims")) {
    keepdims = keepdimsAttr.getSInt();
  }

  llvm::SmallVector<int64_t> axesStorage;
  std::optional<llvm::ArrayRef<int64_t>> reducedAxes =
      resolveReductionAxes(op, data, noopWithEmptyAxes, axesStorage);

  auto resultTypeOr = inferReduceResultType(op, data, reducedAxes, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceMin: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  mlir::FailureOr<mlir::Value> init = createReductionEmptyTensor(
      rewriter, loc, resultType, data, reducedAxes, keepdims);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "ReduceMin result type is incompatible with the reduction shape");

  mlir::Value axesOperand;
  if (op->getNumOperands() > 1 &&
      !mlir::isa<mlir::NoneType>(op->getOperand(1).getType())) {
    axesOperand = op->getOperand(1);
  } else {
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesStorage.size())}, rewriter.getI64Type());
    auto axesAttr = mlir::DenseIntElementsAttr::get(
        axesType, llvm::ArrayRef<int64_t>(axesStorage));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceMinOp::create(rewriter, loc, context, data,
                                              axesOperand, *init, keepdimsAttr,
                                              noopWithEmptyAxesAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceMinConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ReduceMinToHip>(ctx);
}

} // namespace hip
} // namespace mlir
