/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceMax -> hip.reduce_max
/// Reuses the same axes/keepdims/noop_with_empty_axes handling as ReduceSum.
struct ReduceMaxToHip : public mlir::RewritePattern {
  ReduceMaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceMax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceMaxToHip::matchAndRewrite(mlir::Operation *op,
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

  // Statically-known reduced axes (only when axes is NOT a runtime operand).
  // Used both to materialize the axes constant below and to infer the result
  // type when the ONNX importer left the result unranked (see below).
  llvm::SmallVector<int64_t> axesVec;
  bool axesStaticallyKnown =
      resolveReductionAxes(op, data, noopWithEmptyAxes, axesVec);

  // The ONNX importer can leave the ReduceMax result unranked (e.g. Phi's
  // pos_ids_reformat ReduceMax(position_ids) feeding GreaterOrEqual); infer a
  // ranked result type in that case (see inferReduceResultType).
  auto resultTypeOr =
      inferReduceResultType(op, data, axesVec, axesStaticallyKnown, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceMax: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  mlir::FailureOr<mlir::Value> init = createReductionEmptyTensor(
      rewriter, loc, resultType, data, axesVec, axesStaticallyKnown, keepdims);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "ReduceMax result type is incompatible with the reduction shape");

  mlir::Value axesOperand;
  if (op->getNumOperands() > 1) {
    axesOperand = op->getOperand(1);
  } else {
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceMaxOp::create(rewriter, loc, context, data,
                                              axesOperand, *init, keepdimsAttr,
                                              noopWithEmptyAxesAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceMaxConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<ReduceMaxToHip>(ctx);
}

} // namespace hip
} // namespace mlir
