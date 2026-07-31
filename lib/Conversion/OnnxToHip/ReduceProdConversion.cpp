/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceProd -> hip.reduce_prod
///
/// Mirrors the existing ReduceSum/ReduceMax conversion: lifts an optional
/// `axes` operand or attribute into a required tensor operand, threads
/// `keepdims` and `noop_with_empty_axes` through, and shares lowering with
/// other reduction ops via the unified ReduceLowering template.
struct ReduceProdToHip : public mlir::RewritePattern {
  ReduceProdToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceProd", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceProdToHip::matchAndRewrite(mlir::Operation *op,
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
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = keepdimsAttr.getSInt();

  // Reduced axes, resolved from the attribute or a compile-time-constant
  // operand, so the destination shape can map each output dimension back to the
  // input dimension it comes from.
  llvm::SmallVector<int64_t> axesStorage;
  std::optional<llvm::ArrayRef<int64_t>> reducedAxes =
      resolveReductionAxes(op, data, noopWithEmptyAxes, axesStorage);

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

  // Resolve the result type (infer if the importer left it unranked).
  auto resultTypeOr = inferReduceResultType(op, data, reducedAxes, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceProd: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  mlir::FailureOr<mlir::Value> init = createReductionEmptyTensor(
      rewriter, loc, resultType, data, reducedAxes, keepdims);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "ReduceProd result type is incompatible with the reduction shape");

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceProdOp::create(
      rewriter, loc, context, data, axesOperand, *init, keepdimsAttr, noopAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceProdConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<ReduceProdToHip>(ctx);
}

} // namespace hip
} // namespace mlir
