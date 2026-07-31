/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceL2 -> hip.reduce_l2
///
/// Direct, dim-tolerant conversion: sqrt(sum(x^2)) along the reduced axes
/// happens inside the runtime kernel, so no static reduce axis is required
/// at runtime. Compile-time shape inference still needs ranked input/output
/// or statically-known axes to build the DPS init tensor.
struct ReduceL2ToHip : public mlir::RewritePattern {
  ReduceL2ToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceL2", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceL2ToHip::matchAndRewrite(mlir::Operation *op,
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

  auto resultTypeOr = inferReduceResultType(op, data, reducedAxes, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(
        op, "ReduceL2: cannot infer unranked result (need ranked input and "
            "static axes)");
  mlir::RankedTensorType resultType = *resultTypeOr;

  mlir::FailureOr<mlir::Value> init = createReductionEmptyTensor(
      rewriter, loc, resultType, data, reducedAxes, keepdims);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "ReduceL2 result type is incompatible with the reduction shape");

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp =
      mlir::hip::ReduceL2Op::create(rewriter, loc, context, data, axesOperand,
                                    *init, keepdimsAttr, noopWithEmptyAxesAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateReduceL2ConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<ReduceL2ToHip>(ctx);
}

} // namespace hip
} // namespace mlir
