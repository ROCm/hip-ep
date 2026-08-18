/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Softmax -> hip.miopen.softmax
struct SoftmaxToHip : public mlir::RewritePattern {
  SoftmaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Softmax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SoftmaxToHip::matchAndRewrite(mlir::Operation *op,
                              mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected one input and one result");

  mlir::Value input = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !resultType)
    return rewriter.notifyMatchFailure(
        op, "Softmax input and result must be ranked tensors");
  int64_t rank = inputType.getRank();
  if (rank == 0 || resultType.getRank() != rank)
    return rewriter.notifyMatchFailure(
        op, "Softmax requires equal positive input and result ranks");
  if (inputType.getElementType() != resultType.getElementType())
    return rewriter.notifyMatchFailure(
        op, "Softmax input and result element types must match");
  for (int64_t axis : llvm::seq<int64_t>(0, rank)) {
    int64_t inputExtent = inputType.getDimSize(axis);
    int64_t resultExtent = resultType.getDimSize(axis);
    if (!mlir::ShapedType::isDynamic(inputExtent) &&
        !mlir::ShapedType::isDynamic(resultExtent) &&
        inputExtent != resultExtent)
      return rewriter.notifyMatchFailure(
          op, "Softmax input and result shapes must match");
  }

  int64_t axis = -1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getInt();
  if (axis < 0)
    axis += rank;
  if (axis != rank - 1)
    return rewriter.notifyMatchFailure(
        op, "only last-dimension Softmax is supported");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();

  mlir::Location loc = op->getLoc();
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::MiopenSoftmaxOp::create(rewriter, loc, *ctxOrFailure,
                                                  input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Sigmoid -> hip.sigmoid
struct SigmoidToHip : public mlir::RewritePattern {
  SigmoidToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Sigmoid", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SigmoidToHip::matchAndRewrite(mlir::Operation *op,
                              mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp =
      mlir::hip::SigmoidOp::create(rewriter, loc, context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Tanh -> hip.tanh
struct TanhToHip : public mlir::RewritePattern {
  TanhToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Tanh", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
TanhToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(op,
                                       "Tanh expects a ranked tensor result");
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::TanhOp::create(rewriter, loc, context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Softplus -> hip.softplus
struct SoftplusToHip : public mlir::RewritePattern {
  SoftplusToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Softplus", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SoftplusToHip::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp =
      mlir::hip::SoftplusOp::create(rewriter, loc, context, input, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Gelu -> hip.gelu
struct GeluToHip : public mlir::RewritePattern {
  GeluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gelu", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
GeluToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Extract and validate approximate attribute from ONNX op (default to "none")
  mlir::StringAttr approximateAttr = rewriter.getStringAttr("none");
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("approximate")) {
    llvm::StringRef approxValue = attr.getValue();
    // Only "none" and "tanh" are valid per ONNX Gelu spec
    if (approxValue != "none" && approxValue != "tanh") {
      return rewriter.notifyMatchFailure(op, "invalid approximate attribute '" +
                                                 approxValue.str() +
                                                 "', must be 'none' or 'tanh'");
    }
    approximateAttr = attr;
  }

  auto hipOp = mlir::hip::GeluOp::create(rewriter, loc, context, input, init,
                                         approximateAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateActivationConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<SoftmaxToHip, SigmoidToHip, TanhToHip, SoftplusToHip, GeluToHip>(
      ctx);
}

} // namespace hip
} // namespace mlir
