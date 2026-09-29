/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// onnx.Custom(GroupNorm, com.microsoft) -> hip.group_norm
//
//   y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
//
// Mean and variance are per (N, group). activation 1 applies SiLU after the
// affine transform. channels_last 0 is NCHW; 1 is NHWC.

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct GroupNormToHip : public mlir::RewritePattern {
  GroupNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
GroupNormToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "GroupNorm")
    return rewriter.notifyMatchFailure(op, "not a GroupNorm operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for GroupNorm");

  if (op->getNumOperands() != 3 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "GroupNorm expects 3 operands (X, gamma, beta) and 1 result");

  auto groupsAttr = op->getAttrOfType<mlir::IntegerAttr>("groups");
  if (!groupsAttr)
    return rewriter.notifyMatchFailure(op, "GroupNorm requires groups");
  int64_t groups = groupsAttr.getValue().getSExtValue();
  if (groups <= 0)
    return rewriter.notifyMatchFailure(op, "GroupNorm groups must be positive");

  auto activationAttr = op->getAttrOfType<mlir::IntegerAttr>("activation");
  if (!activationAttr)
    return rewriter.notifyMatchFailure(op, "GroupNorm requires activation");
  int64_t activation = activationAttr.getValue().getSExtValue();
  if (activation != 0 && activation != 1)
    return rewriter.notifyMatchFailure(
        op, "GroupNorm activation must be 0 (none) or 1 (SiLU)");

  int64_t channelsLast = 1;
  if (auto channelsLastAttr =
          op->getAttrOfType<mlir::IntegerAttr>("channels_last"))
    channelsLast = channelsLastAttr.getValue().getSExtValue();
  if (channelsLast != 0 && channelsLast != 1)
    return rewriter.notifyMatchFailure(
        op, "GroupNorm channels_last must be 0 (NCHW) or 1 (NHWC)");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);
  mlir::Value bias = op->getOperand(2);

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType || inputType.getRank() < 3)
    return rewriter.notifyMatchFailure(
        op, "GroupNorm requires ranked input of rank >= 3");

  auto scaleType = mlir::dyn_cast<mlir::RankedTensorType>(scale.getType());
  auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
  if (!scaleType || scaleType.getRank() != 1 || !biasType ||
      biasType.getRank() != 1)
    return rewriter.notifyMatchFailure(
        op, "GroupNorm gamma and beta must be 1-D of length C");

  int64_t channelAxis = channelsLast ? inputType.getRank() - 1 : 1;
  if (!inputType.isDynamicDim(channelAxis)) {
    int64_t channels = inputType.getDimSize(channelAxis);
    if (channels % groups != 0)
      return rewriter.notifyMatchFailure(
          op, "GroupNorm channel extent is not divisible by groups");
    if (scaleType.hasStaticShape() && scaleType.getDimSize(0) != channels)
      return rewriter.notifyMatchFailure(
          op, "GroupNorm gamma length must equal the channel extent");
    if (biasType.hasStaticShape() && biasType.getDimSize(0) != channels)
      return rewriter.notifyMatchFailure(
          op, "GroupNorm beta length must equal the channel extent");
  }

  llvm::APFloat epsValue(1.0e-05f);
  if (auto epsAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon"))
    epsValue = epsAttr.getValue();

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, input);

  auto hipOp = mlir::hip::GroupNormOp::create(
      rewriter, loc, context, input, scale, bias, outputInit,
      rewriter.getI64IntegerAttr(groups),
      rewriter.getI64IntegerAttr(activation),
      rewriter.getI64IntegerAttr(channelsLast),
      rewriter.getF32FloatAttr(epsValue.convertToFloat()));
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateGroupNormConversionPatterns(RewritePatternSet &patterns,
                                         MLIRContext *ctx) {
  patterns.add<GroupNormToHip>(ctx);
}

} // namespace hip
} // namespace mlir
