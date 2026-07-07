/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Resolve (data, bias) from the two BiasGelu operands.
///
/// MS spec names A=data and B=1D bias, but some fused graphs may store the
/// operands in the opposite order when one side is rank-1. When both ranks are
/// known at compile time, pick the higher-rank tensor as data and the rank-1
/// tensor as bias; otherwise keep schema order.
static std::pair<mlir::Value, mlir::Value> resolveDataAndBias(mlir::Value op0,
                                                              mlir::Value op1) {
  auto type0 = mlir::dyn_cast<mlir::RankedTensorType>(op0.getType());
  auto type1 = mlir::dyn_cast<mlir::RankedTensorType>(op1.getType());
  if (!type0 || !type1)
    return {op0, op1};

  if (type0.getRank() == 1 && type1.getRank() > 1)
    return {op1, op0};
  return {op0, op1};
}

/// onnx.Custom(com.microsoft.BiasGelu) -> hip.bias_gelu.
///
/// Before:
///   %out = "onnx.Custom"(%data, %bias) {
///     function_name = "BiasGelu", domain_name = "com.microsoft"
///   } : (tensor<1x128x768xf16>, tensor<768xf16>) -> tensor<1x128x768xf16>
///
/// After:
///   %out_init = tensor.empty(...)
///   %out = hip.bias_gelu(%ctx) ins(%data, %bias) outs(%out_init)
struct BiasGeluToHip : public mlir::RewritePattern {
  BiasGeluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
BiasGeluToHip::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "BiasGelu")
    return rewriter.notifyMatchFailure(op, "not a BiasGelu operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for BiasGelu");

  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(op, "BiasGelu expects exactly 2 inputs");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  auto [data, bias] = resolveDataAndBias(op->getOperand(0), op->getOperand(1));

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "BiasGelu expects ranked output");

  auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
  if (biasType && biasType.hasStaticShape() && biasType.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "bias must be a 1D tensor");

  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, data);
  auto hipOp =
      mlir::hip::BiasGeluOp::create(rewriter, loc, context, data, bias, init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateBiasGeluConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<BiasGeluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
