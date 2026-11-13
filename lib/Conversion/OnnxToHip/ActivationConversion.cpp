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
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp =
      mlir::hip::MiopenSoftmaxOp::create(rewriter, loc, context, input, init);
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

/// onnx.Custom(com.microsoft.Gelu) -> hip.gelu with approximate="none".
/// The contrib op has no approximate attribute; it is the erf formula.
struct MicrosoftGeluToHip : public mlir::RewritePattern {
  MicrosoftGeluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MicrosoftGeluToHip::matchAndRewrite(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "Gelu")
    return rewriter.notifyMatchFailure(op, "not a Gelu operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op,
                                       "domain must be com.microsoft for Gelu");

  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "com.microsoft Gelu expects 1 input and 1 result");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(
        op, "com.microsoft Gelu expects ranked output");

  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
  auto hipOp = mlir::hip::GeluOp::create(rewriter, loc, context, input, init,
                                         rewriter.getStringAttr("none"));
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateActivationConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<SoftmaxToHip, SigmoidToHip, TanhToHip, SoftplusToHip, GeluToHip,
               MicrosoftGeluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
