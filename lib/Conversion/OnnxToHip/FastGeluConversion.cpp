/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static mlir::Value getOptionalOperand(mlir::Operation *op, size_t idx) {
  if (idx >= op->getNumOperands())
    return nullptr;
  mlir::Value val = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(val.getType()))
    return nullptr;
  return val;
}

/// Resolve (data, bias) when an optional 1D bias operand is present.
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

/// onnx.Custom(com.microsoft.FastGelu) -> hip.fast_gelu.
struct FastGeluToHip : public mlir::RewritePattern {
  FastGeluToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
FastGeluToHip::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "FastGelu")
    return rewriter.notifyMatchFailure(op, "not a FastGelu operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for FastGelu");

  if (op->getNumOperands() < 1 || op->getNumOperands() > 2)
    return rewriter.notifyMatchFailure(op, "FastGelu expects 1 or 2 inputs");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value bias = getOptionalOperand(op, 1);
  if (bias) {
    auto [data, resolvedBias] = resolveDataAndBias(input, bias);
    input = data;
    bias = resolvedBias;

    auto biasType = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
    if (biasType && biasType.hasStaticShape() && biasType.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "bias must be a 1D tensor");
  }

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "FastGelu expects ranked output");

  auto init = createSameShapeEmptyTensor(rewriter, loc, resultType, input);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "FastGelu result type must match the input shape");

  llvm::SmallVector<mlir::Value> operands = {context, input};
  if (bias)
    operands.push_back(bias);
  operands.push_back(*init);

  auto hipOp = mlir::hip::FastGeluOp::create(rewriter, loc, operands);
  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

} // namespace

void populateFastGeluConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<FastGeluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
