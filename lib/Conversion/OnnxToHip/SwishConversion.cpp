/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SwishConversion.cpp - onnx.Swish -> hip.swish ---------------------===//
//
// Lowers ONNX Swish to one fused HIP operation:
//
//   y = x * sigmoid(alpha * x)
//
// Keeping this fused avoids two or three kernel launches from decomposing the
// ONNX function body into Mul/Sigmoid/Mul. It also preserves the full ONNX
// type set: f16, f32, bf16, and f64.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

struct SwishToHip : public mlir::RewritePattern {
  SwishToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Swish", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.Swish expects 1 operand and 1 result");

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Swish requires ranked tensor input and result");

    mlir::Type elemType = inputType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16() &&
        !elemType.isF64())
      return rewriter.notifyMatchFailure(
          op, "onnx.Swish supports only f16, f32, bf16, and f64");
    if (inputType != resultType)
      return rewriter.notifyMatchFailure(
          op, "onnx.Swish input and result types must match");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    double alpha = 1.0;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
      alpha = attr.getValueAsDouble();

    mlir::Location loc = op->getLoc();
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto swish = mlir::hip::SwishOp::create(rewriter, loc, resultType,
                                            *ctxOrFailure, input, init,
                                            rewriter.getF64FloatAttr(alpha));
    rewriter.replaceOp(op, swish->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateSwishConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<SwishToHip>(ctx);
}

} // namespace hip
} // namespace mlir
