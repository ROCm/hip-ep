/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// onnx.QuantizeLinear -> hip.quantize_linear
// y = saturate((x / scale) + zero_point)
// onnx.DequantizeLinear -> hip.dequantize_linear
// y = (x - zero_point) * scale

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

mlir::Value getOptionalOperand(mlir::Operation *op, size_t idx) {
  if (idx >= op->getNumOperands())
    return nullptr;
  mlir::Value val = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(val.getType()))
    return nullptr;
  return val;
}

static int64_t getOnnxIntAttr(mlir::Operation *op, llvm::StringRef name,
  int64_t defaultValue) {
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
    return attr.getValue().getSExtValue();
  return defaultValue;
}

struct QdqOperands {
  mlir::Value context;
  mlir::Value input;
  mlir::Value scale;
  mlir::Value zeroPoint;
  mlir::Value init;
  mlir::RankedTensorType resultType;
};

// The reason for bringing this up is that there might 
// be a future conversion from com.ms.qdq to hip.
mlir::FailureOr<QdqOperands> matchQdqCommon(mlir::Operation *op,
                                            mlir::PatternRewriter &rewriter) {
  size_t numOperands = op->getNumOperands();
  if (numOperands < 2 || numOperands > 3 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected 2-3 inputs and 1 output");

  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "expected ranked output");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "failed to get context argument");

  QdqOperands operands;
  operands.context = *ctxOrFailure;
  operands.input = op->getOperand(0);
  operands.scale = op->getOperand(1);
  operands.zeroPoint = getOptionalOperand(op, 2);
  operands.resultType = resultType;
  operands.init =
      createEmptyTensor(rewriter, op->getLoc(), resultType, operands.input);
  return operands;
}

struct QuantizeLinearToHip : public mlir::RewritePattern {
  QuantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.QuantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto operandsOr = matchQdqCommon(op, rewriter);
    if (mlir::failed(operandsOr))
      return mlir::failure();
    const QdqOperands &in = *operandsOr;
    // read attributes with default values
    auto axisAttr = rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "axis", 1));
    auto blockSizeAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "block_size", 0));
    auto precisionAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "precision", 0));
    auto saturateAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "saturate", 1));

    auto hipOp = mlir::hip::QuantizeLinearOp::create(
        rewriter, op->getLoc(), mlir::TypeRange{in.resultType}, in.context,
        in.input, in.scale, in.zeroPoint, in.init, axisAttr, blockSizeAttr,
        precisionAttr, saturateAttr);
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

struct DequantizeLinearToHip : public mlir::RewritePattern {
  DequantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DequantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto operandsOr = matchQdqCommon(op, rewriter);
    if (mlir::failed(operandsOr))
      return mlir::failure();
    const QdqOperands &in = *operandsOr;

    auto axisAttr = rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "axis", 1));
    auto blockSizeAttr =
        rewriter.getI64IntegerAttr(getOnnxIntAttr(op, "block_size", 0));

    auto hipOp = mlir::hip::DequantizeLinearOp::create(
        rewriter, op->getLoc(), mlir::TypeRange{in.resultType}, in.context,
        in.input, in.scale, in.zeroPoint, in.init, axisAttr, blockSizeAttr);
    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

} // namespace

void populateQdqConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<QuantizeLinearToHip, DequantizeLinearToHip>(ctx);
}

} // namespace hip
} // namespace mlir
