/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// com.microsoft.DequantizeLinear -> hip.ms_dequantize_linear
//===----------------------------------------------------------------------===//
//
// Before:
//   %out = "onnx.Custom"(%input, %scale, %zp)
//       {function_name = "DequantizeLinear", domain_name = "com.microsoft"}
//       : (tensor<...xi8>, tensor<...xf16>, tensor<...xi8>) -> tensor<...xf16>
//
// After:
//   %init = tensor.empty() : tensor<...xf16>
//   %out = hip.ms_dequantize_linear(%ctx)
//       ins(%input, %scale : ...) zero_point(%zp : ...) outs(%init : ...)
//       {axis = 1, input_elem_size = 1, scale_elem_size = 2}

struct MsDequantizeLinearToHip : public mlir::RewritePattern {
  MsDequantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult MsDequantizeLinearToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "DequantizeLinear")
    return rewriter.notifyMatchFailure(op, "not DequantizeLinear");
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op, "not com.microsoft");

  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected 1 result");
  if (op->getNumOperands() < 2)
    return rewriter.notifyMatchFailure(op, "need at least input + scale");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  mlir::Value input = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);
  mlir::Value zeroPoint;
  if (op->getNumOperands() >= 3) {
    mlir::Value v = op->getOperand(2);
    if (v && !mlir::isa<mlir::NoneType>(v.getType()))
      zeroPoint = v;
  }

  // Output type matches scale element type (e.g. fp16 or fp32).
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Derive dynamic sizes from input (output has same spatial shape).
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  llvm::SmallVector<mlir::Value> dynSizes;
  for (auto i : llvm::seq<int64_t>(0, inputType.getRank())) {
    if (resultType.isDynamicDim(static_cast<unsigned>(i)))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, input, i));
  }

  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, resultType.getShape(), resultType.getElementType(),
      dynSizes);

  // Axis attribute (default 1 for com.microsoft DequantizeLinear).
  auto axisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
  int64_t axisVal = axisIntAttr ? axisIntAttr.getSInt() : 1;
  // Per-tensor quant: a scalar (single-element) scale applies one scale to the
  // whole tensor. Force axis out-of-range so the lowering emits n_channels=1.
  // Without this, the default axis=1 treats the sequence dim as channels, which
  // is correct only for M=1 (decode) but corrupts M>1 (prefill).
  if (auto st = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType()))
    if (st.getNumElements() == 1)
      axisVal = -1;
  auto axisAttr = rewriter.getI64IntegerAttr(axisVal);

  // Element sizes in bytes for input and scale.
  int64_t inputElemSize = 1; // default: int8/uint8
  if (auto inputTy = mlir::dyn_cast<mlir::RankedTensorType>(input.getType())) {
    auto et = inputTy.getElementType();
    if (et.isF16() || et.isBF16() || et.isInteger(16))
      inputElemSize = 2;
    else if (et.isF32() || et.isInteger(32))
      inputElemSize = 4;
  }
  int64_t scaleElemSize = 2; // default: fp16
  if (auto scaleTy = mlir::dyn_cast<mlir::RankedTensorType>(scale.getType())) {
    auto et = scaleTy.getElementType();
    if (et.isF32() || et.isInteger(32))
      scaleElemSize = 4;
  }
  auto inputElemSizeAttr = rewriter.getI64IntegerAttr(inputElemSize);
  auto scaleElemSizeAttr = rewriter.getI64IntegerAttr(scaleElemSize);

  auto hipOp = mlir::hip::MsDequantizeLinearOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, context, input, scale,
      zeroPoint, init, axisAttr, inputElemSizeAttr, scaleElemSizeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// com.microsoft.QuantizeLinear -> hip.ms_quantize_linear
//===----------------------------------------------------------------------===//
//
// Before:
//   %out = "onnx.Custom"(%input, %scale, %zp)
//       {function_name = "QuantizeLinear", domain_name = "com.microsoft"}
//       : (tensor<...xf16>, tensor<...xf16>, tensor<...xi8>) -> tensor<...xi8>
//
// After:
//   %init = tensor.empty() : tensor<...xi8>
//   %out = hip.ms_quantize_linear(%ctx)
//       ins(%input, %scale : ...) zero_point(%zp : ...) outs(%init : ...)

struct MsQuantizeLinearToHip : public mlir::RewritePattern {
  MsQuantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult MsQuantizeLinearToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "QuantizeLinear")
    return rewriter.notifyMatchFailure(op, "not QuantizeLinear");
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op, "not com.microsoft");

  if (op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected 1 result");
  if (op->getNumOperands() < 2)
    return rewriter.notifyMatchFailure(op, "need at least input + scale");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  mlir::Value input = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);
  mlir::Value zeroPoint;
  if (op->getNumOperands() >= 3) {
    mlir::Value v = op->getOperand(2);
    if (v && !mlir::isa<mlir::NoneType>(v.getType()))
      zeroPoint = v;
  }

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

  llvm::SmallVector<mlir::Value> dynSizes;
  for (auto i : llvm::seq<int64_t>(0, inputType.getRank())) {
    if (resultType.isDynamicDim(static_cast<unsigned>(i)))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, input, i));
  }

  mlir::Value init = mlir::tensor::EmptyOp::create(
      rewriter, loc, resultType.getShape(), resultType.getElementType(),
      dynSizes);

  auto axisIntAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
  int64_t axisVal = axisIntAttr ? axisIntAttr.getSInt() : 1;
  // Per-tensor quant (scalar scale): force n_channels=1 (see DequantizeLinear).
  if (auto st = mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType()))
    if (st.getNumElements() == 1)
      axisVal = -1;
  auto axisAttr = rewriter.getI64IntegerAttr(axisVal);

  int64_t inputElemSize = 2; // default: fp16
  if (auto inputTy = mlir::dyn_cast<mlir::RankedTensorType>(input.getType())) {
    auto et = inputTy.getElementType();
    if (et.isF32() || et.isInteger(32))
      inputElemSize = 4;
    else if (et.isInteger(8) || et.isUnsignedInteger(8))
      inputElemSize = 1;
  }
  int64_t outputElemSize = 1; // default: int8/uint8
  if (auto et = resultType.getElementType();
      et.isF16() || et.isBF16() || et.isInteger(16))
    outputElemSize = 2;
  else if (et.isF32() || et.isInteger(32))
    outputElemSize = 4;

  auto inputElemSizeAttr = rewriter.getI64IntegerAttr(inputElemSize);
  auto outputElemSizeAttr = rewriter.getI64IntegerAttr(outputElemSize);

  auto hipOp = mlir::hip::MsQuantizeLinearOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, context, input, scale,
      zeroPoint, init, axisAttr, inputElemSizeAttr, outputElemSizeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateMsDequantizeLinearConversionPatterns(RewritePatternSet &patterns,
                                                  MLIRContext *ctx) {
  patterns.add<MsDequantizeLinearToHip>(ctx);
  patterns.add<MsQuantizeLinearToHip>(ctx);
}

} // namespace hip
} // namespace mlir
