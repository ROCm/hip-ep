/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// Map ONNX TensorProto.DataType enum to an MLIR element type for dequant output.
static mlir::Type onnxDataTypeToMlir(mlir::MLIRContext *ctx, int64_t onnxDt) {
  switch (onnxDt) {
  case 1:
    return mlir::Float32Type::get(ctx);
  case 10:
    return mlir::Float16Type::get(ctx);
  case 16:
    return mlir::BFloat16Type::get(ctx);
  default:
    return {};
  }
}

//===----------------------------------------------------------------------===//
// onnx.DequantizeLinear -> hip.dequantize_linear
//===----------------------------------------------------------------------===//
//
// Before:
//   %y = "onnx.DequantizeLinear"(%x, %scale, %zp)
//       {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64}
//       : (tensor<4x8xi8>, tensor<8xf32>, tensor<8xui8>) -> tensor<4x8xf32>
//
// After:
//   %init = tensor.empty() : tensor<4x8xf32>
//   %y = hip.dequantize_linear(%ctx)
//       ins(%x, %scale : tensor<4x8xi8>, tensor<8xf32>)
//       zero_points(%zp : tensor<8xui8>)
//       outs(%init : tensor<4x8xf32>)
//       {axis = 1, block_size = 0, output_dtype = 0}
//       : tensor<4x8xf32>

struct DequantizeLinearToHip : public mlir::RewritePattern {
  DequantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.DequantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult DequantizeLinearToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() < 2)
    return rewriter.notifyMatchFailure(op, "expected at least x and x_scale");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value x = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);

  mlir::Value zeroPoint;
  if (op->getNumOperands() >= 3) {
    mlir::Value v = op->getOperand(2);
    if (v && !mlir::isa<mlir::NoneType>(v.getType()))
      zeroPoint = v;
  }

  auto xType = mlir::cast<mlir::RankedTensorType>(x.getType());
  auto scaleType = mlir::cast<mlir::RankedTensorType>(scale.getType());

  int64_t axis = 1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();

  int64_t blockSize = 0;
  if (auto blockAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size"))
    blockSize = blockAttr.getSInt();

  int64_t outputDtypeAttr = 0;
  if (auto outDtAttr = op->getAttrOfType<mlir::IntegerAttr>("output_dtype"))
    outputDtypeAttr = outDtAttr.getSInt();

  mlir::Type outElemType;
  if (outputDtypeAttr != 0) {
    outElemType = onnxDataTypeToMlir(rewriter.getContext(), outputDtypeAttr);
    if (!outElemType)
      return rewriter.notifyMatchFailure(op, "unsupported output_dtype attribute");
  } else {
    outElemType = scaleType.getElementType();
  }

  if (!outElemType.isF32() && !outElemType.isF16() && !outElemType.isBF16())
    return rewriter.notifyMatchFailure(op, "unsupported dequant output element type");

  auto xElem = xType.getElementType();
  if (!xElem.isSignlessInteger(8) && !xElem.isUnsignedInteger(8) &&
      !xElem.isSignlessInteger(32))
    return rewriter.notifyMatchFailure(op, "unsupported dequant input element type");

  auto resultType = mlir::RankedTensorType::get(xType.getShape(), outElemType);
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);

  auto axisAttr = rewriter.getI64IntegerAttr(axis);
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSize);
  auto outputDtypeAttrVal = rewriter.getI64IntegerAttr(outputDtypeAttr);

  auto hipOp = mlir::hip::DequantizeLinearOp::create(
      rewriter, loc, resultType, context, x, scale, zeroPoint, init, axisAttr,
      blockSizeAttr, outputDtypeAttrVal);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateDequantizeLinearConversionPatterns(RewritePatternSet &patterns,
                                                MLIRContext *ctx) {
  patterns.add<DequantizeLinearToHip>(ctx);
}

} // namespace hip
} // namespace mlir
