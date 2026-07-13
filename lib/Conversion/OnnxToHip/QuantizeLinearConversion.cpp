/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static mlir::Type onnxDataTypeToMlir(mlir::MLIRContext *ctx, int64_t onnxDt) {
  switch (onnxDt) {
  case 2:
    return mlir::IntegerType::get(ctx, 8, mlir::IntegerType::Unsigned);
  case 3:
    return mlir::IntegerType::get(ctx, 8);
  case 4:
    return mlir::IntegerType::get(ctx, 16, mlir::IntegerType::Unsigned);
  case 5:
    return mlir::IntegerType::get(ctx, 16);
  default:
    return {};
  }
}

//===----------------------------------------------------------------------===//
// onnx.QuantizeLinear -> hip.quantize_linear
//===----------------------------------------------------------------------===//
//
// Before:
//   %y = "onnx.QuantizeLinear"(%x, %scale, %zp)
//       {axis = 1 : si64, block_size = 0 : si64, output_dtype = 0 : si64,
//        precision = 0 : si64, saturate = 1 : si64}
//       : (tensor<4x8xf32>, tensor<8xf32>, tensor<8xui8>) -> tensor<4x8xui8>
//
// After:
//   %init = tensor.empty() : tensor<4x8xui8>
//   %y = hip.quantize_linear(%ctx)
//       ins(%x, %scale : tensor<4x8xf32>, tensor<8xf32>)
//       zero_points(%zp : tensor<8xui8>)
//       outs(%init : tensor<4x8xui8>)
//       {axis = 1, block_size = 0, output_dtype = 0, precision = 0, saturate = 1}
//       : tensor<4x8xui8>

struct QuantizeLinearToHip : public mlir::RewritePattern {
  QuantizeLinearToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.QuantizeLinear", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult QuantizeLinearToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  if (op->getNumOperands() < 2)
    return rewriter.notifyMatchFailure(op, "expected at least x and y_scale");

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
  auto xElem = xType.getElementType();
  if (!xElem.isF32() && !xElem.isF16() && !xElem.isBF16() &&
      !xElem.isSignlessInteger(32))
    return rewriter.notifyMatchFailure(op, "unsupported quantize input element type");

  int64_t axis = 1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();

  int64_t blockSize = 0;
  if (auto blockAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size"))
    blockSize = blockAttr.getSInt();

  int64_t outputDtypeAttr = 0;
  if (auto outDtAttr = op->getAttrOfType<mlir::IntegerAttr>("output_dtype"))
    outputDtypeAttr = outDtAttr.getSInt();

  int64_t precisionAttr = 0;
  if (auto precAttr = op->getAttrOfType<mlir::IntegerAttr>("precision"))
    precisionAttr = precAttr.getSInt();

  int64_t saturateAttr = 1;
  if (auto satAttr = op->getAttrOfType<mlir::IntegerAttr>("saturate"))
    saturateAttr = satAttr.getSInt();

  mlir::Type outElemType;
  if (outputDtypeAttr != 0) {
    outElemType = onnxDataTypeToMlir(rewriter.getContext(), outputDtypeAttr);
    if (!outElemType)
      return rewriter.notifyMatchFailure(op, "unsupported output_dtype attribute");
  } else if (zeroPoint) {
    outElemType =
        mlir::cast<mlir::RankedTensorType>(zeroPoint.getType()).getElementType();
  } else {
    outElemType =
        mlir::IntegerType::get(rewriter.getContext(), 8, mlir::IntegerType::Unsigned);
  }

  if (!outElemType.isSignlessInteger(8) && !outElemType.isUnsignedInteger(8) &&
      !outElemType.isSignlessInteger(16) && !outElemType.isUnsignedInteger(16))
    return rewriter.notifyMatchFailure(op, "unsupported quantize output element type");

  auto resultType = mlir::RankedTensorType::get(xType.getShape(), outElemType);
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, x);

  auto axisAttrVal = rewriter.getI64IntegerAttr(axis);
  auto blockSizeAttrVal = rewriter.getI64IntegerAttr(blockSize);
  auto outputDtypeAttrVal = rewriter.getI64IntegerAttr(outputDtypeAttr);
  auto precisionAttrVal = rewriter.getI64IntegerAttr(precisionAttr);
  auto saturateAttrVal = rewriter.getI64IntegerAttr(saturateAttr);

  auto hipOp = mlir::hip::QuantizeLinearOp::create(
      rewriter, loc, resultType, context, x, scale, zeroPoint, init,
      axisAttrVal, blockSizeAttrVal, outputDtypeAttrVal, precisionAttrVal,
      saturateAttrVal);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateQuantizeLinearConversionPatterns(RewritePatternSet &patterns,
                                              MLIRContext *ctx) {
  patterns.add<QuantizeLinearToHip>(ctx);
}

} // namespace hip
} // namespace mlir
