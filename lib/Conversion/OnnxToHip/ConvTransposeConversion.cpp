/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ConvTranspose -> hip.conv_transpose
///
/// Before:
///   %y = "onnx.ConvTranspose"(%x, %w, %b)
///       {kernel_shape = [3, 3], strides = [2, 2]}
///       : (tensor<1x1x3x3xf32>, tensor<1x2x3x3xf32>, tensor<2xf32>)
///          -> tensor<1x2x7x7xf32>
///
/// After:
///   %init = tensor.empty() : tensor<1x2x7x7xf32>
///   %y = hip.conv_transpose(%ctx) ins(%x, %w, %b : ...) outs(%init : ...)
///          {kernel_shape = [3, 3], strides = [2, 2], pads = [0, 0, 0, 0],
///           dilations = [1, 1], output_padding = [0, 0], group = 1}
struct ConvTransposeToHip : public mlir::RewritePattern {
  ConvTransposeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ConvTranspose", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ConvTransposeToHip::matchAndRewrite(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value weights = op->getOperand(1);

  // ONNX ConvTranspose has 2-3 operands; bias (operand 2) is optional and
  // may arrive as onnx.NoValue (NoneType).
  bool hasBias = op->getNumOperands() > 2 &&
                 !mlir::isa<mlir::NoneType>(op->getOperand(2).getType());
  mlir::Value bias = hasBias ? op->getOperand(2) : nullptr;

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto weightsType = mlir::dyn_cast<mlir::RankedTensorType>(weights.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !weightsType || !resultType)
    return rewriter.notifyMatchFailure(
        op, "ConvTranspose requires ranked input, weights, and result");
  if (inputType.getRank() != 4 || weightsType.getRank() != 4 ||
      resultType.getRank() != 4)
    return rewriter.notifyMatchFailure(
        op, "ConvTranspose currently supports rank-4 NCHW tensors only");
  if (auto autoPad = op->getAttrOfType<mlir::StringAttr>("auto_pad"))
    if (autoPad.getValue() != "NOTSET")
      return rewriter.notifyMatchFailure(
          op, "ConvTranspose auto_pad modes are not supported; use explicit "
              "pads");
  if (op->hasAttr("output_shape"))
    return rewriter.notifyMatchFailure(
        op, "ConvTranspose output_shape is not supported by the current "
            "runtime ABI");

  // Weight layout for ConvTranspose is [C, M/group, kH, kW] (input channels
  // first), so spatial kernel dims are at indices 2..N of the weight tensor.
  int64_t numSpatial = weightsType.getRank() - 2;

  llvm::SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape")) {
    for (auto a : attr)
      kernelShape.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Infer kernel_shape from the weight tensor's trailing spatial dims.
    for (int64_t i : llvm::seq<int64_t>(numSpatial)) {
      int64_t kernelDim = weightsType.getDimSize(2 + i);
      if (mlir::ShapedType::isDynamic(kernelDim))
        return rewriter.notifyMatchFailure(
            op, "dynamic weight kernel dimensions require kernel_shape");
      kernelShape.push_back(kernelDim);
    }
  }

  llvm::SmallVector<int64_t> strides;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("strides")) {
    for (auto a : attr)
      strides.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    strides.assign(numSpatial, 1);
  }

  llvm::SmallVector<int64_t> pads;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
    for (auto a : attr)
      pads.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    pads.assign(numSpatial * 2, 0);
  }

  llvm::SmallVector<int64_t> dilations;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("dilations")) {
    for (auto a : attr)
      dilations.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    dilations.assign(numSpatial, 1);
  }

  llvm::SmallVector<int64_t> outputPadding;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("output_padding")) {
    for (auto a : attr)
      outputPadding.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    outputPadding.assign(numSpatial, 0);
  }

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();

  mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> resultShape =
      mlir::hip::reifyConvTransposeResultShape(
          rewriter, loc, input, weights, kernelShape, strides, pads, dilations,
          outputPadding, group, [&]() { return op->emitError(); });
  if (mlir::failed(resultShape))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
      rewriter, loc, resultType, *resultShape);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "ConvTranspose result type is incompatible with the inferred "
            "shape");

  auto kernelShapeAttr = rewriter.getI64ArrayAttr(kernelShape);
  auto stridesAttr = rewriter.getI64ArrayAttr(strides);
  auto padsAttr = rewriter.getI64ArrayAttr(pads);
  auto dilationsAttr = rewriter.getI64ArrayAttr(dilations);
  auto outputPaddingAttr = rewriter.getI64ArrayAttr(outputPadding);
  auto groupAttr = rewriter.getI64IntegerAttr(group);

  llvm::SmallVector<mlir::Value> operands = {context, input, weights};
  if (bias)
    operands.push_back(bias);
  operands.push_back(*init);

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr));
  attrs.push_back(rewriter.getNamedAttr("strides", stridesAttr));
  attrs.push_back(rewriter.getNamedAttr("pads", padsAttr));
  attrs.push_back(rewriter.getNamedAttr("dilations", dilationsAttr));
  attrs.push_back(rewriter.getNamedAttr("output_padding", outputPaddingAttr));
  attrs.push_back(rewriter.getNamedAttr("group", groupAttr));

  auto hipOp = mlir::hip::ConvTransposeOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, operands, attrs);

  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

} // namespace

void populateConvTransposeConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {
  patterns.add<ConvTransposeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
