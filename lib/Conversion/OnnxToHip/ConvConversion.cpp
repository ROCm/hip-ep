/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Conv -> hip.conv
struct ConvToHip : public mlir::RewritePattern {
  ConvToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Conv", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ConvToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value weights = op->getOperand(1);

  // ONNX Conv always has 3 operands, but bias can be onnx.NoValue (NoneType)
  bool hasBias = op->getNumOperands() > 2 &&
                 !mlir::isa<mlir::NoneType>(op->getOperand(2).getType());
  mlir::Value bias = hasBias ? op->getOperand(2) : nullptr;

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Extract attributes from onnx.Conv
  llvm::SmallVector<int64_t> kernelShape;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("kernel_shape")) {
    for (auto a : attr)
      kernelShape.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  }

  llvm::SmallVector<int64_t> strides;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("strides")) {
    for (auto a : attr)
      strides.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default strides = 1 for each spatial dimension
    strides.assign(kernelShape.size(), 1);
  }

  llvm::SmallVector<int64_t> pads;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("pads")) {
    for (auto a : attr)
      pads.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default pads = 0
    pads.assign(kernelShape.size() * 2, 0);
  }

  llvm::SmallVector<int64_t> dilations;
  if (auto attr = op->getAttrOfType<mlir::ArrayAttr>("dilations")) {
    for (auto a : attr)
      dilations.push_back(
          mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
  } else {
    // Default dilations = 1
    dilations.assign(kernelShape.size(), 1);
  }

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();

  // Create output tensor
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimIdx))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx));
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  // Build attributes for hip.conv
  auto kernelShapeAttr = rewriter.getI64ArrayAttr(kernelShape);
  auto stridesAttr = rewriter.getI64ArrayAttr(strides);
  auto padsAttr = rewriter.getI64ArrayAttr(pads);
  auto dilationsAttr = rewriter.getI64ArrayAttr(dilations);
  auto groupAttr = rewriter.getI64IntegerAttr(group);

  // Build operands vector: context, input, weights, [bias], init
  llvm::SmallVector<mlir::Value> operands = {context, input, weights};
  if (bias)
    operands.push_back(bias);
  operands.push_back(init);

  // Build attributes
  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("kernel_shape", kernelShapeAttr));
  attrs.push_back(rewriter.getNamedAttr("strides", stridesAttr));
  attrs.push_back(rewriter.getNamedAttr("pads", padsAttr));
  attrs.push_back(rewriter.getNamedAttr("dilations", dilationsAttr));
  attrs.push_back(rewriter.getNamedAttr("group", groupAttr));

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp = mlir::hip::ConvOp::create(rewriter, loc, operands, attrs);

  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

} // namespace

void populateConvConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<ConvToHip>(ctx);
}

} // namespace hip
} // namespace mlir
