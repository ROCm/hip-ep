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

  // Create the output (DPS destination) tensor. For each DYNAMIC output dim we
  // must materialize its runtime extent. The naive "output dim == input dim"
  // copy is only correct for the batch axis and for stride-1 same-padding
  // convs; a strided / downsampling conv has spatial extents that are a
  // non-identity floor-division of the input extent, so we emit that
  // arithmetic here. This is what lets the downstream shape program
  // (BuildShapeFunctionPass) fold the true output extent and the EP size the
  // ORT output buffer correctly for dynamic-spatial vision-style models.
  //
  // ONNX Conv output spatial formula (explicit pads; auto_pad NOTSET — the
  // SAME_*/VALID auto_pad modes are not handled here, matching the rest of
  // this converter which reads only the explicit `pads` attribute):
  //   out[s] = floor((in[s] + pad_begin[s] + pad_end[s]
  //                   - dilation[s]*(kernel[s]-1) - 1) / stride[s]) + 1
  // The pad/dilation/kernel/stride terms are all compile-time constants, so
  // each spatial dim collapses to a single `addi/divsi/addi` chain over
  // `tensor.dim(input, dimIdx)` (divsi == floor for the non-negative numerator
  // any valid conv produces). divsi/addi are speculatable with the constant
  // divisor and are hoistable by hip-hoist-alloc-size-arith / hip-pool-allocs.
  //
  // Before (strides=[2,2], pads=[1,1,1,1], k=3x3, input <1x3x?x?>):
  //   %h = tensor.dim %in, 2
  //   %w = tensor.dim %in, 3
  //   %o = tensor.empty(%h, %w) : tensor<1x16x?x?xf32>   // WRONG: out==in
  // After:
  //   %h  = tensor.dim %in, 2
  //   %h1 = arith.addi %h, -1                            // +
  //   (pad_lo+pad_hi-dil*(k-1)-1) %h2 = arith.divsi %h1, 2 // / stride %ho =
  //   arith.addi %h2, 1 (… same for W …) %o  = tensor.empty(%ho, %wo) :
  //   tensor<1x16x?x?xf32>
  const int64_t rank = resultType.getRank();
  const int64_t numSpatial = static_cast<int64_t>(kernelShape.size());
  // Leading (non-spatial) output dims: N (batch) and C (channels).
  const int64_t numLeading = rank - numSpatial;
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dimIdx : llvm::seq<int64_t>(rank)) {
    if (!resultType.isDynamicDim(dimIdx))
      continue;
    mlir::Value sz;
    if (dimIdx < numLeading) {
      // Batch (dim 0) passes through from the input; output channels (dim 1)
      // equal the weight tensor's leading dim (M). Both are rarely dynamic
      // (channels are a static architecture constant), handled for safety.
      sz = (dimIdx == 0)
               ? mlir::tensor::DimOp::create(rewriter, loc, input, 0)
               : mlir::tensor::DimOp::create(rewriter, loc, weights, 0);
    } else {
      const int64_t s = dimIdx - numLeading;
      const int64_t padLo = pads[s];
      const int64_t padHi = pads[numSpatial + s];
      const int64_t cConst =
          padLo + padHi - dilations[s] * (kernelShape[s] - 1) - 1;
      mlir::Value inDim =
          mlir::tensor::DimOp::create(rewriter, loc, input, dimIdx);
      mlir::Value cVal =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, cConst);
      mlir::Value sum = mlir::arith::AddIOp::create(rewriter, loc, inDim, cVal);
      mlir::Value strideVal =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, strides[s]);
      mlir::Value div =
          mlir::arith::DivSIOp::create(rewriter, loc, sum, strideVal);
      mlir::Value one = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      sz = mlir::arith::AddIOp::create(rewriter, loc, div, one);
    }
    dynSizes.push_back(sz);
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
