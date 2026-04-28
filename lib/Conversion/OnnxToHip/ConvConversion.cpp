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
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto weightsType = mlir::cast<mlir::RankedTensorType>(weights.getType());

  // Degenerate rank-0 (or rank-1) Conv lives only in Kokoro's iSTFT
  // dead-code path -- skip conversion so dropUnsupportedOnnxOps can
  // replace the onnx.Conv with a tensor.empty placeholder.
  if (inputType.getRank() < 3 || resultType.getRank() < 3)
    return rewriter.notifyMatchFailure(
        op, "ConvToHip: input/output rank too low (degenerate)");

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

  // hip.conv lowering requires NCHW (rank-4) tensors.  ONNX Conv1D
  // (NCW, rank-3) is reshaped here to NC1W by inserting a unit H dim
  // at axis 2 of input, axis 2 of weights, axis 2 of result.  The
  // per-spatial conv attrs (kernel/strides/pads/dilations) get their
  // H entry filled in as 1/1/[0,0]/1.
  bool was1D = inputType.getRank() == 3;
  if (was1D) {
    if (kernelShape.size() != 1 || strides.size() != 1 ||
        dilations.size() != 1 || pads.size() != 2) {
      return rewriter.notifyMatchFailure(
          op, "1D Conv has unexpected per-spatial attribute sizes");
    }
    auto inserted0H = [&](mlir::RankedTensorType t) {
      llvm::SmallVector<int64_t> shape(t.getShape().begin(),
                                       t.getShape().end());
      shape.insert(shape.begin() + 2, 1);
      return mlir::RankedTensorType::get(shape, t.getElementType());
    };
    auto unsqueezeH = [&](mlir::Value v, mlir::RankedTensorType srcType) {
      auto dstType = inserted0H(srcType);
      mlir::SmallVector<mlir::ReassociationIndices> reassoc = {{0}, {1, 2}, {3}};
      llvm::SmallVector<mlir::OpFoldResult> outShape;
      for (int64_t i = 0; i < dstType.getRank(); ++i) {
        if (dstType.isDynamicDim(i)) {
          int64_t srcDim = i;
          if (i == 2)
            outShape.push_back(rewriter.getIndexAttr(1));
          else {
            // Map dst dim back to src dim: 0->0, 1->1, 2->none, 3->2.
            int64_t s = (i < 2) ? i : i - 1;
            outShape.push_back(
                mlir::tensor::DimOp::create(rewriter, loc, v, s).getResult());
          }
        } else {
          outShape.push_back(rewriter.getIndexAttr(dstType.getDimSize(i)));
        }
      }
      return mlir::tensor::ExpandShapeOp::create(rewriter, loc, dstType, v,
                                                  reassoc, outShape)
          .getResult();
    };
    input = unsqueezeH(input, inputType);
    weights = unsqueezeH(weights, weightsType);
    inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
    weightsType = mlir::cast<mlir::RankedTensorType>(weights.getType());
    kernelShape = {1, kernelShape[0]};
    strides     = {1, strides[0]};
    dilations   = {1, dilations[0]};
    // ONNX pads layout for 1D: [pad_w_begin, pad_w_end].  4D wants
    // [pad_h_begin, pad_w_begin, pad_h_end, pad_w_end].
    pads = {0, pads[0], 0, pads[1]};
    // Result type also gets H=1 inserted.
    resultType = inserted0H(resultType);
  }

  // Create output tensor.  For Conv, batch (dim 0) is taken from input,
  // channels-out (dim 1) from weights (dim 0), and spatial dims follow:
  //   out[i] = floor((in[i] + padBegin[i] + padEnd[i]
  //                   - dilation[i]*(kernel[i]-1) - 1) / stride[i]) + 1
  int64_t rank = resultType.getRank();
  int64_t numSpatial = rank - 2;
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t d = 0; d < rank; ++d) {
    if (!resultType.isDynamicDim(d))
      continue;
    if (d == 0) {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, input, d));
    } else if (d == 1) {
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, weights, (int64_t)0));
    } else {
      int64_t si = d - 2;
      int64_t s = (si < (int64_t)strides.size()) ? strides[si] : 1;
      int64_t k = (si < (int64_t)kernelShape.size()) ? kernelShape[si] : 1;
      int64_t dl = (si < (int64_t)dilations.size()) ? dilations[si] : 1;
      int64_t pb = (si < (int64_t)pads.size()) ? pads[si] : 0;
      int64_t pe = (si + numSpatial < (int64_t)pads.size())
                       ? pads[si + numSpatial]
                       : 0;
      // effective_kernel = dilation*(kernel-1) + 1
      int64_t effK = dl * (k - 1) + 1;
      // ONNX Conv output dim:
      //   floor((in + pad_begin + pad_end - effective_kernel) / stride) + 1
      int64_t adj = pb + pe - effK;

      mlir::Value inDim =
          mlir::tensor::DimOp::create(rewriter, loc, input, d);
      mlir::Value inIdx = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), inDim);
      mlir::Value adjVal = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(adj));
      mlir::Value strideVal = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(s));
      // floor((in + adj) / stride) + 1
      mlir::Value sum =
          mlir::arith::AddIOp::create(rewriter, loc, inIdx, adjVal);
      mlir::Value div =
          mlir::arith::DivSIOp::create(rewriter, loc, sum, strideVal);
      mlir::Value one = mlir::arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(1));
      mlir::Value outDim64 =
          mlir::arith::AddIOp::create(rewriter, loc, div, one);
      mlir::Value outDim = mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), outDim64);
      dynSizes.push_back(outDim);
    }
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

  // Create hip.conv operation using generic builder
  auto hipOp = mlir::hip::ConvOp::create(
      rewriter, loc, mlir::TypeRange{resultType}, operands, attrs);
  mlir::Value out = hipOp.getResult(0);

  // 1D Conv: collapse the inserted H=1 dim back so the result type
  // matches the original onnx.Conv result.
  if (was1D) {
    auto onnxResultType = mlir::cast<mlir::RankedTensorType>(
        op->getResult(0).getType());
    mlir::SmallVector<mlir::ReassociationIndices> reassoc = {{0}, {1, 2}, {3}};
    out = mlir::tensor::CollapseShapeOp::create(rewriter, loc, onnxResultType,
                                                  out, reassoc)
              .getResult();
  }

  rewriter.replaceOp(op, out);
  return mlir::success();
}

} // namespace

void mlir::hip::populateConvConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<ConvToHip>(ctx);
}

} // namespace hip
} // namespace mlir
