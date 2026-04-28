/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Conversion patterns for the Tier 6 (sequence/decoder) ONNX ops:
//
//   - onnx.Pad           -> hip.pad
//   - onnx.Expand        -> hip.expand
//   - onnx.Range         -> hip.range  (when not foldable to a constant)
//   - onnx.ConvTranspose -> hip.conv_transpose
//   - onnx.Resize        -> hip.resize
//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// onnx.Pad -> hip.pad
//===----------------------------------------------------------------------===//
//
// ONNX Pad signature (opset 11+):
//     pad(input, pads [, value [, axes]])  -> output
//
// `pads` is a 1-D int64 tensor of length 2*rank: pads_begin then
// pads_end, in axis order.  Modes: "constant" (default), "reflect",
// "edge", "wrap".  We support constant/reflect/edge today; wrap can
// be added when needed.

struct PadToHip : public RewritePattern {
  PadToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Pad", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() < 2 || op->getNumOperands() > 4)
      return rewriter.notifyMatchFailure(
          op, "onnx.Pad expects 2..4 operands (input, pads [, value [, axes]])");

    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "ranked tensors required");
    int64_t rank = inputType.getRank();
    // Degenerate rank-0 Pad (Kokoro's iSTFT noise path emits one):
    // padding a scalar produces the scalar unchanged.
    if (rank == 0 && resultType.getRank() == 0) {
      rewriter.replaceOp(op, input);
      return success();
    }

    auto padsOr = extractI64Constant(op->getOperand(1));
    if (failed(padsOr))
      return rewriter.notifyMatchFailure(op, "pads must be a constant");
    SmallVector<int64_t> pads = *padsOr;

    SmallVector<int64_t> padsBegin(rank, 0);
    SmallVector<int64_t> padsEnd(rank, 0);
    if (op->getNumOperands() >= 4) {
      auto axesOr = extractI64Constant(op->getOperand(3));
      if (failed(axesOr))
        return rewriter.notifyMatchFailure(op, "axes must be a constant");
      SmallVector<int64_t> axes = *axesOr;
      if ((int64_t)pads.size() != 2 * (int64_t)axes.size())
        return rewriter.notifyMatchFailure(op, "pads/axes length mismatch");
      for (size_t i = 0; i < axes.size(); ++i) {
        int64_t ax = axes[i];
        if (ax < 0)
          ax += rank;
        if (ax < 0 || ax >= rank)
          return rewriter.notifyMatchFailure(op, "axis out of range");
        padsBegin[ax] = pads[i];
        padsEnd[ax] = pads[i + axes.size()];
      }
    } else {
      if ((int64_t)pads.size() != 2 * rank)
        return rewriter.notifyMatchFailure(
            op, "pads must have 2*rank elements");
      for (int64_t i = 0; i < rank; ++i) {
        padsBegin[i] = pads[i];
        padsEnd[i] = pads[i + rank];
      }
    }

    // Mode: 0 = constant, 1 = reflect, 2 = edge (wrap not supported yet).
    int64_t mode = 0;
    if (auto m = op->getAttrOfType<StringAttr>("mode")) {
      StringRef s = m.getValue();
      if (s == "constant")
        mode = 0;
      else if (s == "reflect")
        mode = 1;
      else if (s == "edge")
        mode = 2;
      else
        return rewriter.notifyMatchFailure(
            op, "onnx.Pad mode must be constant/reflect/edge");
    }

    // Pad value: optional 3rd operand (constant scalar).  Defaults to 0.
    float padValue = 0.0f;
    if (op->getNumOperands() >= 3 && mode == 0) {
      Value v = op->getOperand(2);
      if (auto def = v.getDefiningOp()) {
        ElementsAttr valueAttr;
        if (auto vAttr = def->getAttrOfType<ElementsAttr>("value"))
          valueAttr = vAttr;
        else if (auto a = dyn_cast<mlir::arith::ConstantOp>(def))
          valueAttr = dyn_cast<ElementsAttr>(a.getValue());
        if (auto dense = dyn_cast_or_null<DenseElementsAttr>(valueAttr)) {
          if (dense.getNumElements() == 1) {
            Type elem = dense.getElementType();
            if (elem.isF32())
              padValue = *dense.value_begin<float>();
            else if (elem.isF16() || elem.isBF16()) {
              APFloat f = *dense.value_begin<APFloat>();
              padValue = static_cast<float>(f.convertToFloat());
            }
          }
        }
      }
    }

    Location loc = op->getLoc();
    SmallVector<Value> dynSizes;
    for (int64_t d = 0; d < resultType.getRank(); ++d) {
      if (!resultType.isDynamicDim(d))
        continue;
      Value inDim = tensor::DimOp::create(rewriter, loc, input, d);
      Value inDim64 = arith::IndexCastOp::create(
          rewriter, loc, rewriter.getI64Type(), inDim);
      Value padBegin = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(padsBegin[d]));
      Value padEnd = arith::ConstantOp::create(
          rewriter, loc, rewriter.getI64IntegerAttr(padsEnd[d]));
      Value padded = arith::AddIOp::create(
          rewriter, loc, arith::AddIOp::create(rewriter, loc, inDim64, padBegin),
          padEnd);
      dynSizes.push_back(arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), padded));
    }
    Value init = tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        dynSizes);
    auto hipOp = PadOp::create(rewriter, loc, resultType, context, input, init,
                                rewriter.getI64ArrayAttr(padsBegin),
                                rewriter.getI64ArrayAttr(padsEnd),
                                rewriter.getI64IntegerAttr(mode),
                                rewriter.getF32FloatAttr(padValue));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.Expand -> hip.expand
//===----------------------------------------------------------------------===//
//
// ONNX Expand broadcasts `input` to the shape provided by `shape`
// (a 1-D int64 tensor).  We only need the result type to drive the
// broadcast; the runtime kernel reads stride-zero on broadcast dims.

struct ExpandToHip : public RewritePattern {
  ExpandToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Expand", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() != 2)
      return rewriter.notifyMatchFailure(
          op, "onnx.Expand expects 2 operands (input, shape)");
    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "ranked tensors required");

    Location loc = op->getLoc();
    // Build init: dynamic dims come from the shape operand.  For now we
    // require the shape operand to be a constant we can read; the dynamic
    // case can be handled later via a runtime shape tensor.
    auto shapeOr = extractI64Constant(op->getOperand(1));
    SmallVector<Value> dynSizes;
    int64_t outRank = resultType.getRank();
    for (int64_t d = 0; d < outRank; ++d) {
      if (!resultType.isDynamicDim(d))
        continue;
      if (succeeded(shapeOr) && d < (int64_t)shapeOr->size()) {
        int64_t v = (*shapeOr)[d];
        if (v >= 0) {
          dynSizes.push_back(
              mlir::arith::ConstantIndexOp::create(rewriter, loc, v));
          continue;
        }
      }
      // Dynamic Expand shape: read the requested output dimension from the
      // runtime shape tensor.  Falling back to the input dim is incorrect for
      // broadcast axes (for Kokoro style conditioning it collapses [1,N,128]
      // to [1,1,128]).
      Value idx = mlir::arith::ConstantIndexOp::create(rewriter, loc, d);
      Value dim = mlir::tensor::ExtractOp::create(
          rewriter, loc, op->getOperand(1), ValueRange{idx});
      dynSizes.push_back(mlir::arith::IndexCastOp::create(
          rewriter, loc, rewriter.getIndexType(), dim));
    }
    Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        dynSizes);
    auto hipOp =
        ExpandOp::create(rewriter, loc, resultType, context, input, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.Range -> hip.range  (when start/limit/delta are runtime tensors)
//===----------------------------------------------------------------------===//
//
// The constant-fold path lives in Tier2ShapeConversion.cpp; this
// handles the runtime case (Kokoro hits it for a Range whose `limit`
// comes from a duration predictor).

struct RangeToHip : public RewritePattern {
  RangeToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Range", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    if (op->getNumOperands() != 3)
      return rewriter.notifyMatchFailure(op, "onnx.Range needs 3 operands");
    Value start = op->getOperand(0);
    Value limit = op->getOperand(1);
    Value delta = op->getOperand(2);
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be ranked");

    Location loc = op->getLoc();
    SmallVector<Value> dynSizes;
    if (resultType.getRank() == 1 && resultType.isDynamicDim(0)) {
      // Compute the runtime length n = max((limit - start) / delta, 0).
      // start/limit/delta are rank-0 tensors; tensor.extract pulls the
      // scalar so we can do arith on it.
      auto elemType = resultType.getElementType();
      Value sScalar =
          mlir::tensor::ExtractOp::create(rewriter, loc, start, mlir::ValueRange{})
              .getResult();
      Value lScalar =
          mlir::tensor::ExtractOp::create(rewriter, loc, limit, mlir::ValueRange{})
              .getResult();
      Value dScalar =
          mlir::tensor::ExtractOp::create(rewriter, loc, delta, mlir::ValueRange{})
              .getResult();
      Value n;
      if (elemType.isInteger()) {
        Value diff = mlir::arith::SubIOp::create(rewriter, loc, lScalar,
                                                  sScalar);
        n = mlir::arith::DivSIOp::create(rewriter, loc, diff, dScalar);
        // Cast to index.
        n = mlir::arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(), n);
      } else {
        Value diff = mlir::arith::SubFOp::create(rewriter, loc, lScalar,
                                                  sScalar);
        Value div = mlir::arith::DivFOp::create(rewriter, loc, diff, dScalar);
        Value asI64 = mlir::arith::FPToSIOp::create(
            rewriter, loc, rewriter.getI64Type(), div);
        n = mlir::arith::IndexCastOp::create(rewriter, loc,
                                              rewriter.getIndexType(), asI64);
      }
      // clamp to >= 0
      Value zeroIdx =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 0);
      n = mlir::arith::MaxSIOp::create(rewriter, loc, n, zeroIdx);
      dynSizes.push_back(n);
    }
    Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        dynSizes);

    auto hipOp = RangeOp::create(rewriter, loc, resultType, context, start,
                                  limit, delta, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.ConvTranspose -> hip.conv_transpose
//===----------------------------------------------------------------------===//

struct ConvTransposeToHip : public RewritePattern {
  ConvTransposeToHip(MLIRContext *ctx)
      : RewritePattern("onnx.ConvTranspose", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Location loc = op->getLoc();
    Value input = op->getOperand(0);
    Value weights = op->getOperand(1);
    bool hasBias = op->getNumOperands() > 2 &&
                   !isa<NoneType>(op->getOperand(2).getType());
    Value bias = hasBias ? op->getOperand(2) : nullptr;

    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    auto inType = dyn_cast<RankedTensorType>(input.getType());
    if (!inType || inType.getRank() < 3 || resultType.getRank() < 3)
      return rewriter.notifyMatchFailure(
          op, "ConvTransposeToHip: input/output rank too low (degenerate)");

    SmallVector<int64_t> kernelShape;
    if (auto attr = op->getAttrOfType<ArrayAttr>("kernel_shape")) {
      for (auto a : attr)
        kernelShape.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());
    }
    SmallVector<int64_t> strides;
    if (auto attr = op->getAttrOfType<ArrayAttr>("strides"))
      for (auto a : attr)
        strides.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());
    else
      strides.assign(kernelShape.size(), 1);

    SmallVector<int64_t> pads;
    if (auto attr = op->getAttrOfType<ArrayAttr>("pads"))
      for (auto a : attr)
        pads.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());
    else
      pads.assign(kernelShape.size() * 2, 0);

    SmallVector<int64_t> dilations;
    if (auto attr = op->getAttrOfType<ArrayAttr>("dilations"))
      for (auto a : attr)
        dilations.push_back(cast<IntegerAttr>(a).getValue().getSExtValue());
    else
      dilations.assign(kernelShape.size(), 1);

    SmallVector<int64_t> outputPadding;
    if (auto attr = op->getAttrOfType<ArrayAttr>("output_padding"))
      for (auto a : attr)
        outputPadding.push_back(
            cast<IntegerAttr>(a).getValue().getSExtValue());

    int64_t group = 1;
    if (auto attr = op->getAttrOfType<IntegerAttr>("group"))
      group = attr.getValue().getSExtValue();

    // Build the output tensor.  ConvTranspose spatial dimensions follow:
    //   out[i] = stride[i] * (in[i] - 1) + outPad[i]
    //            + ((kernel[i]-1)*dilation[i] + 1) - padBegin[i] - padEnd[i]
    // Batch (dim 0) and channel (dim 1) are taken from the input.
    int64_t rank = resultType.getRank();
    int64_t numSpatial = rank - 2;

    SmallVector<Value> dynSizes;
    for (int64_t d = 0; d < rank; ++d) {
      if (!resultType.isDynamicDim(d))
        continue;
      if (d < 2) {
        // Batch or channel dim: take from input
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, input, d));
      } else {
        // Spatial dim index within the spatial vectors
        int64_t si = d - 2;
        int64_t s = (si < (int64_t)strides.size()) ? strides[si] : 1;
        int64_t k = (si < (int64_t)kernelShape.size()) ? kernelShape[si] : 1;
        int64_t dl = (si < (int64_t)dilations.size()) ? dilations[si] : 1;
        int64_t pb = (si < (int64_t)pads.size()) ? pads[si] : 0;
        int64_t pe = (si + numSpatial < (int64_t)pads.size())
                         ? pads[si + numSpatial]
                         : 0;
        int64_t op_ =
            (si < (int64_t)outputPadding.size()) ? outputPadding[si] : 0;
        // effective_kernel = (k - 1) * dl + 1
        int64_t effK = (k - 1) * dl + 1;
        // constant part = effK + op_ - pb - pe
        int64_t cst = effK + op_ - pb - pe;
        // out = s * (in - 1) + cst  =>  out = s*in - s + cst
        Value inDim =
            tensor::DimOp::create(rewriter, loc, input, d);
        Value inIdx = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), inDim);
        Value strideVal = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(s));
        Value oneVal = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(1));
        Value cstVal = arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(cst));
        // s * (in - 1) + cst
        Value inMinus1 =
            arith::SubIOp::create(rewriter, loc, inIdx, oneVal);
        Value scaled =
            arith::MulIOp::create(rewriter, loc, strideVal, inMinus1);
        Value outDim64 =
            arith::AddIOp::create(rewriter, loc, scaled, cstVal);
        Value outDim = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), outDim64);
        dynSizes.push_back(outDim);
      }
    }
    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                         resultType.getElementType(), dynSizes);

    SmallVector<Value> operands = {context, input, weights};
    if (bias)
      operands.push_back(bias);
    operands.push_back(init);

    SmallVector<NamedAttribute> attrs;
    attrs.push_back(rewriter.getNamedAttr("kernel_shape",
                                          rewriter.getI64ArrayAttr(kernelShape)));
    attrs.push_back(
        rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(strides)));
    attrs.push_back(
        rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(pads)));
    attrs.push_back(rewriter.getNamedAttr("dilations",
                                          rewriter.getI64ArrayAttr(dilations)));
    if (!outputPadding.empty())
      attrs.push_back(rewriter.getNamedAttr(
          "output_padding", rewriter.getI64ArrayAttr(outputPadding)));
    attrs.push_back(
        rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)));

    auto hipOp =
        ConvTransposeOp::create(rewriter, loc, TypeRange{resultType}, operands,
                                 attrs);
    rewriter.replaceOp(op, hipOp.getResult(0));
    return success();
  }
};

//===----------------------------------------------------------------------===//
// onnx.Resize -> hip.resize
//===----------------------------------------------------------------------===//
//
// ONNX Resize signature (opset 13+):
//   Resize(X, roi, scales, sizes [, antialias])
// One of `scales` / `sizes` is empty.  The output type is sufficient
// to drive the runtime kernel; we pick the interpolation mode and
// coord_transform from attributes.

struct ResizeToHip : public RewritePattern {
  ResizeToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Resize", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(op, "ranked tensors required");

    int64_t mode = 0; // nearest by default
    if (auto m = op->getAttrOfType<StringAttr>("mode")) {
      StringRef s = m.getValue();
      if (s == "nearest")
        mode = 0;
      else if (s == "linear")
        mode = 1;
      else if (s == "cubic")
        mode = 2;
      else
        return rewriter.notifyMatchFailure(
            op, "onnx.Resize mode must be nearest/linear/cubic");
    }
    int64_t coordXform = 0; // half_pixel by default
    if (auto a = op->getAttrOfType<StringAttr>("coordinate_transformation_mode")) {
      StringRef s = a.getValue();
      if (s == "half_pixel")
        coordXform = 0;
      else if (s == "pytorch_half_pixel")
        coordXform = 1;
      else if (s == "align_corners")
        coordXform = 2;
      else if (s == "asymmetric")
        coordXform = 3;
      else if (s == "tf_crop_and_resize")
        coordXform = 4;
    }

    float cubicCoeffA = -0.75f;
    if (auto a = op->getAttrOfType<FloatAttr>("cubic_coeff_a"))
      cubicCoeffA = static_cast<float>(a.getValueAsDouble());

    Location loc = op->getLoc();

    // Build output tensor.  For dynamic spatial dims the size comes from the
    // scales or sizes operand, not from the input tensor.
    SmallVector<Value> dynSizes;
    int64_t rank = resultType.getRank();

    // Try to extract constant scales or sizes from ONNX operands.
    // Resize(X, roi, scales [, sizes]): operand indices 0, 1, 2, [3]
    SmallVector<double> scales;
    SmallVector<int64_t> sizes;

    // Skip NoneType operands when finding scales (operand 2)
    for (int64_t opIdx = 2;
         opIdx < (int64_t)op->getNumOperands() && scales.empty(); ++opIdx) {
      Value operand = op->getOperand(opIdx);
      if (isa<NoneType>(operand.getType()))
        continue;
      Operation *def = operand.getDefiningOp();
      if (!def)
        continue;
      ElementsAttr attr;
      if (def->getName().getStringRef() == "onnx.Constant")
        attr = def->getAttrOfType<ElementsAttr>("value");
      else if (auto a = dyn_cast<arith::ConstantOp>(def))
        attr = dyn_cast<ElementsAttr>(a.getValue());
      if (!attr)
        continue;
      auto dense = dyn_cast<DenseElementsAttr>(attr);
      if (!dense)
        continue;
      Type elemTy = dense.getElementType();
      if (elemTy.isF32())
        for (float v : dense.getValues<float>())
          scales.push_back(v);
      else if (elemTy.isF64())
        for (double v : dense.getValues<double>())
          scales.push_back(v);
      else if (elemTy.isInteger(64))
        for (int64_t v : dense.getValues<int64_t>())
          sizes.push_back(v);
    }

    for (int64_t d = 0; d < rank; ++d) {
      if (!resultType.isDynamicDim(d))
        continue;
      if (!sizes.empty() && d < (int64_t)sizes.size() && sizes[d] > 0) {
        dynSizes.push_back(
            arith::ConstantIndexOp::create(rewriter, loc, sizes[d]));
      } else if (!scales.empty() && d < (int64_t)scales.size() &&
                 scales[d] != 0.0) {
        // output_dim = floor(input_dim * scale)
        Value inDim = tensor::DimOp::create(rewriter, loc, input, d);
        Value inI64 = arith::IndexCastOp::create(
            rewriter, loc, rewriter.getI64Type(), inDim);
        Value inF64 = arith::SIToFPOp::create(rewriter, loc,
                                               rewriter.getF64Type(), inI64);
        Value scaleVal = arith::ConstantOp::create(
            rewriter, loc,
            rewriter.getF64FloatAttr(scales[d]));
        Value outF64 = arith::MulFOp::create(rewriter, loc, inF64, scaleVal);
        Value outI64 = arith::FPToSIOp::create(rewriter, loc,
                                                rewriter.getI64Type(), outF64);
        dynSizes.push_back(arith::IndexCastOp::create(
            rewriter, loc, rewriter.getIndexType(), outI64));
      } else {
        // Fallback: use input dimension (identity resize)
        dynSizes.push_back(tensor::DimOp::create(rewriter, loc, input, d));
      }
    }

    Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                          resultType.getElementType(), dynSizes);
    auto hipOp = ResizeOp::create(rewriter, loc, resultType, context, input,
                                   init, rewriter.getI64IntegerAttr(mode),
                                   rewriter.getI64IntegerAttr(coordXform),
                                   rewriter.getF32FloatAttr(cubicCoeffA));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateTier6ConversionPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {
  patterns.add<PadToHip>(ctx);
  patterns.add<ExpandToHip>(ctx);
  patterns.add<RangeToHip>(ctx);
  patterns.add<ConvTransposeToHip>(ctx);
  patterns.add<ResizeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
