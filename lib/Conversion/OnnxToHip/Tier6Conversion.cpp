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

/// Helper: pull a 1-D int64 constant tensor from a Value.  Accepts
/// onnx.Constant, arith.constant, and bufferization.to_tensor with a
/// `hip.inline_value` attribute.
static FailureOr<SmallVector<int64_t>> extractI64Constant(Value v) {
  if (!v)
    return failure();
  Operation *def = v.getDefiningOp();
  if (!def)
    return failure();
  ElementsAttr valueAttr;
  StringRef name = def->getName().getStringRef();
  if (name == "onnx.Constant") {
    valueAttr = def->getAttrOfType<ElementsAttr>("value");
  } else if (auto a = dyn_cast<mlir::arith::ConstantOp>(def)) {
    valueAttr = dyn_cast<ElementsAttr>(a.getValue());
  } else if (auto toT = dyn_cast<mlir::bufferization::ToTensorOp>(def)) {
    valueAttr = toT->getAttrOfType<ElementsAttr>("hip.inline_value");
  } else {
    return failure();
  }
  if (!valueAttr)
    return failure();
  auto dense = dyn_cast<DenseElementsAttr>(valueAttr);
  if (!dense)
    return failure();
  Type elem = dense.getElementType();
  SmallVector<int64_t> out;
  if (elem.isInteger(64)) {
    for (int64_t v : dense.getValues<int64_t>())
      out.push_back(v);
  } else if (elem.isInteger(32)) {
    for (int32_t v : dense.getValues<int32_t>())
      out.push_back(static_cast<int64_t>(v));
  } else {
    return failure();
  }
  return out;
}

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
    Value init = createEmptyTensor(rewriter, loc, resultType, input);
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
      // Fallback: tensor.dim on the input at the corresponding position
      // (right-aligned ONNX broadcast).
      int64_t inRank = inputType.getRank();
      int64_t srcDim = d - (outRank - inRank);
      if (srcDim >= 0 && srcDim < inRank)
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, input, srcDim));
      else
        dynSizes.push_back(
            mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
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
      // We don't know the runtime length at compile time -- the kernel
      // computes it from start/limit/delta and writes up to the
      // allocated buffer's size.  Pre-allocate a big enough buffer; the
      // caller is responsible for ensuring the model's downstream ops
      // don't read beyond the actual length.  Use 1 as a placeholder so
      // tensor.empty doesn't crash; the kernel will resize.
      dynSizes.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
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

    Value init = createEmptyTensor(rewriter, loc, resultType, input);

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
    Value init = createEmptyTensor(rewriter, loc, resultType, input);
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
