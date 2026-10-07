/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// onnx.QLinearConv -> hip.qlinear_conv
//
// Before:
//   %y = "onnx.QLinearConv"(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b)
//        {kernel_shape = [3, 3], strides = [2, 2], pads = [1, 1, 1, 1],
//         dilations = [1, 1], group = 1, auto_pad = "NOTSET"}
//        : (tensor<1x3x8x8xui8>, tensor<f32>, tensor<ui8>,
//           tensor<4x3x3x3xi8>, tensor<f32>, tensor<i8>,
//           tensor<f32>, tensor<ui8>, tensor<4xi32>)
//        -> tensor<1x4x4x4xui8>
// After:
//   %init = tensor.empty() : tensor<1x4x4x4xui8>
//   %y = hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
//            ...) outs(%init : tensor<1x4x4x4xui8>) {kernel_shape = [3, 3],
//            ...}
//
// Input and output quantization is per-tensor. Weight quantization is
// per-tensor or per output channel. auto_pad must be NOTSET; the pads
// attribute carries the window. Bias is optional.
static FailureOr<SmallVector<int64_t>>
readI64Array(Operation *op, StringRef name, int64_t count,
             ArrayRef<int64_t> fallback) {
  SmallVector<int64_t> values;
  if (!op->hasAttr(name)) {
    values.assign(fallback.begin(), fallback.end());
    return values;
  }
  auto attr = op->getAttrOfType<ArrayAttr>(name);
  if (!attr || static_cast<int64_t>(attr.size()) != count)
    return failure();
  for (Attribute entry : attr) {
    auto intAttr = dyn_cast<IntegerAttr>(entry);
    if (!intAttr)
      return failure();
    values.push_back(intAttr.getInt());
  }
  return values;
}

static bool isEightBit(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && integer.getWidth() == 8;
}

// Per-tensor is one element. Per-output-channel is `channels` elements.
static LogicalResult checkScaleZp(PatternRewriter &rewriter, Operation *op,
                                  Value scale, Value zp, Type storage,
                                  int64_t channels, bool allowPerChannel,
                                  StringRef name) {
  auto scaleType = dyn_cast<RankedTensorType>(scale.getType());
  auto zpType = dyn_cast<RankedTensorType>(zp.getType());
  auto reject = [&](const char *suffix) {
    return rewriter.notifyMatchFailure(op, (llvm::Twine(name) + suffix).str());
  };
  if (!scaleType || !zpType || scaleType.getRank() > 1 || zpType.getRank() > 1)
    return reject(" scale and zero point must be rank 0 or 1");
  if (!scaleType.getElementType().isF32())
    return reject(" scale must be f32");
  if (zpType.getElementType() != storage)
    return reject(" zero point type must match the quantized tensor");
  if (!scaleType.hasStaticShape() || !zpType.hasStaticShape())
    return success();
  int64_t nScale = scaleType.getNumElements();
  if (nScale != zpType.getNumElements() || nScale < 1)
    return reject(" scale and zero point lengths must match");
  if (nScale == 1)
    return success();
  if (allowPerChannel && channels != ShapedType::kDynamic && nScale == channels)
    return success();
  return reject(" quantization must be per-tensor or per output channel");
}

static int64_t convOutDim(int64_t input, int64_t padBegin, int64_t padEnd,
                          int64_t dilation, int64_t kernel, int64_t stride) {
  if (stride <= 0 || kernel <= 0 || dilation <= 0)
    return -1;
  int64_t numer = input + padBegin + padEnd - dilation * (kernel - 1) - 1;
  if (numer < 0)
    return -1;
  return numer / stride + 1;
}

struct QLinearConvToHip : public RewritePattern {
  explicit QLinearConvToHip(MLIRContext *ctx)
      : RewritePattern("onnx.QLinearConv", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 ||
        (op->getNumOperands() != 8 && op->getNumOperands() != 9))
      return rewriter.notifyMatchFailure(
          op, "onnx.QLinearConv expects 8 or 9 operands and 1 result");

    if (auto padMode = op->getAttrOfType<StringAttr>("auto_pad")) {
      if (padMode.getValue() != "NOTSET")
        return rewriter.notifyMatchFailure(
            op, "onnx.QLinearConv auto_pad must be NOTSET");
    }

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();

    Value input = op->getOperand(0);
    Value inputScale = op->getOperand(1);
    Value inputZp = op->getOperand(2);
    Value weights = op->getOperand(3);
    Value weightScale = op->getOperand(4);
    Value weightZp = op->getOperand(5);
    Value outputScale = op->getOperand(6);
    Value outputZp = op->getOperand(7);
    Value bias;
    if (op->getNumOperands() == 9)
      bias = op->getOperand(8);
    if (bias && isa<NoneType>(bias.getType()))
      bias = Value();

    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto weightsType = dyn_cast<RankedTensorType>(weights.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !weightsType || !resultType || inputType.getRank() != 4 ||
        weightsType.getRank() != 4 || resultType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "onnx.QLinearConv lowering expects rank-4 input, weights, and "
              "result");
    for (int64_t dim : llvm::seq<int64_t>(1, 4)) {
      if (resultType.isDynamicDim(dim))
        return rewriter.notifyMatchFailure(
            op, "onnx.QLinearConv result channel and spatial dims must be "
                "static");
    }

    if (!isEightBit(inputType.getElementType()) ||
        !isEightBit(weightsType.getElementType()) ||
        !isEightBit(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "onnx.QLinearConv input, weights, and result must be 8-bit");

    int64_t group = 1;
    if (auto groupAttr = op->getAttrOfType<IntegerAttr>("group"))
      group = groupAttr.getInt();
    if (group < 1)
      return rewriter.notifyMatchFailure(op, "group must be positive");

    FailureOr<SmallVector<int64_t>> kernelOr = failure();
    if (op->hasAttr("kernel_shape")) {
      kernelOr = readI64Array(op, "kernel_shape", 2, {1, 1});
      if (failed(kernelOr))
        return rewriter.notifyMatchFailure(
            op, "kernel_shape must be an integer array of length 2");
    } else if (weightsType.isDynamicDim(2) || weightsType.isDynamicDim(3)) {
      return rewriter.notifyMatchFailure(
          op, "kernel_shape is required when weight spatial dims are dynamic");
    } else {
      kernelOr = SmallVector<int64_t>{weightsType.getDimSize(2),
                                      weightsType.getDimSize(3)};
    }
    auto stridesOr = readI64Array(op, "strides", 2, {1, 1});
    auto dilationsOr = readI64Array(op, "dilations", 2, {1, 1});
    auto padsOr = readI64Array(op, "pads", 4, {0, 0, 0, 0});
    if (failed(kernelOr) || failed(stridesOr) || failed(dilationsOr) ||
        failed(padsOr))
      return rewriter.notifyMatchFailure(
          op, "kernel_shape, strides, dilations, and pads must be integer "
              "arrays of the 2D lengths");

    SmallVector<int64_t> kernel = *kernelOr;
    SmallVector<int64_t> strides = *stridesOr;
    SmallVector<int64_t> dilations = *dilationsOr;
    SmallVector<int64_t> pads = *padsOr;
    if (kernel[0] <= 0 || kernel[1] <= 0 || strides[0] <= 0 ||
        strides[1] <= 0 || dilations[0] <= 0 || dilations[1] <= 0)
      return rewriter.notifyMatchFailure(
          op, "kernel, stride, and dilation entries must be positive");

    int64_t outChannels = weightsType.isDynamicDim(0)
                              ? ShapedType::kDynamic
                              : weightsType.getDimSize(0);
    if (!weightsType.isDynamicDim(0) && resultType.getDimSize(1) != outChannels)
      return rewriter.notifyMatchFailure(
          op, "result channels must match the weight output channels");
    if (!inputType.isDynamicDim(0) && !resultType.isDynamicDim(0) &&
        inputType.getDimSize(0) != resultType.getDimSize(0))
      return rewriter.notifyMatchFailure(op, "batch dimensions must match");

    if (!inputType.isDynamicDim(1) && !weightsType.isDynamicDim(1) &&
        weightsType.getDimSize(1) * group != inputType.getDimSize(1))
      return rewriter.notifyMatchFailure(
          op, "weight input channels times group must equal input channels");
    if (!weightsType.isDynamicDim(2) && weightsType.getDimSize(2) != kernel[0])
      return rewriter.notifyMatchFailure(op,
                                         "kernel height must match weights");
    if (!weightsType.isDynamicDim(3) && weightsType.getDimSize(3) != kernel[1])
      return rewriter.notifyMatchFailure(op, "kernel width must match weights");

    for (int axis : {0, 1}) {
      int64_t inDim = inputType.getDimSize(axis + 2);
      if (inputType.isDynamicDim(axis + 2))
        continue;
      int64_t expected =
          convOutDim(inDim, pads[axis], pads[axis + 2], dilations[axis],
                     kernel[axis], strides[axis]);
      if (expected != resultType.getDimSize(axis + 2))
        return rewriter.notifyMatchFailure(
            op, "result spatial shape does not match the convolution formula");
    }

    if (failed(checkScaleZp(rewriter, op, inputScale, inputZp,
                            inputType.getElementType(),
                            /*channels=*/ShapedType::kDynamic,
                            /*allowPerChannel=*/false, "input")) ||
        failed(checkScaleZp(rewriter, op, weightScale, weightZp,
                            weightsType.getElementType(), outChannels,
                            /*allowPerChannel=*/true, "weight")) ||
        failed(checkScaleZp(rewriter, op, outputScale, outputZp,
                            resultType.getElementType(),
                            /*channels=*/ShapedType::kDynamic,
                            /*allowPerChannel=*/false, "output")))
      return failure();

    if (bias) {
      auto biasType = dyn_cast<RankedTensorType>(bias.getType());
      if (!biasType || biasType.getRank() != 1 ||
          !biasType.getElementType().isInteger(32))
        return rewriter.notifyMatchFailure(op, "bias must be rank-1 i32");
      if (biasType.hasStaticShape() && outChannels != ShapedType::kDynamic &&
          biasType.getDimSize(0) != outChannels)
        return rewriter.notifyMatchFailure(
            op, "bias length must equal the output channels");
    }

    Location loc = op->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = QLinearConvOp::create(
        rewriter, loc, resultType, *ctxOrFailure, input, inputScale, inputZp,
        weights, weightScale, weightZp, outputScale, outputZp, bias, init,
        rewriter.getI64ArrayAttr(kernel), rewriter.getI64ArrayAttr(strides),
        rewriter.getI64ArrayAttr(pads), rewriter.getI64ArrayAttr(dilations),
        rewriter.getI64IntegerAttr(group));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void populateQLinearConvConversionPatterns(RewritePatternSet &patterns,
                                           MLIRContext *ctx) {
  patterns.add<QLinearConvToHip>(ctx);
}

} // namespace hip
} // namespace mlir
