/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

// com.microsoft.NhwcConv is onnx.Conv with NHWC activations and weights laid
// out as [M, kH, kW, C/group]. hip.conv consumes NCHW activations and
// [M, C/group, kH, kW] weights, so the layouts are transposed around it:
//
// Before:
//   %y = "onnx.Custom"(%x, %w, %b) {
//          function_name = "NhwcConv", domain_name = "com.microsoft",
//          kernel_shape = [3, 3], pads = [1, 1, 1, 1], strides = [1, 1],
//          dilations = [1, 1], group = 1 : si64}
//        : (tensor<?x?x?x4xf16>, tensor<512x3x3x4xf16>, tensor<512xf16>)
//          -> tensor<?x?x?x512xf16>
// After:
//   %xn = hip.transpose %x {perm = [0, 3, 1, 2]} : -> tensor<?x4x?x?xf16>
//   %wn = hip.transpose %w {perm = [0, 3, 1, 2]} : -> tensor<512x4x3x3xf16>
//   %yn = hip.conv(%xn, %wn, %b) {kernel_shape = [3, 3], ...}
//       : -> tensor<?x512x?x?xf16>
//   %y  = hip.transpose %yn {perm = [0, 2, 3, 1]} : -> tensor<?x?x?x512xf16>
struct NhwcConvToHip : public RewritePattern {
  NhwcConvToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

Value dimOf(PatternRewriter &rewriter, Location loc, Value tensor,
            int64_t index) {
  return tensor::DimOp::create(rewriter, loc, tensor, index);
}

// out = (in + padBegin + padEnd - dilation * (kernel - 1) - 1) / stride + 1
Value convOutExtent(PatternRewriter &rewriter, Location loc, Value inExtent,
                    int64_t kernel, int64_t stride, int64_t dilation,
                    int64_t padBegin, int64_t padEnd) {
  auto i64 = [&](int64_t v) {
    return arith::ConstantIndexOp::create(rewriter, loc, v);
  };
  int64_t addend = padBegin + padEnd - dilation * (kernel - 1) - 1;
  Value adjusted = arith::AddIOp::create(rewriter, loc, inExtent, i64(addend));
  Value divided = arith::DivSIOp::create(rewriter, loc, adjusted, i64(stride));
  return arith::AddIOp::create(rewriter, loc, divided, i64(1));
}

Value transposeRank4(PatternRewriter &rewriter, Location loc, Value context,
                     Value input, RankedTensorType resultType,
                     ArrayRef<int64_t> perm, ArrayRef<Value> dynSizes) {
  Value init = tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
  return TransposeOp::create(rewriter, loc, context, input, init,
                             rewriter.getI64ArrayAttr(perm))
      ->getResult(0);
}

LogicalResult NhwcConvToHip::matchAndRewrite(Operation *op,
                                             PatternRewriter &rewriter) const {
  auto funcName = op->getAttrOfType<StringAttr>("function_name");
  if (!funcName || funcName.getValue() != "NhwcConv")
    return rewriter.notifyMatchFailure(op, "not NhwcConv");

  auto domain = op->getAttrOfType<StringAttr>("domain_name");
  if (!domain || domain.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(op,
                                       "NhwcConv domain must be com.microsoft");

  auto autoPad = op->getAttrOfType<StringAttr>("auto_pad");
  StringRef padMode = autoPad ? autoPad.getValue() : "NOTSET";
  if (padMode != "NOTSET" && padMode != "VALID")
    return rewriter.notifyMatchFailure(
        op, "NhwcConv auto_pad must be NOTSET or VALID");

  if (op->getNumOperands() < 2 || op->getNumOperands() > 3 ||
      op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(
        op, "NhwcConv expects X, W, optional B, and one result");

  Value input = op->getOperand(0);
  Value weights = op->getOperand(1);
  Value bias;
  if (op->getNumOperands() == 3 && !isa<NoneType>(op->getOperand(2).getType()))
    bias = op->getOperand(2);

  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto weightType = dyn_cast<RankedTensorType>(weights.getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !weightType || !resultType || inputType.getRank() != 4 ||
      weightType.getRank() != 4 || resultType.getRank() != 4)
    return rewriter.notifyMatchFailure(op, "NhwcConv requires rank-4 tensors");
  if (inputType.getElementType() != weightType.getElementType() ||
      inputType.getElementType() != resultType.getElementType())
    return rewriter.notifyMatchFailure(op, "NhwcConv dtypes must match");
  if (bias) {
    auto biasType = dyn_cast<RankedTensorType>(bias.getType());
    if (!biasType || biasType.getRank() != 1 ||
        biasType.getElementType() != inputType.getElementType())
      return rewriter.notifyMatchFailure(op, "NhwcConv bias must be 1-D");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  Value context = *ctxOrFailure;
  Location loc = op->getLoc();

  auto readInts = [&](StringRef name, int64_t count, int64_t fallback,
                      SmallVectorImpl<int64_t> &out) -> LogicalResult {
    out.assign(count, fallback);
    auto attr = op->getAttrOfType<ArrayAttr>(name);
    if (!attr)
      return success();
    if (static_cast<int64_t>(attr.size()) != count)
      return failure();
    for (auto [index, element] : llvm::enumerate(attr)) {
      auto intAttr = dyn_cast<IntegerAttr>(element);
      if (!intAttr)
        return failure();
      out[index] = intAttr.getValue().getSExtValue();
    }
    return success();
  };

  SmallVector<int64_t> kernelShape, strides, dilations, pads;
  if (failed(readInts("strides", 2, 1, strides)) ||
      failed(readInts("dilations", 2, 1, dilations)))
    return rewriter.notifyMatchFailure(
        op, "NhwcConv strides/dilations must be 2-D");
  if (padMode == "VALID")
    pads.assign(4, 0);
  else if (failed(readInts("pads", 4, 0, pads)))
    return rewriter.notifyMatchFailure(op,
                                       "NhwcConv pads must have 4 elements");

  auto kernelAttr = op->getAttrOfType<ArrayAttr>("kernel_shape");
  if (kernelAttr) {
    if (failed(readInts("kernel_shape", 2, 1, kernelShape)))
      return rewriter.notifyMatchFailure(op,
                                         "NhwcConv kernel_shape must be 2-D");
  } else if (weightType.isDynamicDim(1) || weightType.isDynamicDim(2)) {
    return rewriter.notifyMatchFailure(
        op,
        "NhwcConv kernel_shape is missing and weight spatial dims are dynamic");
  } else {
    kernelShape = {weightType.getDimSize(1), weightType.getDimSize(2)};
  }
  if (llvm::any_of(kernelShape, [](int64_t k) { return k <= 0; }) ||
      llvm::any_of(strides, [](int64_t s) { return s <= 0; }) ||
      llvm::any_of(dilations, [](int64_t d) { return d <= 0; }))
    return rewriter.notifyMatchFailure(
        op, "NhwcConv kernel, stride, and dilation must be positive");

  int64_t group = 1;
  if (auto groupAttr = op->getAttrOfType<IntegerAttr>("group"))
    group = groupAttr.getValue().getSExtValue();
  if (group <= 0)
    return rewriter.notifyMatchFailure(op, "NhwcConv group must be positive");

  Type elem = inputType.getElementType();
  // NHWC [N, H, W, C] -> NCHW [N, C, H, W].
  SmallVector<int64_t> nchwShape = {
      inputType.getDimSize(0), inputType.getDimSize(3), inputType.getDimSize(1),
      inputType.getDimSize(2)};
  SmallVector<Value> inputDyn;
  if (inputType.isDynamicDim(0))
    inputDyn.push_back(dimOf(rewriter, loc, input, 0));
  if (inputType.isDynamicDim(3))
    inputDyn.push_back(dimOf(rewriter, loc, input, 3));
  if (inputType.isDynamicDim(1))
    inputDyn.push_back(dimOf(rewriter, loc, input, 1));
  if (inputType.isDynamicDim(2))
    inputDyn.push_back(dimOf(rewriter, loc, input, 2));
  Value inputNchw = transposeRank4(rewriter, loc, context, input,
                                   RankedTensorType::get(nchwShape, elem),
                                   {0, 3, 1, 2}, inputDyn);

  // OHWI [M, kH, kW, C/group] -> OIHW [M, C/group, kH, kW].
  SmallVector<int64_t> oihwShape = {
      weightType.getDimSize(0), weightType.getDimSize(3),
      weightType.getDimSize(1), weightType.getDimSize(2)};
  SmallVector<Value> weightDyn;
  if (weightType.isDynamicDim(0))
    weightDyn.push_back(dimOf(rewriter, loc, weights, 0));
  if (weightType.isDynamicDim(3))
    weightDyn.push_back(dimOf(rewriter, loc, weights, 3));
  if (weightType.isDynamicDim(1))
    weightDyn.push_back(dimOf(rewriter, loc, weights, 1));
  if (weightType.isDynamicDim(2))
    weightDyn.push_back(dimOf(rewriter, loc, weights, 2));
  Value weightOihw = transposeRank4(rewriter, loc, context, weights,
                                    RankedTensorType::get(oihwShape, elem),
                                    {0, 3, 1, 2}, weightDyn);

  // NHWC result [N, Ho, Wo, M] -> NCHW [N, M, Ho, Wo].
  SmallVector<int64_t> convShape = {
      resultType.getDimSize(0), resultType.getDimSize(3),
      resultType.getDimSize(1), resultType.getDimSize(2)};
  auto convType = RankedTensorType::get(convShape, elem);
  SmallVector<Value> spatialOut(2);
  for (int axis : {0, 1}) {
    bool needed =
        convType.isDynamicDim(axis + 2) || resultType.isDynamicDim(axis + 1);
    if (!needed)
      continue;
    spatialOut[axis] = convOutExtent(
        rewriter, loc, dimOf(rewriter, loc, input, axis + 1), kernelShape[axis],
        strides[axis], dilations[axis], pads[axis], pads[2 + axis]);
  }
  SmallVector<Value> convDyn;
  if (convType.isDynamicDim(0))
    convDyn.push_back(dimOf(rewriter, loc, input, 0));
  if (convType.isDynamicDim(1))
    convDyn.push_back(dimOf(rewriter, loc, weights, 0));
  if (convType.isDynamicDim(2))
    convDyn.push_back(spatialOut[0]);
  if (convType.isDynamicDim(3))
    convDyn.push_back(spatialOut[1]);
  Value convInit = tensor::EmptyOp::create(rewriter, loc, convType.getShape(),
                                           elem, convDyn);

  SmallVector<Value> operands = {context, inputNchw, weightOihw};
  if (bias)
    operands.push_back(bias);
  operands.push_back(convInit);
  SmallVector<NamedAttribute> attrs = {
      rewriter.getNamedAttr("kernel_shape",
                            rewriter.getI64ArrayAttr(kernelShape)),
      rewriter.getNamedAttr("strides", rewriter.getI64ArrayAttr(strides)),
      rewriter.getNamedAttr("pads", rewriter.getI64ArrayAttr(pads)),
      rewriter.getNamedAttr("dilations", rewriter.getI64ArrayAttr(dilations)),
      rewriter.getNamedAttr("group", rewriter.getI64IntegerAttr(group)),
  };
  Value convResult =
      ConvOp::create(rewriter, loc, operands, attrs)->getResult(0);

  SmallVector<Value> outDyn;
  if (resultType.isDynamicDim(0))
    outDyn.push_back(dimOf(rewriter, loc, input, 0));
  if (resultType.isDynamicDim(1))
    outDyn.push_back(spatialOut[0]);
  if (resultType.isDynamicDim(2))
    outDyn.push_back(spatialOut[1]);
  if (resultType.isDynamicDim(3))
    outDyn.push_back(dimOf(rewriter, loc, weights, 0));
  Value output = transposeRank4(rewriter, loc, context, convResult, resultType,
                                {0, 2, 3, 1}, outDyn);
  rewriter.replaceOp(op, output);
  return success();
}

} // namespace

void mlir::hip::populateNhwcConvConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {
  patterns.add<NhwcConvToHip>(ctx);
}
