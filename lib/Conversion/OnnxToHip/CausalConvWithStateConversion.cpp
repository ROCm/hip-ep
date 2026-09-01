/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// com.microsoft.CausalConvWithState -> hip.causal_conv_with_state
struct CausalConvWithStateToHip : public mlir::RewritePattern {
  CausalConvWithStateToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult CausalConvWithStateToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "CausalConvWithState")
    return rewriter.notifyMatchFailure(op,
                                       "not a CausalConvWithState operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for CausalConvWithState");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // CausalConvWithState has 2-4 inputs:
  //   input (required), weight (required), bias (optional), past_state
  //   (optional)
  size_t numOps = op->getNumOperands();
  if (numOps < 2 || numOps > 4)
    return rewriter.notifyMatchFailure(
        op, "CausalConvWithState expects 2-4 operands");

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  mlir::Value input = op->getOperand(0);
  mlir::Value weight = op->getOperand(1);
  mlir::Value bias = getOptionalOperand(2);
  mlir::Value pastState = getOptionalOperand(3);

  // Extract attributes with defaults
  auto activationAttr = op->getAttrOfType<mlir::StringAttr>("activation");
  if (!activationAttr)
    activationAttr = rewriter.getStringAttr("none");

  auto ndimAttrOnnx = op->getAttrOfType<mlir::IntegerAttr>("ndim");
  auto ndimAttr =
      ndimAttrOnnx
          ? rewriter.getI64IntegerAttr(ndimAttrOnnx.getValue().getSExtValue())
          : rewriter.getI64IntegerAttr(1);

  // Outputs: output (same shape as input), present_state [B,C,K-1].
  size_t numResults = op->getNumResults();
  if (numResults != 2)
    return rewriter.notifyMatchFailure(
        op, "CausalConvWithState expects exactly 2 results");

  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto presentStateType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
  auto biasType = bias ? mlir::dyn_cast<mlir::RankedTensorType>(bias.getType())
                       : mlir::RankedTensorType{};
  auto pastStateType =
      pastState ? mlir::dyn_cast<mlir::RankedTensorType>(pastState.getType())
                : mlir::RankedTensorType{};
  if (!outputType || !presentStateType || !inputType || !weightType ||
      (bias && !biasType) || (pastState && !pastStateType))
    return rewriter.notifyMatchFailure(
        op, "CausalConvWithState requires ranked tensor operands and results");

  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  if (biasType)
    biasShape = biasType.getShape();
  std::optional<llvm::ArrayRef<int64_t>> pastStateShape;
  if (pastStateType)
    pastStateShape = pastStateType.getShape();
  auto staticShapes = mlir::hip::inferCausalConvWithStateOutputShapes(
      inputType.getShape(), weightType.getShape(), biasShape, pastStateShape,
      ndimAttr.getInt(), /*channelsLast=*/false,
      [&]() { return op->emitError(); });
  if (mlir::failed(staticShapes))
    return mlir::failure();

  auto importedTypeAgrees = [](mlir::RankedTensorType imported,
                               llvm::ArrayRef<int64_t> expected) {
    if (imported.getRank() != static_cast<int64_t>(expected.size()))
      return false;
    for (int64_t dim : llvm::seq<int64_t>(0, imported.getRank()))
      if (!imported.isDynamicDim(dim) &&
          !mlir::ShapedType::isDynamic(expected[dim]) &&
          imported.getDimSize(dim) != expected[dim])
        return false;
    return true;
  };
  if (!importedTypeAgrees(outputType, (*staticShapes)[0]) ||
      !importedTypeAgrees(presentStateType, (*staticShapes)[1]))
    return rewriter.notifyMatchFailure(
        op, "CausalConvWithState imported result types disagree with "
            "[input, (B,C,K-1)]");

  // The same mixed helper backs op reification. It validates every shape
  // precondition before emitting dynamic dimension arithmetic.
  auto resultShapes = mlir::hip::reifyCausalConvWithStateOutputShapes(
      rewriter, loc, input, weight, bias, pastState, ndimAttr.getInt(),
      /*channelsLast=*/false, [&]() { return op->emitError(); });
  if (mlir::failed(resultShapes))
    return mlir::failure();
  auto outputInit = createEmptyTensorFromReifiedShape(rewriter, loc, outputType,
                                                      (*resultShapes)[0]);
  auto presentStateInit = createEmptyTensorFromReifiedShape(
      rewriter, loc, presentStateType, (*resultShapes)[1]);
  if (mlir::failed(outputInit) || mlir::failed(presentStateInit))
    return rewriter.notifyMatchFailure(
        op, "CausalConvWithState result types are not reifiable");

  // Build operands
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(input);
  operands.push_back(weight);
  if (bias)
    operands.push_back(bias);
  if (pastState)
    operands.push_back(pastState);
  operands.push_back(*outputInit);
  operands.push_back(*presentStateInit);

  // Build attributes
  mlir::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("activation", activationAttr));
  attrs.push_back(rewriter.getNamedAttr("ndim", ndimAttr));

  // Build result types
  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  resultTypes.push_back(presentStateType);

  // Create operation with operand_segment_sizes
  auto state = mlir::OperationState(loc, "hip.causal_conv_with_state");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);

  // Segments: [ctx(1), input(1), weight(1), bias(0|1), past_state(0|1),
  //            output(1), present_state(1)]
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(1); // ctx
  segmentSizes.push_back(1); // input
  segmentSizes.push_back(1); // weight
  segmentSizes.push_back(bias ? 1 : 0);
  segmentSizes.push_back(pastState ? 1 : 0);
  segmentSizes.push_back(1); // output
  segmentSizes.push_back(1); // present_state

  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Plain rank-3 depthwise causal onnx.Conv -> hip.causal_conv_with_state
//
// A depthwise 1D convolution with left-only padding IS a causal convolution
// whose carry state happens to be zero:
//
//   ONNX Conv(pads=[k-1, 0], stride 1, dilation 1, group=C, W=[C,1,k])
//     out[b,c,t] = sum_j W[c,0,j] * Xpad[b,c,t+j]
//     Xpad[i]    = 0 for i < k-1, else X[i-(k-1)]
//
// which is exactly the "virtual sequence" hip_causal_conv_prefill slides over
// with a null past_state (see causal_conv_step_kernel.hip). Routing these here
// buys two things over the generic hip.conv path: one fused launch instead of
// MIOpen conv + a separate bias miopenOpTensor, and eligibility for the
// canonicalizer in HipCausalConvCanonicalize.cpp, which absorbs the [0,2,1]
// Transpose pair that a channels-last exporter wraps around every one of these.
//
// The gemma-4 E2B/E4B audio encoders contain 12 of these each (k=5, C=1024).
//
// Benefit 2 beats ConvToHip's 1: ConvConversion.cpp accepts every rank-3 Conv,
// so without the higher benefit this pattern would never be tried.
//===----------------------------------------------------------------------===//
struct DepthwiseCausalConvToHip : public mlir::RewritePattern {
  DepthwiseCausalConvToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Conv", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

/// Read an i64 ArrayAttr into `out`. Returns false when the attribute exists
/// but is not an array of integers.
bool readIntArrayAttr(mlir::Operation *op, llvm::StringRef name,
                      llvm::SmallVectorImpl<int64_t> &out) {
  auto attr = op->getAttrOfType<mlir::ArrayAttr>(name);
  if (!attr)
    return true; // absent: caller applies the ONNX default
  for (mlir::Attribute a : attr) {
    auto i = mlir::dyn_cast<mlir::IntegerAttr>(a);
    if (!i)
      return false;
    out.push_back(i.getValue().getSExtValue());
  }
  return true;
}

mlir::LogicalResult DepthwiseCausalConvToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  // auto_pad rewrites the pads we are about to inspect, so only the explicit
  // form can be matched on.
  if (auto autoPad = op->getAttrOfType<mlir::StringAttr>("auto_pad"))
    if (autoPad.getValue() != "NOTSET")
      return rewriter.notifyMatchFailure(op, "causal_conv.auto_pad");

  auto inputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
  auto weightType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(1).getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !weightType || !resultType)
    return rewriter.notifyMatchFailure(op, "causal_conv.unranked");

  // 1D only: the fused kernel, and `channels_last`, are both ndim=1.
  if (inputType.getRank() != 3 || weightType.getRank() != 3 ||
      resultType.getRank() != 3)
    return rewriter.notifyMatchFailure(op, "causal_conv.not_rank3");

  // fp16/fp32 -- the element sizes wrap_causal_conv_with_state accepts.
  mlir::Type elemType = inputType.getElementType();
  if (!elemType.isF16() && !elemType.isF32())
    return rewriter.notifyMatchFailure(op, "causal_conv.dtype");
  if (weightType.getElementType() != elemType ||
      resultType.getElementType() != elemType)
    return rewriter.notifyMatchFailure(op, "causal_conv.mixed_dtype");

  // Depthwise: weight is [C, 1, k], and C must agree with both the input and
  // the output channel count. All three extents have to be static -- the
  // channel count and kernel width are kernel launch parameters.
  const int64_t channels = weightType.getDimSize(0);
  const int64_t kernel = weightType.getDimSize(2);
  if (channels == mlir::ShapedType::kDynamic ||
      kernel == mlir::ShapedType::kDynamic || weightType.getDimSize(1) != 1)
    return rewriter.notifyMatchFailure(op, "causal_conv.weight_not_depthwise");
  if (inputType.getDimSize(1) != channels ||
      resultType.getDimSize(1) != channels)
    return rewriter.notifyMatchFailure(op, "causal_conv.channel_mismatch");

  // k == 1 is excluded on purpose: the carry state is [B, C, k-1], so k == 1
  // would need a zero-extent buffer for a convolution that is only a
  // per-channel scale. The generic conv path handles it without that wrinkle.
  if (kernel < 2 || kernel > 8)
    return rewriter.notifyMatchFailure(op, "causal_conv.kernel_width");

  int64_t group = 1;
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("group"))
    group = attr.getValue().getSExtValue();
  if (group != channels)
    return rewriter.notifyMatchFailure(op,
                                       "causal_conv.not_grouped_by_channel");

  llvm::SmallVector<int64_t> kernelShape, strides, pads, dilations;
  if (!readIntArrayAttr(op, "kernel_shape", kernelShape) ||
      !readIntArrayAttr(op, "strides", strides) ||
      !readIntArrayAttr(op, "pads", pads) ||
      !readIntArrayAttr(op, "dilations", dilations))
    return rewriter.notifyMatchFailure(op, "causal_conv.bad_attr");

  if (!kernelShape.empty() &&
      (kernelShape.size() != 1 || kernelShape[0] != kernel))
    return rewriter.notifyMatchFailure(op, "causal_conv.kernel_shape_mismatch");
  if (!strides.empty() && (strides.size() != 1 || strides[0] != 1))
    return rewriter.notifyMatchFailure(op, "causal_conv.stride_ne_1");
  if (!dilations.empty() && (dilations.size() != 1 || dilations[0] != 1))
    return rewriter.notifyMatchFailure(op, "causal_conv.dilation_ne_1");
  // Left-only padding of exactly k-1 is what makes the convolution causal and
  // length-preserving. Any other padding is a different operation.
  if (pads.size() != 2 || pads[0] != kernel - 1 || pads[1] != 0)
    return rewriter.notifyMatchFailure(op, "causal_conv.pads_not_causal");

  // With stride 1 and pads [k-1, 0] the output length equals the input length.
  // When both are static, say so explicitly rather than trusting the importer.
  if (!inputType.isDynamicDim(2) && !resultType.isDynamicDim(2) &&
      inputType.getDimSize(2) != resultType.getDimSize(2))
    return rewriter.notifyMatchFailure(op, "causal_conv.length_not_preserved");
  if (!inputType.isDynamicDim(0) && !resultType.isDynamicDim(0) &&
      inputType.getDimSize(0) != resultType.getDimSize(0))
    return rewriter.notifyMatchFailure(op, "causal_conv.batch_mismatch");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "causal_conv.no_context");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value weight = op->getOperand(1);
  mlir::Value bias = nullptr;
  if (op->getNumOperands() > 2 &&
      !mlir::isa<mlir::NoneType>(op->getOperand(2).getType()))
    bias = op->getOperand(2);

  // Output shape matches the input positionally, so dynamic batch/length come
  // straight off the input rather than off this op's own (about to be replaced)
  // result.
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, resultType, input);

  // present_state is [B, C, k-1]. Nothing reads it here -- ONNX Conv is
  // stateless -- but the runtime requires the buffer, so materialise it and let
  // PoolAllocs fold it into the byte pool with everything else.
  llvm::SmallVector<int64_t> stateShape = {resultType.getDimSize(0), channels,
                                           kernel - 1};
  auto presentStateType = mlir::RankedTensorType::get(stateShape, elemType);
  llvm::SmallVector<mlir::Value> stateDynSizes;
  if (presentStateType.isDynamicDim(0))
    stateDynSizes.push_back(
        mlir::tensor::DimOp::create(rewriter, loc, input, /*index=*/0));
  mlir::Value presentStateInit = mlir::tensor::EmptyOp::create(
      rewriter, loc, presentStateType.getShape(), elemType, stateDynSizes);

  llvm::SmallVector<mlir::Value> operands = {context, input, weight};
  if (bias)
    operands.push_back(bias);
  operands.push_back(outputInit);
  operands.push_back(presentStateInit);

  // Segments: [ctx, input, weight, bias(0|1), past_state(0), output,
  //            present_state]. past_state is absent, which the kernel reads as
  //            a zero carry -- the definition of pads=[k-1, 0].
  llvm::SmallVector<int32_t> segmentSizes = {1, 1, 1, bias ? 1 : 0, 0, 1, 1};

  auto state = mlir::OperationState(loc, "hip.causal_conv_with_state");
  state.addOperands(operands);
  state.addAttribute("activation", rewriter.getStringAttr("none"));
  state.addAttribute("ndim", rewriter.getI64IntegerAttr(1));
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));
  state.addTypes({resultType, presentStateType});

  auto hipOp = rewriter.create(state);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateCausalConvWithStateConversionPatterns(RewritePatternSet &patterns,
                                                   MLIRContext *ctx) {
  patterns.add<CausalConvWithStateToHip>(ctx);
  patterns.add<DepthwiseCausalConvToHip>(ctx);
}

} // namespace hip
} // namespace mlir
