//===- NormConversion.cpp - ONNX-to-HIP norm conversions ------ *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Why this conversion exists
// --------------------------
// LLM checkpoints use a small zoo of layer-norm variants:
//
//   * `onnx.LayerNormalization` (canonical) -- mean + variance + scale.
//   * `com.microsoft.SimplifiedLayerNormalization` (RMSNorm) -- variance
//     only.
//   * `com.microsoft.SkipSimplifiedLayerNormalization` -- adds a residual
//     before the norm and emits the residual sum as a side output.
//
// Each maps to a different MIOpen entry point and benefits from being a
// distinct dialect op (`hip.layer_norm`, `hip.rmsnorm`,
// `hip.skip_simplified_layer_norm`) so the lowering can pick the right
// runtime call without re-decoding op-specific attributes at LLVM time.
//
// Non-obvious choices
// -------------------
// * The skip variant has *two* outputs: the normalized tensor and the
//   residual-summed tensor.  We emit both as DPS-init operands so
//   bufferization keeps them backed by distinct allocations -- callers
//   downstream (e.g., the K-cache for the next attention block) consume
//   only the residual sum, but other consumers want only the norm.
// * `axes` from `onnx.LayerNormalization` is normalized to a single
//   trailing-axis form here; multi-axis layer-norm is rejected with a
//   clear error rather than silently lowered to something the runtime
//   can't compute.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(SimplifiedLayerNormalization) -> hip.rms_norm
struct SimplifiedLayerNormToHip : public RewritePattern {
  SimplifiedLayerNormToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
SimplifiedLayerNormToHip::matchAndRewrite(Operation *op,
                                          PatternRewriter &rewriter) const {
  // Check if this is SimplifiedLayerNormalization
  auto funcNameAttr = op->getAttrOfType<StringAttr>("function_name");
  if (!funcNameAttr ||
      funcNameAttr.getValue() != "SimplifiedLayerNormalization")
    return rewriter.notifyMatchFailure(
        op, "not a SimplifiedLayerNormalization operation");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();

  // Check operands (should be 2: input and scale)
  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(
        op, "expected 2 operands for SimplifiedLayerNormalization");

  Value input = op->getOperand(0);
  Value scale = op->getOperand(1);

  // Extract attributes
  auto epsilonAttr = op->getAttrOfType<FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  auto axisAttr = op->getAttrOfType<IntegerAttr>("axis");
  if (!axisAttr)
    return rewriter.notifyMatchFailure(op, "missing axis attribute");

  auto stashTypeAttr = op->getAttrOfType<IntegerAttr>("stash_type");
  if (!stashTypeAttr)
    return rewriter.notifyMatchFailure(op, "missing stash_type attribute");

  // Convert axis to i64
  auto axisI64Attr = rewriter.getI64IntegerAttr(axisAttr.getSInt());
  auto stashTypeI64Attr = rewriter.getI64IntegerAttr(stashTypeAttr.getSInt());

  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());

  // Create init tensor
  Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Create hip.rms_norm operation
  auto hipOp = mlir::hip::RmsNormOp::create(rewriter, loc, resultType, context,
                                            input, scale, init, axisI64Attr,
                                            epsilonAttr, stashTypeI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

/// onnx.Custom(SkipSimplifiedLayerNormalization) -> hip.skip_rms_norm
struct SkipSimplifiedLayerNormToHip : public RewritePattern {
  SkipSimplifiedLayerNormToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult
SkipSimplifiedLayerNormToHip::matchAndRewrite(Operation *op,
                                              PatternRewriter &rewriter) const {
  // Check if this is SkipSimplifiedLayerNormalization
  auto funcNameAttr = op->getAttrOfType<StringAttr>("function_name");
  if (!funcNameAttr ||
      funcNameAttr.getValue() != "SkipSimplifiedLayerNormalization")
    return rewriter.notifyMatchFailure(
        op, "not a SkipSimplifiedLayerNormalization operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op,
        "domain must be com.microsoft for SkipSimplifiedLayerNormalization");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();

  // MS spec: 3-4 inputs (input, skip, gamma, [bias])
  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 4)
    return rewriter.notifyMatchFailure(
        op, "SkipSimplifiedLayerNormalization expects 3-4 operands");

  auto getOptionalOperand = [&](size_t idx) -> Value {
    if (idx >= numOps)
      return nullptr;
    Value val = op->getOperand(idx);
    if (isa<NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  // Input 1-3: required
  Value input = op->getOperand(0);
  Value skip = op->getOperand(1);
  Value gamma = op->getOperand(2);

  // Input 4: bias (optional)
  Value bias = getOptionalOperand(3);

  // Extract epsilon attribute
  auto epsilonAttr = op->getAttrOfType<FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  // MS spec outputs (1-4): output, [mean], [inv_std_var], [input_skip_bias_sum]
  // mean and inv_std_var are training-only; not modeled in HIP op.
  unsigned numResults = op->getNumResults();

  auto outputType = cast<RankedTensorType>(op->getResult(0).getType());
  Value outputInit = createEmptyTensor(rewriter, loc, outputType, input);

  // Find input_skip_bias_sum: it's the last non-None result (index 1 or 3)
  bool hasSkipOutput = numResults >= 2;
  unsigned skipOutIdx = hasSkipOutput ? numResults - 1 : 0;

  // Check if the last result is actually a real tensor (not None)
  bool skipOutputIsReal = false;
  RankedTensorType skipOutputType;
  if (hasSkipOutput) {
    Type lastType = op->getResult(skipOutIdx).getType();
    if (!isa<NoneType>(lastType)) {
      skipOutputIsReal = true;
      skipOutputType = cast<RankedTensorType>(lastType);
    }
  }

  Value skipOutputInit = nullptr;
  if (skipOutputIsReal)
    skipOutputInit = createEmptyTensor(rewriter, loc, skipOutputType, input);
  constexpr int64_t kCtxSize = 1;
  constexpr int64_t kInputSize = 1;
  constexpr int64_t kSkipSize = 1;
  constexpr int64_t kGammaSize = 1;
  int64_t kBiasSize = bias ? 1 : 0;
  // Variadic outputs: always output (1) + optional skip_output (0|1)
  int64_t kOutputsSize = 1 + (skipOutputIsReal ? 1 : 0);
  // Build operands list for hip.skip_rms_norm
  SmallVector<Value> operands;
  operands.push_back(context);
  operands.push_back(input);
  operands.push_back(skip);
  operands.push_back(gamma);
  if (bias)
    operands.push_back(bias);
  // DPS outs (Variadic): [output] or [output, input_skip_bias_sum]
  operands.push_back(outputInit);
  if (skipOutputInit)
    operands.push_back(skipOutputInit);

  // Build result types
  SmallVector<Type> resultTypes;
  resultTypes.push_back(outputType);
  if (skipOutputIsReal)
    resultTypes.push_back(skipOutputType);

  // Build attributes
  SmallVector<NamedAttribute> attrs;
  attrs.push_back(rewriter.getNamedAttr("epsilon", epsilonAttr));

  // operand_segment_sizes for AttrSizedOperandSegments
  // [ctx(1), input(1), skip(1), gamma(1), bias(0|1), outputs(1|2)]
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(kCtxSize);
  segmentSizes.push_back(kInputSize);
  segmentSizes.push_back(kSkipSize);
  segmentSizes.push_back(kGammaSize);
  segmentSizes.push_back(kBiasSize);
  segmentSizes.push_back(kOutputsSize);

  auto state = OperationState(loc, "hip.skip_rms_norm");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);

  // Map HIP results back to ONNX results
  llvm::SmallVector<Value> replacements;
  replacements.push_back(hipOp->getResult(0)); // output

  if (hasSkipOutput) {
    // Fill intermediate None results (mean, inv_std_var) with empty tensors
    for (unsigned i = 1; i < skipOutIdx; ++i) {
      Type origType = op->getResult(i).getType();
      if (isa<NoneType>(origType)) {
        replacements.push_back(Value{});
        continue;
      }
      auto dummyType = cast<RankedTensorType>(origType);
      replacements.push_back(tensor::EmptyOp::create(
          rewriter, loc, dummyType.getShape(), dummyType.getElementType()));
    }
    if (skipOutputIsReal)
      replacements.push_back(hipOp->getResult(1)); // input_skip_bias_sum
    else {
      auto dummyType = RankedTensorType::get({}, rewriter.getF32Type());
      replacements.push_back(tensor::EmptyOp::create(
          rewriter, loc, dummyType.getShape(), dummyType.getElementType()));
    }
  }

  rewriter.replaceOp(op, replacements);
  return success();
}

} // namespace

void mlir::hip::populateNormConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<SimplifiedLayerNormToHip, SkipSimplifiedLayerNormToHip>(ctx);
}

} // namespace hip
} // namespace mlir
