/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Custom(SimplifiedLayerNormalization) -> hip.rms_norm
struct SimplifiedLayerNormToHip : public mlir::RewritePattern {
  SimplifiedLayerNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult SimplifiedLayerNormToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  // Check if this is SimplifiedLayerNormalization
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr ||
      funcNameAttr.getValue() != "SimplifiedLayerNormalization")
    return rewriter.notifyMatchFailure(
        op, "not a SimplifiedLayerNormalization operation");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // Check operands (should be 2: input and scale)
  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(
        op, "expected 2 operands for SimplifiedLayerNormalization");

  mlir::Value input = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);

  // Extract attributes
  auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
  if (!axisAttr)
    return rewriter.notifyMatchFailure(op, "missing axis attribute");

  auto stashTypeAttr = op->getAttrOfType<mlir::IntegerAttr>("stash_type");
  if (!stashTypeAttr)
    return rewriter.notifyMatchFailure(op, "missing stash_type attribute");

  // Convert axis to i64
  auto axisI64Attr = rewriter.getI64IntegerAttr(axisAttr.getSInt());
  auto stashTypeI64Attr = rewriter.getI64IntegerAttr(stashTypeAttr.getSInt());

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

  // Create init tensor
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Create hip.rms_norm operation
  auto hipOp = mlir::hip::RmsNormOp::create(rewriter, loc, resultType, context,
                                            input, scale, init, axisI64Attr,
                                            epsilonAttr, stashTypeI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.Custom(SkipSimplifiedLayerNormalization) -> hip.skip_rms_norm
struct SkipSimplifiedLayerNormToHip : public mlir::RewritePattern {
  SkipSimplifiedLayerNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult SkipSimplifiedLayerNormToHip::matchAndRewrite(
    mlir::Operation *op, mlir::PatternRewriter &rewriter) const {
  // Check if this is SkipSimplifiedLayerNormalization
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr ||
      funcNameAttr.getValue() != "SkipSimplifiedLayerNormalization")
    return rewriter.notifyMatchFailure(
        op, "not a SkipSimplifiedLayerNormalization operation");

  // Check domain is "com.microsoft"
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op,
        "domain must be com.microsoft for SkipSimplifiedLayerNormalization");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // MS spec: 3-4 inputs (input, skip, gamma, [bias])
  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 4)
    return rewriter.notifyMatchFailure(
        op, "SkipSimplifiedLayerNormalization expects 3-4 operands");

  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps)
      return nullptr;
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType()))
      return nullptr;
    return val;
  };

  // Input 1-3: required
  mlir::Value input = op->getOperand(0);
  mlir::Value skip = op->getOperand(1);
  mlir::Value gamma = op->getOperand(2);

  // Input 4: bias (optional)
  mlir::Value bias = getOptionalOperand(3);

  // Extract epsilon attribute
  auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  // MS spec outputs (1-4): output, [mean], [inv_std_var], [input_skip_bias_sum]
  // mean and inv_std_var are training-only; not modeled in HIP op.
  unsigned numResults = op->getNumResults();

  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, input);

  // Find input_skip_bias_sum: it's the last non-None result (index 1 or 3)
  bool hasSkipOutput = numResults >= 2;
  unsigned skipOutIdx = hasSkipOutput ? numResults - 1 : 0;

  // Check if the last result is actually a real tensor (not None)
  bool skipOutputIsReal = false;
  mlir::RankedTensorType skipOutputType;
  if (hasSkipOutput) {
    mlir::Type lastType = op->getResult(skipOutIdx).getType();
    if (!mlir::isa<mlir::NoneType>(lastType)) {
      skipOutputIsReal = true;
      skipOutputType = mlir::cast<mlir::RankedTensorType>(lastType);
    }
  }

  mlir::Value skipOutputInit = nullptr;
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
  mlir::SmallVector<mlir::Value> operands;
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
  mlir::SmallVector<mlir::Type> resultTypes;
  resultTypes.push_back(outputType);
  if (skipOutputIsReal)
    resultTypes.push_back(skipOutputType);

  // Build attributes
  mlir::SmallVector<mlir::NamedAttribute> attrs;
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

  auto state = mlir::OperationState(loc, "hip.skip_rms_norm");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(resultTypes);
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  auto hipOp = rewriter.create(state);

  // Map HIP results back to ONNX results
  llvm::SmallVector<mlir::Value> replacements;
  replacements.push_back(hipOp->getResult(0)); // output

  if (hasSkipOutput) {
    // Fill intermediate None results (mean, inv_std_var) with empty tensors
    for (unsigned i = 1; i < skipOutIdx; ++i) {
      mlir::Type origType = op->getResult(i).getType();
      if (mlir::isa<mlir::NoneType>(origType)) {
        replacements.push_back(mlir::Value{});
        continue;
      }
      auto dummyType = mlir::cast<mlir::RankedTensorType>(origType);
      replacements.push_back(mlir::tensor::EmptyOp::create(
          rewriter, loc, dummyType.getShape(), dummyType.getElementType()));
    }
    if (skipOutputIsReal)
      replacements.push_back(hipOp->getResult(1)); // input_skip_bias_sum
    else {
      auto dummyType = mlir::RankedTensorType::get({}, rewriter.getF32Type());
      replacements.push_back(mlir::tensor::EmptyOp::create(
          rewriter, loc, dummyType.getShape(), dummyType.getElementType()));
    }
  }

  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

} // namespace

void populateNormConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<SimplifiedLayerNormToHip, SkipSimplifiedLayerNormToHip>(ctx);
}

} // namespace hip
} // namespace mlir
