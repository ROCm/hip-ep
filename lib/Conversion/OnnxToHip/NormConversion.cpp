/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// Norm conversions (ONNX -> HIP dialect).
//
// All Norm-family operators live here so they share helpers and so the file
// layout makes it obvious where to plug in future variants (BatchNorm,
// GroupNorm, InstanceNorm, ...).
//
// Currently implemented:
//   - onnx.Custom(SimplifiedLayerNormalization)         -> hip.rms_norm
//   - onnx.RMSNormalization (standard, opset 23+)       -> hip.rms_norm
//   - onnx.Custom(SkipSimplifiedLayerNormalization)     -> hip.skip_rms_norm
//   - onnx.Custom(SkipLayerNormalization)               -> add + layer_norm
//   - onnx.LayerNormalization (standard, opset 17+)     -> hip.layer_norm
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Shared helpers
//===----------------------------------------------------------------------===//

/// Returns \p op's operand at \p idx if it exists and is not a NoneType,
/// otherwise nullptr. Useful for extracting ONNX optional inputs (which appear
/// either as missing operands or as values with `none` type).
inline mlir::Value getOptionalOperand(mlir::Operation *op, size_t idx) {
  if (idx >= op->getNumOperands())
    return nullptr;
  mlir::Value v = op->getOperand(idx);
  if (mlir::isa<mlir::NoneType>(v.getType()))
    return nullptr;
  return v;
}

/// Returns \p op's result at \p idx if it exists and is not a NoneType,
/// otherwise nullptr. Mirror of getOptionalOperand for results.
inline mlir::Value getOptionalResult(mlir::Operation *op, unsigned idx) {
  if (idx >= op->getNumResults())
    return nullptr;
  mlir::Value v = op->getResult(idx);
  if (mlir::isa<mlir::NoneType>(v.getType()))
    return nullptr;
  return v;
}

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

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp =
      mlir::hip::RmsNormOp::create(rewriter, loc, context, input, scale, init,
                                   axisI64Attr, epsilonAttr, stashTypeI64Attr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

/// onnx.RMSNormalization -> hip.rms_norm
///
/// Standard ONNX RMSNormalization (opset 23+) is semantically identical to
/// com.microsoft SimplifiedLayerNormalization: RMS over the trailing axes
/// followed by a broadcast multiply with scale.
struct RMSNormalizationToHip : public mlir::RewritePattern {
  RMSNormalizationToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.RMSNormalization", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
RMSNormalizationToHip::matchAndRewrite(mlir::Operation *op,
                                       mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() != 2)
    return rewriter.notifyMatchFailure(
        op, "expected 2 operands for RMSNormalization");

  mlir::Value input = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);

  int64_t axis = -1;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();

  llvm::APFloat epsValue(9.99999974E-6f);
  if (auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon"))
    epsValue = epsilonAttr.getValue();

  int64_t stashType = 1;
  if (auto stashTypeAttr = op->getAttrOfType<mlir::IntegerAttr>("stash_type"))
    stashType = stashTypeAttr.getSInt();

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, input);

  auto hipOp = mlir::hip::RmsNormOp::create(
      rewriter, loc, context, input, scale, init,
      rewriter.getI64IntegerAttr(axis),
      rewriter.getF32FloatAttr(epsValue.convertToFloat()),
      rewriter.getI64IntegerAttr(stashType));

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

  // Input 1-3: required
  mlir::Value input = op->getOperand(0);
  mlir::Value skip = op->getOperand(1);
  mlir::Value gamma = op->getOperand(2);

  // Input 4: bias (optional)
  mlir::Value bias = getOptionalOperand(op, 3);

  // Extract epsilon attribute
  auto epsilonAttr = op->getAttrOfType<mlir::FloatAttr>("epsilon");
  if (!epsilonAttr)
    return rewriter.notifyMatchFailure(op, "missing epsilon attribute");

  // MS outputs are positional: output, mean, inv_std_var,
  // input_skip_bias_sum. The HIP runtime is inference-only, so real training
  // stats are unsupported and must not be mistaken for the residual output.
  unsigned numResults = op->getNumResults();
  if (numResults < 1 || numResults > 4)
    return rewriter.notifyMatchFailure(
        op, "SkipSimplifiedLayerNormalization expects 1-4 results");
  for (unsigned index : {1u, 2u})
    if (index < numResults &&
        !mlir::isa<mlir::NoneType>(op->getResult(index).getType()))
      return rewriter.notifyMatchFailure(
          op, "training mean/inv_std_var outputs are not supported");

  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  auto skipType = mlir::dyn_cast<mlir::RankedTensorType>(skip.getType());
  auto gammaType = mlir::dyn_cast<mlir::RankedTensorType>(gamma.getType());
  auto biasType = bias ? mlir::dyn_cast<mlir::RankedTensorType>(bias.getType())
                       : mlir::RankedTensorType{};
  if (!outputType || !inputType || !skipType || !gammaType ||
      (bias && !biasType))
    return rewriter.notifyMatchFailure(
        op, "SkipSimplifiedLayerNormalization requires ranked tensors");

  bool skipOutputIsReal =
      numResults == 4 && !mlir::isa<mlir::NoneType>(op->getResult(3).getType());
  mlir::RankedTensorType skipOutputType;
  if (skipOutputIsReal) {
    skipOutputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(3).getType());
    if (!skipOutputType)
      return rewriter.notifyMatchFailure(
          op, "input_skip_bias_sum must be a ranked tensor");
  }
  unsigned hipOutputCount = skipOutputIsReal ? 2 : 1;

  std::optional<llvm::ArrayRef<int64_t>> biasShape;
  if (biasType)
    biasShape = biasType.getShape();
  auto staticShapes = mlir::hip::inferSkipRmsNormOutputShapes(
      inputType.getShape(), skipType.getShape(), gammaType.getShape(),
      biasShape, hipOutputCount, [&]() { return op->emitError(); });
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
      (skipOutputIsReal &&
       !importedTypeAgrees(skipOutputType, (*staticShapes)[1])))
    return rewriter.notifyMatchFailure(
        op, "SkipSimplifiedLayerNormalization outputs must match input shape");

  auto resultShapes = mlir::hip::reifySkipRmsNormOutputShapes(
      rewriter, loc, input, skip, gamma, bias, hipOutputCount,
      [&]() { return op->emitError(); });
  if (mlir::failed(resultShapes))
    return mlir::failure();
  auto outputInit = createEmptyTensorFromReifiedShape(rewriter, loc, outputType,
                                                      (*resultShapes)[0]);
  if (mlir::failed(outputInit))
    return rewriter.notifyMatchFailure(op, "output shape is not reifiable");
  mlir::Value skipOutputInit;
  if (skipOutputIsReal) {
    auto init = createEmptyTensorFromReifiedShape(rewriter, loc, skipOutputType,
                                                  (*resultShapes)[1]);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "input_skip_bias_sum shape is not reifiable");
    skipOutputInit = *init;
  }
  constexpr int64_t kCtxSize = 1;
  constexpr int64_t kInputSize = 1;
  constexpr int64_t kSkipSize = 1;
  constexpr int64_t kGammaSize = 1;
  int64_t kBiasSize = bias ? 1 : 0;
  // Variadic outputs: always output (1) + optional skip_output (0|1)
  int64_t kOutputsSize = hipOutputCount;
  // Build operands list for hip.skip_rms_norm
  mlir::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(input);
  operands.push_back(skip);
  operands.push_back(gamma);
  if (bias)
    operands.push_back(bias);
  // DPS outs (Variadic): [output] or [output, input_skip_bias_sum]
  operands.push_back(*outputInit);
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

  for (unsigned index = 1; index < numResults; ++index)
    replacements.push_back(index == 3 && skipOutputIsReal ? hipOp->getResult(1)
                                                          : mlir::Value{});

  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// onnx.Custom(SkipLayerNormalization) -> hip.add + hip.layer_norm
//===----------------------------------------------------------------------===//

/// com.microsoft.SkipLayerNormalization -> hip.add + hip.layer_norm.
///
/// This is STANDARD (mean-subtracting) LayerNorm with a skip/residual add and
/// bias, distinct from SkipSimplifiedLayerNormalization (which is RMS norm and
/// maps to hip.skip_rms_norm). No fused hip op exists for the standard-LN skip
/// variant, so we compose:
///   sum    = input + skip               (hip.add)
///   output = LayerNorm(sum, gamma, beta, epsilon)   (hip.layer_norm)
///   output[3] (input_skip_bias_sum) = sum           (the pre-norm residual)
///
/// MS spec (com.microsoft.SkipLayerNormalization):
///   inputs (3-5): input, skip, gamma, [beta], [bias-on-input].
///                 Whisper uses exactly 4 (input, skip, gamma, beta) with no
///                 5th input-bias, so input_skip_bias_sum = input + skip.
///   outputs (1-4): output, [mean], [inv_std_var], [input_skip_bias_sum].
///                  Whisper consumes output[0] and output[3]; mean/inv_std are
///                  emitted as None and never materialized.
///
/// We emit hip.* ops DIRECTLY (not onnx.*): the greedy driver applies these
/// patterns with ExistingOps strictness, so a freshly-emitted onnx.* op would
/// not be revisited for lowering (see AttentionConversion.cpp:174-178 and
/// MultiHeadAttentionConversion.cpp:40 for the same gotcha).
struct SkipLayerNormToHip : public mlir::RewritePattern {
  SkipLayerNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
SkipLayerNormToHip::matchAndRewrite(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
  // Match com.microsoft.SkipLayerNormalization (NOT the Simplified variant,
  // which is handled separately and maps to hip.skip_rms_norm).
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "SkipLayerNormalization")
    return rewriter.notifyMatchFailure(
        op, "not a SkipLayerNormalization operation");

  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft")
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for SkipLayerNormalization");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  // MS spec: 3-5 inputs (input, skip, gamma, [beta], [input-bias]).
  size_t numOps = op->getNumOperands();
  if (numOps < 3 || numOps > 5)
    return rewriter.notifyMatchFailure(
        op, "SkipLayerNormalization expects 3-5 operands");

  mlir::Value input = op->getOperand(0);
  mlir::Value skip = op->getOperand(1);
  mlir::Value gamma = op->getOperand(2);
  // beta (output bias for the LayerNorm) is optional.
  mlir::Value beta = getOptionalOperand(op, 3);
  // 5th input is an optional bias added to `input` BEFORE the skip add (so it
  // also feeds input_skip_bias_sum). Whisper does not use it; reject if present
  // so we never silently drop it.
  mlir::Value inputBias = getOptionalOperand(op, 4);
  if (inputBias)
    return rewriter.notifyMatchFailure(
        op, "5-input SkipLayerNormalization (input-bias) not supported");

  // Epsilon (default 1e-5 per MS spec).
  llvm::APFloat epsValue(9.99999974E-6f);
  if (auto a = op->getAttrOfType<mlir::FloatAttr>("epsilon"))
    epsValue = a.getValue();

  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());

  // === 1. sum = input + skip (this is also output[3] input_skip_bias_sum) ===
  mlir::Value sumInit = createEmptyTensor(rewriter, loc, inputType, input);
  mlir::Value sum = mlir::hip::AddOp::create(rewriter, loc, inputType, context,
                                             input, skip, sumInit)
                        .getResult(0);

  // === 2. output = LayerNorm(sum, gamma, beta, epsilon) =====================
  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value outputInit =
      createEmptyTensor(rewriter, loc, outputType, input);

  // hip.layer_norm uses AttrSizedOperandSegments: [ctx, input, scale, bias?,
  // outputs*]. Whisper requests only output[0], so a single output buffer.
  llvm::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(sum);
  operands.push_back(gamma);
  if (beta)
    operands.push_back(beta);
  operands.push_back(outputInit);

  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(/*ctx=*/1);
  segmentSizes.push_back(/*input=*/1);
  segmentSizes.push_back(/*scale=*/1);
  segmentSizes.push_back(beta ? 1 : 0);
  segmentSizes.push_back(/*outputs=*/1);

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  // Standard ONNX LayerNorm defaults: axis=-1, stash_type=1 (fp32 reduction).
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(-1)));
  attrs.push_back(rewriter.getNamedAttr(
      "epsilon", rewriter.getF32FloatAttr(epsValue.convertToFloat())));
  attrs.push_back(
      rewriter.getNamedAttr("stash_type", rewriter.getI64IntegerAttr(1)));

  mlir::OperationState state(loc, "hip.layer_norm");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(outputType);
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));
  mlir::Operation *lnOp = rewriter.create(state);
  mlir::Value output = lnOp->getResult(0);

  // === 3. Map results: output[0] = LN output; output[1..2] (mean, inv_std)
  //        are None and dropped; output[3] (input_skip_bias_sum) = sum. =====
  unsigned numResults = op->getNumResults();
  llvm::SmallVector<mlir::Value> replacements;
  replacements.push_back(output); // output[0]
  for (unsigned i = 1; i < numResults; ++i) {
    mlir::Type origType = op->getResult(i).getType();
    if (mlir::isa<mlir::NoneType>(origType)) {
      replacements.push_back(mlir::Value{}); // mean / inv_std -> None
      continue;
    }
    // The last real (non-None) result is input_skip_bias_sum = input + skip.
    replacements.push_back(sum);
  }

  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// onnx.LayerNormalization (standard ONNX opset 17+) -> hip.layer_norm
//===----------------------------------------------------------------------===//

/// Standard ONNX LayerNormalization to hip.layer_norm.
///
/// ONNX spec (opset 17+):
///   inputs:  X (required), Scale (required), B (optional)
///   outputs: Y (required), Mean (optional), InvStdDev (optional)
///   attrs:   axis (default -1), epsilon (default 1e-5), stash_type (default 1)
///
/// Mean and InvStdDev are training-only outputs. We model them faithfully
/// (so frontends that legitimately request them keep round-tripping) but the
/// runtime is allowed to skip computing them when the corresponding output
/// tensor is omitted -- the lowering passes nullptr in that case.
struct LayerNormToHip : public mlir::RewritePattern {
  LayerNormToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.LayerNormalization", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
LayerNormToHip::matchAndRewrite(mlir::Operation *op,
                                mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();

  size_t numOps = op->getNumOperands();
  if (numOps < 2 || numOps > 3)
    return rewriter.notifyMatchFailure(
        op, "LayerNormalization expects 2 or 3 operands (X, Scale, [B])");

  // Required inputs.
  mlir::Value input = op->getOperand(0);
  mlir::Value scale = op->getOperand(1);
  // Optional bias.
  mlir::Value bias = getOptionalOperand(op, 2);

  // Attributes (all default-valued in the ONNX spec).
  int64_t axis = -1;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = a.getSInt();

  llvm::APFloat epsValue(9.99999974E-6f);
  if (auto a = op->getAttrOfType<mlir::FloatAttr>("epsilon"))
    epsValue = a.getValue();

  int64_t stashType = 1;
  if (auto a = op->getAttrOfType<mlir::IntegerAttr>("stash_type"))
    stashType = a.getSInt();

  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");

  // First output (Y) is required. Optional InvStdDev requires a real mean
  // scratch output because the HIP/runtime ABI uses contiguous [Y, Mean,
  // InvStdDev] slots even when ONNX leaves Mean as None.
  auto outputType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value meanResult = getOptionalResult(op, 1);
  mlir::Value invStdResult = getOptionalResult(op, 2);
  bool needsMean = meanResult || invStdResult;
  unsigned numHipOutputs = 1 + static_cast<unsigned>(needsMean) +
                           static_cast<unsigned>(bool(invStdResult));
  mlir::FailureOr<mlir::ReifiedRankedShapedTypeDims> outputShapes =
      mlir::hip::reifyLayerNormOutputShapes(rewriter, loc, input, axis,
                                            numHipOutputs);
  if (mlir::failed(outputShapes))
    return rewriter.notifyMatchFailure(
        op, "LayerNormalization axis/output shape is invalid");
  mlir::FailureOr<mlir::Type> statsElementType =
      mlir::hip::inferLayerNormStatsType(rewriter.getContext(), stashType);
  if (mlir::failed(statsElementType))
    return rewriter.notifyMatchFailure(
        op, "LayerNormalization runtime supports stash_type 0/1 or 10");

  mlir::FailureOr<mlir::Value> outputInit = createEmptyTensorFromReifiedShape(
      rewriter, loc, outputType, (*outputShapes)[0]);
  if (mlir::failed(outputInit))
    return rewriter.notifyMatchFailure(
        op, "LayerNormalization Y type contradicts input shape");

  mlir::Value meanInit, invStdInit;
  mlir::RankedTensorType meanType, invStdType;
  if (needsMean) {
    meanType = meanResult
                   ? mlir::cast<mlir::RankedTensorType>(meanResult.getType())
                   : getTensorTypeFromReifiedShape((*outputShapes)[1],
                                                   *statsElementType);
    if (meanType.getElementType() != *statsElementType)
      return rewriter.notifyMatchFailure(
          op, "LayerNormalization Mean dtype contradicts stash_type");
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, meanType, (*outputShapes)[1]);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "LayerNormalization Mean shape contradicts axis");
    meanInit = *init;
  }
  if (invStdResult) {
    invStdType = mlir::cast<mlir::RankedTensorType>(invStdResult.getType());
    if (invStdType.getElementType() != *statsElementType)
      return rewriter.notifyMatchFailure(
          op, "LayerNormalization InvStdDev dtype contradicts stash_type");
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, invStdType, (*outputShapes)[2]);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "LayerNormalization InvStdDev shape contradicts axis");
    invStdInit = *init;
  }

  // hip.layer_norm uses AttrSizedOperandSegments for [ctx, input, scale,
  // bias?, outputs*]. The Variadic outputs region must be contiguous, so we
  // collapse [output, mean?, inv_std?] into a single trailing run.
  llvm::SmallVector<mlir::Value> hipOutputs;
  llvm::SmallVector<mlir::Type> hipResultTypes;
  hipOutputs.push_back(*outputInit);
  hipResultTypes.push_back(outputType);
  if (meanInit) {
    hipOutputs.push_back(meanInit);
    hipResultTypes.push_back(meanType);
  }
  if (invStdInit) {
    hipOutputs.push_back(invStdInit);
    hipResultTypes.push_back(invStdType);
  }

  // Build operands list.
  llvm::SmallVector<mlir::Value> operands;
  operands.push_back(context);
  operands.push_back(input);
  operands.push_back(scale);
  if (bias)
    operands.push_back(bias);
  for (mlir::Value v : hipOutputs)
    operands.push_back(v);

  // Operand segment sizes for AttrSizedOperandSegments: [ctx, input, scale,
  // bias, outputs].
  llvm::SmallVector<int32_t> segmentSizes;
  segmentSizes.push_back(/*ctx=*/1);
  segmentSizes.push_back(/*input=*/1);
  segmentSizes.push_back(/*scale=*/1);
  segmentSizes.push_back(bias ? 1 : 0);
  segmentSizes.push_back(static_cast<int32_t>(hipOutputs.size()));

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("axis", rewriter.getI64IntegerAttr(axis)));
  attrs.push_back(rewriter.getNamedAttr(
      "epsilon", rewriter.getF32FloatAttr(epsValue.convertToFloat())));
  attrs.push_back(rewriter.getNamedAttr("stash_type",
                                        rewriter.getI64IntegerAttr(stashType)));

  mlir::OperationState state(loc, "hip.layer_norm");
  state.addOperands(operands);
  state.addAttributes(attrs);
  state.addTypes(hipResultTypes);
  state.addAttribute("operand_segment_sizes",
                     rewriter.getDenseI32ArrayAttr(segmentSizes));

  mlir::Operation *hipOp = rewriter.create(state);

  // Map HIP results back to the ONNX result list, preserving Mean / InvStdDev
  // None slots when they were not requested.
  llvm::SmallVector<mlir::Value> replacements;
  replacements.push_back(hipOp->getResult(/*Y=*/0));
  if (op->getNumResults() >= 2) {
    if (meanResult)
      replacements.push_back(hipOp->getResult(/*mean=*/1));
    else
      replacements.push_back(mlir::Value{}); // None -> drop
  }
  if (op->getNumResults() >= 3) {
    if (invStdResult)
      replacements.push_back(hipOp->getResult(/*inv_std=*/2));
    else
      replacements.push_back(mlir::Value{}); // None -> drop
  }

  rewriter.replaceOp(op, replacements);
  return mlir::success();
}

} // namespace

void populateNormConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns
      .add<SimplifiedLayerNormToHip, RMSNormalizationToHip,
           SkipSimplifiedLayerNormToHip, SkipLayerNormToHip, LayerNormToHip>(
          ctx);
}

} // namespace hip
} // namespace mlir
