/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include <limits>

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX QMoE -> HIP QMoE (com.microsoft custom op)
//===----------------------------------------------------------------------===//

struct QMoEToHip : public mlir::RewritePattern {
  QMoEToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
QMoEToHip::matchAndRewrite(mlir::Operation *op,
                           mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "QMoE") {
    return rewriter.notifyMatchFailure(op, "not a QMoE custom op");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  }

  mlir::Location loc = op->getLoc();

  if (op->getNumOperands() < 7) {
    return rewriter.notifyMatchFailure(op,
                                       "expected at least 7 inputs for QMoE");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for QMoE");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  mlir::Value context = *ctxOrFailure;

  auto getOptionalInput = [&](unsigned idx) -> mlir::Value {
    if (idx >= op->getNumOperands()) {
      return mlir::Value{};
    }
    mlir::Value v = op->getOperand(idx);
    if (!v || mlir::isa<mlir::NoneType>(v.getType())) {
      return mlir::Value{};
    }
    return v;
  };

  mlir::Value input = op->getOperand(0);
  mlir::Value routerProbs = op->getOperand(1);
  mlir::Value fc1Weights = op->getOperand(2);
  mlir::Value fc1Scales = op->getOperand(3);
  mlir::Value fc1Bias = getOptionalInput(4);
  mlir::Value fc2Weights = op->getOperand(5);
  mlir::Value fc2Scales = op->getOperand(6);
  mlir::Value fc2Bias = getOptionalInput(7);
  mlir::Value fc3Weights = getOptionalInput(8);
  mlir::Value fc3Scales = getOptionalInput(9);
  mlir::Value fc3Bias = getOptionalInput(10);
  mlir::Value fc1ZeroPoints = getOptionalInput(11);
  mlir::Value fc2ZeroPoints = getOptionalInput(12);
  mlir::Value fc3ZeroPoints = getOptionalInput(13);
  mlir::Value routerWeights =
      getOptionalInput(14); // ONNX v1.25+ router_weights

  auto expertWeightBitsIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("expert_weight_bits");
  auto expertWeightBitsAttr = rewriter.getI64IntegerAttr(
      expertWeightBitsIntAttr ? expertWeightBitsIntAttr.getSInt() : 4);

  auto kIntAttr = op->getAttrOfType<mlir::IntegerAttr>("k");
  auto kAttr = rewriter.getI64IntegerAttr(kIntAttr ? kIntAttr.getSInt() : 1);

  // ms.QMoE's `block_size` attribute is documented as optional (no spec
  // default) and is omitted by some quantization tools (notably AWQ-style
  // exports of MoE models). Without it `wrap_qmoe` later divides by zero.
  // When the attribute is absent or non-positive, derive block_size from the
  // FC1 (gate_up_proj) scales tensor: scales has shape
  //     [num_experts, output_features, k_blocks_fc1]
  // with k_blocks_fc1 = ceil(hidden_size / block_size) and the activation
  // input has shape [..., hidden_size]. So
  //     block_size = ceil(hidden_size / k_blocks_fc1)
  // which gives the exact value when the model uses an even split (the
  // common case).
  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  int64_t blockSizeValue = blockSizeIntAttr ? blockSizeIntAttr.getSInt() : 0;
  if (blockSizeValue <= 0) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto scalesType =
        mlir::dyn_cast<mlir::RankedTensorType>(fc1Scales.getType());
    if (inputType && scalesType && inputType.getRank() > 0 &&
        scalesType.getRank() > 0 &&
        !inputType.isDynamicDim(inputType.getRank() - 1) &&
        !scalesType.isDynamicDim(scalesType.getRank() - 1)) {
      int64_t hiddenDim = inputType.getDimSize(inputType.getRank() - 1);
      int64_t kBlocksDim = scalesType.getDimSize(scalesType.getRank() - 1);
      if (hiddenDim > 0 && kBlocksDim > 0) {
        blockSizeValue = (hiddenDim + kBlocksDim - 1) / kBlocksDim;
      }
    }
  }
  if (blockSizeValue <= 0) {
    return rewriter.notifyMatchFailure(
        op, "QMoE: missing `block_size` attribute and cannot infer it from "
            "input/fc1_scales tensor shapes");
  }
  if (blockSizeValue < 32 || (blockSizeValue & (blockSizeValue - 1)) != 0) {
    return rewriter.notifyMatchFailure(
        op, "QMoE: block_size must be a power of two and at least 32");
  }
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSizeValue);

  auto normIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("normalize_routing_weights");
  auto normalizeAttr =
      rewriter.getI64IntegerAttr(normIntAttr ? normIntAttr.getSInt() : 0);

  auto swigluFusionIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("swiglu_fusion");
  auto swigluFusionAttr = rewriter.getI64IntegerAttr(
      swigluFusionIntAttr ? swigluFusionIntAttr.getSInt() : 0);

  auto sparseIntAttr = op->getAttrOfType<mlir::IntegerAttr>("use_sparse_mixer");
  auto useSparseAttr =
      rewriter.getI64IntegerAttr(sparseIntAttr ? sparseIntAttr.getSInt() : 0);

  auto alphaFloatAttr = op->getAttrOfType<mlir::FloatAttr>("activation_alpha");
  auto activationAlphaAttr =
      alphaFloatAttr ? alphaFloatAttr
                     : rewriter.getF32FloatAttr(1.0f); // ONNX spec default: 1.0

  auto betaFloatAttr = op->getAttrOfType<mlir::FloatAttr>("activation_beta");
  auto activationBetaAttr =
      betaFloatAttr ? betaFloatAttr : rewriter.getF32FloatAttr(0.0f);

  auto limitFloatAttr = op->getAttrOfType<mlir::FloatAttr>("swiglu_limit");
  // ONNX spec: "It is infinite when limit is not provided"
  // Match ONNX Runtime: std::numeric_limits<float>::infinity()
  auto swigluLimitAttr =
      limitFloatAttr
          ? limitFloatAttr
          : rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity());

  auto activationTypeStrAttr =
      op->getAttrOfType<mlir::StringAttr>("activation_type");
  auto activationTypeAttr = activationTypeStrAttr
                                ? activationTypeStrAttr
                                : rewriter.getStringAttr("relu");

  // Runtime kernels support fp16 activations with 2-bit or 4-bit experts.
  if (!mlir::getElementTypeOrSelf(input.getType()).isF16() ||
      (expertWeightBitsAttr.getInt() != 2 &&
       expertWeightBitsAttr.getInt() != 4)) {
    return rewriter.notifyMatchFailure(
        op, "QMoE: only fp16 input with expert_weight_bits=2 or 4 is "
            "supported");
  }

  // wrap_qmoe rejects every other combination at runtime.
  const int64_t swigluFusion = swigluFusionAttr.getInt();
  if (swigluFusion != 1 &&
      !(swigluFusion == 0 && activationTypeAttr.getValue() == "relu2")) {
    return rewriter.notifyMatchFailure(
        op, "QMoE: only swiglu_fusion=1, or swiglu_fusion=0 with "
            "activation_type=relu2, is supported");
  }

  // weights_prepacked (ORT default -1): 0 is the schema [E, N, K/pack]
  // layout, and -1 / 1 are EP-defined. This EP's default layout is the schema
  // one, and it has no alternate layout.
  auto prepackedAttr =
      op->getAttrOfType<mlir::IntegerAttr>("weights_prepacked");
  const int64_t weightsPrepacked = prepackedAttr ? prepackedAttr.getSInt() : -1;
  if (weightsPrepacked != -1 && weightsPrepacked != 0) {
    return rewriter.notifyMatchFailure(
        op, "QMoE: weights_prepacked must be -1 or 0 (schema [E, N, K/pack] "
            "weights); no alternate prepacked layout is implemented");
  }

  const int64_t bits = expertWeightBitsAttr.getInt();
  auto validatePackedShape = [&](mlir::Value weight, mlir::Value zeroPoint,
                                 int64_t logicalK) {
    auto weightType = mlir::dyn_cast<mlir::RankedTensorType>(weight.getType());
    if (!weightType || weightType.getRank() != 3 || logicalK < 0)
      return true;
    const int64_t packedK = (logicalK * bits + 7) / 8;
    if (!weightType.isDynamicDim(2) && weightType.getDimSize(2) != packedK)
      return false;
    if (!zeroPoint)
      return true;
    auto zeroPointType =
        mlir::dyn_cast<mlir::RankedTensorType>(zeroPoint.getType());
    if (!zeroPointType || zeroPointType.getRank() != 3)
      return true;
    for (int64_t dim = 0; dim < 2; ++dim) {
      if (!weightType.isDynamicDim(dim) && !zeroPointType.isDynamicDim(dim) &&
          weightType.getDimSize(dim) != zeroPointType.getDimSize(dim))
        return false;
    }
    const int64_t groups = (logicalK + blockSizeValue - 1) / blockSizeValue;
    const int64_t packedGroups = (groups * bits + 7) / 8;
    return zeroPointType.isDynamicDim(2) ||
           zeroPointType.getDimSize(2) == packedGroups;
  };

  int64_t hidden = -1;
  if (auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      inputType && inputType.getRank() > 0 &&
      !inputType.isDynamicDim(inputType.getRank() - 1))
    hidden = inputType.getDimSize(inputType.getRank() - 1);

  int64_t intermediate = -1;
  if (auto fc1Type =
          mlir::dyn_cast<mlir::RankedTensorType>(fc1Weights.getType());
      fc1Type && fc1Type.getRank() == 3 && !fc1Type.isDynamicDim(1)) {
    const int64_t fusion = swigluFusion == 1 ? 2 : 1;
    if (fc1Type.getDimSize(1) % fusion == 0)
      intermediate = fc1Type.getDimSize(1) / fusion;
  }

  if (!validatePackedShape(fc1Weights, fc1ZeroPoints, hidden) ||
      !validatePackedShape(fc2Weights, fc2ZeroPoints, intermediate) ||
      (fc3Weights && !validatePackedShape(fc3Weights, fc3ZeroPoints, hidden))) {
    return rewriter.notifyMatchFailure(
        op, "QMoE: packed weight or zero-point shape does not match the "
            "quantization attributes");
  }

  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, input);

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp = mlir::hip::QMoEOp::create(
      rewriter, loc, context, input, routerProbs, fc1Weights, fc1Scales,
      fc2Weights, fc2Scales, fc1Bias, fc2Bias, fc3Weights, fc3Scales, fc3Bias,
      fc1ZeroPoints, fc2ZeroPoints, fc3ZeroPoints, routerWeights, init,
      expertWeightBitsAttr, kAttr, blockSizeAttr, normalizeAttr,
      swigluFusionAttr, useSparseAttr, activationAlphaAttr, activationBetaAttr,
      swigluLimitAttr, activationTypeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateQMoEConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<QMoEToHip>(ctx);
}

} // namespace hip
} // namespace mlir
