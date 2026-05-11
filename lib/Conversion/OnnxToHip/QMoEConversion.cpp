//===- QMoEConversion.cpp - ONNX-to-HIP QMoE conversion ------- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include <limits>

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX QMoE -> HIP QMoE (com.microsoft custom op)
//===----------------------------------------------------------------------===//

struct QMoEToHip : public RewritePattern {
  QMoEToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult QMoEToHip::matchAndRewrite(Operation *op,
                                         PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "QMoE") {
    return rewriter.notifyMatchFailure(op, "not a QMoE custom op");
  }
  auto domainAttr = op->getAttrOfType<StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(op, "not a com.microsoft domain op");
  }

  Location loc = op->getLoc();

  if (op->getNumOperands() < 7) {
    return rewriter.notifyMatchFailure(op,
                                       "expected at least 7 inputs for QMoE");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for QMoE");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  Value context = *ctxOrFailure;

  auto getOptionalInput = [&](unsigned idx) -> Value {
    if (idx >= op->getNumOperands()) {
      return Value{};
    }
    Value v = op->getOperand(idx);
    if (!v || isa<NoneType>(v.getType())) {
      return Value{};
    }
    return v;
  };

  Value input = op->getOperand(0);
  Value routerProbs = op->getOperand(1);
  Value fc1Weights = op->getOperand(2);
  Value fc1Scales = op->getOperand(3);
  Value fc1Bias = getOptionalInput(4);
  Value fc2Weights = op->getOperand(5);
  Value fc2Scales = op->getOperand(6);
  Value fc2Bias = getOptionalInput(7);
  Value fc3Weights = getOptionalInput(8);
  Value fc3Scales = getOptionalInput(9);
  Value fc3Bias = getOptionalInput(10);
  Value fc1ZeroPoints = getOptionalInput(11);
  Value fc2ZeroPoints = getOptionalInput(12);
  Value fc3ZeroPoints = getOptionalInput(13);
  Value routerWeights = getOptionalInput(14); // ONNX v1.25+ router_weights

  auto expertWeightBitsIntAttr =
      op->getAttrOfType<IntegerAttr>("expert_weight_bits");
  auto expertWeightBitsAttr = rewriter.getI64IntegerAttr(
      expertWeightBitsIntAttr ? expertWeightBitsIntAttr.getSInt() : 4);

  auto kIntAttr = op->getAttrOfType<IntegerAttr>("k");
  auto kAttr = rewriter.getI64IntegerAttr(kIntAttr ? kIntAttr.getSInt() : 1);

  // ms.QMoE's `block_size` attribute is documented as optional (no spec
  // default) and is omitted by some quantization tools (e.g. AWQ exports of
  // gpt-oss-120b). Without it `wrap_qmoe` later divides by zero. When the
  // attribute is absent or non-positive, derive block_size from the
  // FC1 (gate_up_proj) scales tensor: scales has shape
  //     [num_experts, output_features, k_blocks_fc1]
  // with k_blocks_fc1 = ceil(hidden_size / block_size) and the activation
  // input has shape [..., hidden_size]. So
  //     block_size = ceil(hidden_size / k_blocks_fc1)
  // which gives the exact value when the model uses an even split (the
  // common case).
  auto blockSizeIntAttr = op->getAttrOfType<IntegerAttr>("block_size");
  int64_t blockSizeValue = blockSizeIntAttr ? blockSizeIntAttr.getSInt() : 0;
  if (blockSizeValue <= 0) {
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto scalesType = dyn_cast<RankedTensorType>(fc1Scales.getType());
    if (inputType && scalesType && inputType.getRank() > 0 &&
        scalesType.getRank() > 0) {
      int64_t hiddenDim = inputType.getShape().back();
      int64_t kBlocksDim = scalesType.getShape().back();
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
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSizeValue);

  auto normIntAttr =
      op->getAttrOfType<IntegerAttr>("normalize_routing_weights");
  auto normalizeAttr =
      rewriter.getI64IntegerAttr(normIntAttr ? normIntAttr.getSInt() : 0);

  auto swigluFusionIntAttr = op->getAttrOfType<IntegerAttr>("swiglu_fusion");
  auto swigluFusionAttr = rewriter.getI64IntegerAttr(
      swigluFusionIntAttr ? swigluFusionIntAttr.getSInt() : 0);

  auto sparseIntAttr = op->getAttrOfType<IntegerAttr>("use_sparse_mixer");
  auto useSparseAttr =
      rewriter.getI64IntegerAttr(sparseIntAttr ? sparseIntAttr.getSInt() : 0);

  auto alphaFloatAttr = op->getAttrOfType<FloatAttr>("activation_alpha");
  auto activationAlphaAttr =
      alphaFloatAttr ? alphaFloatAttr
                     : rewriter.getF32FloatAttr(1.0f); // ONNX spec default: 1.0

  auto betaFloatAttr = op->getAttrOfType<FloatAttr>("activation_beta");
  auto activationBetaAttr =
      betaFloatAttr ? betaFloatAttr : rewriter.getF32FloatAttr(0.0f);

  auto limitFloatAttr = op->getAttrOfType<FloatAttr>("swiglu_limit");
  // ONNX spec: "It is infinite when limit is not provided"
  // Match ONNX Runtime: std::numeric_limits<float>::infinity()
  auto swigluLimitAttr =
      limitFloatAttr
          ? limitFloatAttr
          : rewriter.getF32FloatAttr(std::numeric_limits<float>::infinity());

  auto activationTypeStrAttr = op->getAttrOfType<StringAttr>("activation_type");
  auto activationTypeAttr = activationTypeStrAttr
                                ? activationTypeStrAttr
                                : rewriter.getStringAttr("relu");

  auto rt = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, rt, input);

  auto hipOp = mlir::hip::QMoEOp::create(
      rewriter, loc, TypeRange{rt}, context, input, routerProbs, fc1Weights,
      fc1Scales, fc2Weights, fc2Scales, fc1Bias, fc2Bias, fc3Weights, fc3Scales,
      fc3Bias, fc1ZeroPoints, fc2ZeroPoints, fc3ZeroPoints, routerWeights, init,
      expertWeightBitsAttr, kAttr, blockSizeAttr, normalizeAttr,
      swigluFusionAttr, useSparseAttr, activationAlphaAttr, activationBetaAttr,
      swigluLimitAttr, activationTypeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return success();
}

} // namespace

void mlir::hip::populateQMoEConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<QMoEToHip>(ctx);
}

} // namespace hip
} // namespace mlir
