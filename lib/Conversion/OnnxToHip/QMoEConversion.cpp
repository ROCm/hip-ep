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
  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  int64_t blockSizeValue = blockSizeIntAttr ? blockSizeIntAttr.getSInt() : 0;
  if (blockSizeValue <= 0) {
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto scalesType =
        mlir::dyn_cast<mlir::RankedTensorType>(fc1Scales.getType());
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

  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, input);

  auto hipOp = mlir::hip::QMoEOp::create(
      rewriter, loc, mlir::TypeRange{rt}, context, input, routerProbs,
      fc1Weights, fc1Scales, fc2Weights, fc2Scales, fc1Bias, fc2Bias,
      fc3Weights, fc3Scales, fc3Bias, fc1ZeroPoints, fc2ZeroPoints,
      fc3ZeroPoints, routerWeights, init, expertWeightBitsAttr, kAttr,
      blockSizeAttr, normalizeAttr, swigluFusionAttr, useSparseAttr,
      activationAlphaAttr, activationBetaAttr, swigluLimitAttr,
      activationTypeAttr);
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
