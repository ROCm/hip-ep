/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.qmoe -> hip.qmoe
///
/// Custom torch op for quantized Mixture-of-Experts, injected during model
/// export. Bundles router, expert selection, quantized GEMM, and output
/// combination into a single fused op.
///
/// Signature:
///   %out = "torch.aten.qmoe"(%input, %router_probs,
///       %fc1_weights, %fc1_scales, %fc1_bias,
///       %fc2_weights, %fc2_scales, %fc2_bias,
///       %fc3_weights, %fc3_scales, %fc3_bias,
///       %fc1_zero_points, %fc2_zero_points, %fc3_zero_points)
///     attrs: expert_weight_bits, k, block_size, normalize_routing_weights,
///            swiglu_fusion, use_sparse_mixer, activation_alpha,
///            activation_beta, swiglu_limit, activation_type
struct TorchQMoEToHip : public mlir::RewritePattern {
  TorchQMoEToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.qmoe", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    if (op->getNumOperands() < 7)
      return rewriter.notifyMatchFailure(
          op, "expected at least 7 operands for QMoE");
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 result for QMoE");

    auto getOpt = [&](unsigned idx) -> mlir::Value {
      if (idx >= op->getNumOperands())
        return {};
      mlir::Value v = op->getOperand(idx);
      if (isTorchNone(v))
        return {};
      return v;
    };

    mlir::Value input = op->getOperand(0);
    mlir::Value routerProbs = op->getOperand(1);
    mlir::Value fc1Weights = op->getOperand(2);
    mlir::Value fc1Scales = op->getOperand(3);
    mlir::Value fc1Bias = getOpt(4);
    mlir::Value fc2Weights = op->getOperand(5);
    mlir::Value fc2Scales = op->getOperand(6);
    mlir::Value fc2Bias = getOpt(7);
    mlir::Value fc3Weights = getOpt(8);
    mlir::Value fc3Scales = getOpt(9);
    mlir::Value fc3Bias = getOpt(10);
    mlir::Value fc1ZeroPoints = getOpt(11);
    mlir::Value fc2ZeroPoints = getOpt(12);
    mlir::Value fc3ZeroPoints = getOpt(13);

    // Extract attributes with defaults matching the ONNX QMoE spec
    auto getI64 = [&](const char *name, int64_t def) -> mlir::IntegerAttr {
      auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
      return attr ? rewriter.getI64IntegerAttr(attr.getValue().getSExtValue())
                  : rewriter.getI64IntegerAttr(def);
    };
    auto getF32 = [&](const char *name, float def) -> mlir::FloatAttr {
      auto attr = op->getAttrOfType<mlir::FloatAttr>(name);
      return attr ? attr : rewriter.getF32FloatAttr(def);
    };
    auto getStr = [&](const char *name,
                      const char *def) -> mlir::StringAttr {
      auto attr = op->getAttrOfType<mlir::StringAttr>(name);
      return attr ? attr : rewriter.getStringAttr(def);
    };

    auto expertWeightBitsAttr = getI64("expert_weight_bits", 4);
    auto kAttr = getI64("k", 1);
    auto blockSizeAttr = getI64("block_size", 32);
    auto normalizeAttr = getI64("normalize_routing_weights", 0);
    auto swigluFusionAttr = getI64("swiglu_fusion", 0);
    auto useSparseAttr = getI64("use_sparse_mixer", 0);
    auto activationAlphaAttr = getF32("activation_alpha", 0.0f);
    auto activationBetaAttr = getF32("activation_beta", 0.0f);
    auto swigluLimitAttr = getF32("swiglu_limit", 0.0f);
    auto activationTypeAttr = getStr("activation_type", "relu");

    auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init = createEmptyTensorForTorch(rewriter, loc, rt, input);

    auto hipOp = mlir::hip::QMoEOp::create(
        rewriter, loc, mlir::TypeRange{rt}, context, input, routerProbs,
        fc1Weights, fc1Scales, fc2Weights, fc2Scales, fc1Bias, fc2Bias,
        fc3Weights, fc3Scales, fc3Bias, fc1ZeroPoints, fc2ZeroPoints,
        fc3ZeroPoints, init, expertWeightBitsAttr, kAttr, blockSizeAttr,
        normalizeAttr, swigluFusionAttr, useSparseAttr, activationAlphaAttr,
        activationBetaAttr, swigluLimitAttr, activationTypeAttr);

    rewriter.replaceOp(op, hipOp->getResults());
    return mlir::success();
  }
};

} // namespace

void populateTorchQMoEConversionPatterns(mlir::RewritePatternSet &patterns,
                                          mlir::MLIRContext *ctx) {
  patterns.add<TorchQMoEToHip>(ctx);
}

} // namespace hip
} // namespace mlir
