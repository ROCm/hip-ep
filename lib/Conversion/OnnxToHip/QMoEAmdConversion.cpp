/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX QMoE -> HIP QMoEAmd (com.amd custom op, Nemotron-H LatentMoE)
//===----------------------------------------------------------------------===//

struct QMoEAmdToHip : public mlir::RewritePattern {
  QMoEAmdToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
QMoEAmdToHip::matchAndRewrite(mlir::Operation *op,
                              mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "QMoE") {
    return rewriter.notifyMatchFailure(op, "not a QMoE custom op");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.amd") {
    return rewriter.notifyMatchFailure(op, "not a com.amd domain op");
  }

  mlir::Location loc = op->getLoc();

  // Fixed 15-input schema (see backend-mlir-compiler/custom-op-schema/ops/
  // qmoe_schema.cpp) -- unlike com.microsoft::QMoE, every input is required,
  // so there is no optional-operand handling here.
  if (op->getNumOperands() != 15) {
    return rewriter.notifyMatchFailure(
        op, "expected exactly 15 inputs for com.amd QMoE");
  }
  if (op->getNumResults() != 1) {
    return rewriter.notifyMatchFailure(op, "expected 1 output for QMoE");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "failed to get context argument");
  }
  mlir::Value context = *ctxOrFailure;

  mlir::Value hiddenStates = op->getOperand(0);
  mlir::Value fc1ExpertsWeights = op->getOperand(1);
  mlir::Value fc1ExpertsScales = op->getOperand(2);
  mlir::Value fc2ExpertsWeights = op->getOperand(3);
  mlir::Value fc2ExpertsScales = op->getOperand(4);
  mlir::Value fc1LatentWeights = op->getOperand(5);
  mlir::Value fc1LatentScales = op->getOperand(6);
  mlir::Value fc2LatentWeights = op->getOperand(7);
  mlir::Value fc2LatentScales = op->getOperand(8);
  mlir::Value sharedFc1Weights = op->getOperand(9);
  mlir::Value sharedFc1Scales = op->getOperand(10);
  mlir::Value sharedFc2Weights = op->getOperand(11);
  mlir::Value sharedFc2Scales = op->getOperand(12);
  mlir::Value routerWeight = op->getOperand(13);
  mlir::Value correctionBias = op->getOperand(14);

  // Attribute defaults mirror the fixed values documented for this model's
  // export (see CUSTOM_OP_SCHEMA.md shipped alongside the AMD Nemotron-3
  // QMoE artifact); nodes are still free to override any of them.
  auto kIntAttr = op->getAttrOfType<mlir::IntegerAttr>("k");
  auto kAttr = rewriter.getI64IntegerAttr(kIntAttr ? kIntAttr.getSInt() : 22);

  auto expertWeightBitsIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("expert_weight_bits");
  auto expertWeightBitsAttr = rewriter.getI64IntegerAttr(
      expertWeightBitsIntAttr ? expertWeightBitsIntAttr.getSInt() : 4);

  auto blockSizeIntAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
  int64_t blockSizeValue = blockSizeIntAttr ? blockSizeIntAttr.getSInt() : 32;
  if (blockSizeValue <= 0) {
    return rewriter.notifyMatchFailure(
        op, "QMoE (com.amd): invalid or missing block_size attribute");
  }
  auto blockSizeAttr = rewriter.getI64IntegerAttr(blockSizeValue);

  auto normIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("normalize_routing_weights");
  auto normalizeAttr =
      rewriter.getI64IntegerAttr(normIntAttr ? normIntAttr.getSInt() : 1);

  auto useCorrectionBiasIntAttr =
      op->getAttrOfType<mlir::IntegerAttr>("use_correction_bias");
  auto useCorrectionBiasAttr = rewriter.getI64IntegerAttr(
      useCorrectionBiasIntAttr ? useCorrectionBiasIntAttr.getSInt() : 1);

  auto scalingFloatAttr =
      op->getAttrOfType<mlir::FloatAttr>("routed_scaling_factor");
  auto routedScalingFactorAttr =
      scalingFloatAttr ? scalingFloatAttr : rewriter.getF32FloatAttr(1.0f);

  auto activationTypeStrAttr =
      op->getAttrOfType<mlir::StringAttr>("activation_type");
  auto activationTypeAttr = activationTypeStrAttr
                                ? activationTypeStrAttr
                                : rewriter.getStringAttr("relu2");

  auto routingTypeStrAttr = op->getAttrOfType<mlir::StringAttr>("routing_type");
  auto routingTypeAttr = routingTypeStrAttr ? routingTypeStrAttr
                                            : rewriter.getStringAttr("sigmoid");

  // The custom-op schema (backend-mlir-compiler/custom-op-schema/ops/
  // qmoe_schema.cpp) deliberately implements no InferOutputShape, so ORT's
  // Resolve() always leaves QMoE's result type UNRANKED (see that file's
  // comment for why: Ort::ShapeInferContext::SetOutputShape cannot encode
  // hidden_states' dynamic batch/sequence dim without hitting an upstream
  // ORT validation bug). An unchecked `mlir::cast<RankedTensorType>` here
  // would read a garbage rank from the unranked type and crash (see
  // docs/design or bisect notes for the createEmptyTensor AV). Instead,
  // derive the output type from hiddenStates' own (reliably ranked,
  // imported from its already-resolved producer type) type, per the op's
  // documented output-shape == hidden_states contract.
  mlir::RankedTensorType rt;
  if (auto ranked =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType())) {
    rt = ranked;
  } else {
    auto hiddenStatesType =
        mlir::dyn_cast<mlir::RankedTensorType>(hiddenStates.getType());
    if (!hiddenStatesType) {
      return rewriter.notifyMatchFailure(
          op, "QMoE (com.amd): result and hidden_states are both unranked; "
              "cannot infer output shape");
    }
    rt = mlir::RankedTensorType::get(hiddenStatesType.getShape(),
                                     hiddenStatesType.getElementType());
  }
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, hiddenStates);

  // Result type inferred from `init` via InferTypeOpInterface -- DPS
  // contract: result type == outs operand type.
  auto hipOp = mlir::hip::QMoEAmdOp::create(
      rewriter, loc, context, hiddenStates, fc1ExpertsWeights, fc1ExpertsScales,
      fc2ExpertsWeights, fc2ExpertsScales, fc1LatentWeights, fc1LatentScales,
      fc2LatentWeights, fc2LatentScales, sharedFc1Weights, sharedFc1Scales,
      sharedFc2Weights, sharedFc2Scales, routerWeight, correctionBias, init,
      kAttr, expertWeightBitsAttr, blockSizeAttr, normalizeAttr,
      useCorrectionBiasAttr, routedScalingFactorAttr, activationTypeAttr,
      routingTypeAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateQMoEAmdConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<QMoEAmdToHip>(ctx);
}

} // namespace hip
} // namespace mlir
