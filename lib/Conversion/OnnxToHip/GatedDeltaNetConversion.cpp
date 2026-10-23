/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// com.microsoft.GatedDeltaNet -> hip.gated_delta_net
struct GatedDeltaNetToHip : public mlir::RewritePattern {
  GatedDeltaNetToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
GatedDeltaNetToHip::matchAndRewrite(mlir::Operation *op,
                                    mlir::PatternRewriter &rewriter) const {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  if (!funcNameAttr || funcNameAttr.getValue() != "GatedDeltaNet") {
    return rewriter.notifyMatchFailure(op, "not a GatedDeltaNet operation");
  }
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  if (!domainAttr || domainAttr.getValue() != "com.microsoft") {
    return rewriter.notifyMatchFailure(
        op, "domain must be com.microsoft for GatedDeltaNet");
  }

  auto readStr = [&](llvm::StringRef name, llvm::StringRef defaultVal) {
    auto attr = op->getAttrOfType<mlir::StringAttr>(name);
    return attr ? attr.getValue() : defaultVal;
  };
  auto readInt = [&](llvm::StringRef name, int64_t defaultVal) {
    auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
    return attr ? attr.getValue().getSExtValue() : defaultVal;
  };
  if (readStr("update_rule", "gated_delta") != "gated_delta") {
    return rewriter.notifyMatchFailure(op, "update_rule must be gated_delta");
  }
  if (readStr("gate_activation", "none") != "qwen") {
    return rewriter.notifyMatchFailure(op, "gate_activation must be qwen");
  }
  if (readStr("beta_activation", "none") != "sigmoid") {
    return rewriter.notifyMatchFailure(op, "beta_activation must be sigmoid");
  }
  if (readInt("qk_l2_norm", 0) != 1) {
    return rewriter.notifyMatchFailure(op, "qk_l2_norm must be 1");
  }
  if (readInt("state_update_capacity", 0) != 0) {
    return rewriter.notifyMatchFailure(op, "state_update_capacity must be 0");
  }
  float scale = 0.0f;
  if (auto attr = op->getAttrOfType<mlir::FloatAttr>("scale")) {
    scale = static_cast<float>(attr.getValueAsDouble());
  }

  size_t numOps = op->getNumOperands();
  auto getOptionalOperand = [&](size_t idx) -> mlir::Value {
    if (idx >= numOps) {
      return nullptr;
    }
    mlir::Value val = op->getOperand(idx);
    if (mlir::isa<mlir::NoneType>(val.getType())) {
      return nullptr;
    }
    return val;
  };

  if (numOps < 9 || numOps > 11) {
    return rewriter.notifyMatchFailure(op,
                                       "GatedDeltaNet expects 9-11 operands");
  }
  if (getOptionalOperand(1) || getOptionalOperand(2)) {
    return rewriter.notifyMatchFailure(op, "only packed QKV is supported");
  }
  if (getOptionalOperand(3)) {
    return rewriter.notifyMatchFailure(op, "cu_seqlens is not supported");
  }
  if (getOptionalOperand(9) || getOptionalOperand(10)) {
    return rewriter.notifyMatchFailure(
        op, "capture_count / state_update_active are not supported");
  }
  mlir::Value qkv = op->getOperand(0);
  mlir::Value decay = getOptionalOperand(4);
  mlir::Value beta = getOptionalOperand(5);
  mlir::Value initialState = getOptionalOperand(6);
  mlir::Value aLog = getOptionalOperand(7);
  mlir::Value dtBias = getOptionalOperand(8);
  if (!decay || !beta || !initialState || !aLog || !dtBias) {
    return rewriter.notifyMatchFailure(
        op, "decay, beta, initial_state, a_log and dt_bias are required");
  }
  if (op->getNumResults() != 2) {
    return rewriter.notifyMatchFailure(op, "GatedDeltaNet expects 2 results");
  }

  auto qkvType = mlir::dyn_cast<mlir::RankedTensorType>(qkv.getType());
  if (!qkvType || qkvType.getRank() != 3 || qkvType.isDynamicDim(2)) {
    return rewriter.notifyMatchFailure(
        op, "query must be packed rank 3 with static last dim");
  }
  mlir::Type elem = qkvType.getElementType();
  if (!elem.isF16() && !elem.isF32()) {
    return rewriter.notifyMatchFailure(op, "query must be f16 or f32");
  }
  auto stateType =
      mlir::dyn_cast<mlir::RankedTensorType>(initialState.getType());
  if (!stateType || stateType.getRank() != 4 ||
      !stateType.getElementType().isF32()) {
    return rewriter.notifyMatchFailure(op, "initial_state must be rank 4 f32");
  }
  if (stateType.isDynamicDim(1) || stateType.isDynamicDim(2) ||
      stateType.isDynamicDim(3)) {
    return rewriter.notifyMatchFailure(op,
                                       "initial_state dims 1-3 must be static");
  }

  const int64_t packed = qkvType.getDimSize(2);
  const int64_t numHeadsV = stateType.getDimSize(1);
  const int64_t headSizeV = stateType.getDimSize(2);
  const int64_t headSizeQk = stateType.getDimSize(3);
  if (headSizeQk != 32 && headSizeQk != 64 && headSizeQk != 128 &&
      headSizeQk != 256) {
    return rewriter.notifyMatchFailure(op, "dk must be 32, 64, 128 or 256");
  }
  const int64_t qkWidth = packed - numHeadsV * headSizeV;
  if (qkWidth <= 0 || qkWidth % (2 * headSizeQk) != 0) {
    return rewriter.notifyMatchFailure(
        op, "packed query width minus Hv*dv must be a positive multiple of "
            "2*dk");
  }
  if (numHeadsV % (qkWidth / (2 * headSizeQk)) != 0) {
    return rewriter.notifyMatchFailure(op, "Hv must be a multiple of Hqk");
  }

  auto matches = [](mlir::Value v, mlir::Type expectedElem,
                    llvm::ArrayRef<int64_t> expected) {
    auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
    if (!type || type.getElementType() != expectedElem ||
        type.getRank() != static_cast<int64_t>(expected.size())) {
      return false;
    }
    for (auto [i, want] : llvm::enumerate(expected)) {
      int64_t got = type.getDimSize(i);
      if (want != mlir::ShapedType::kDynamic &&
          got != mlir::ShapedType::kDynamic && got != want) {
        return false;
      }
    }
    return true;
  };
  constexpr int64_t kAny = mlir::ShapedType::kDynamic;
  mlir::Type f32 = rewriter.getF32Type();
  const int64_t batch = qkvType.getDimSize(0);
  const int64_t seq = qkvType.getDimSize(1);
  if (!matches(initialState, f32, {batch, kAny, kAny, kAny})) {
    return rewriter.notifyMatchFailure(op,
                                       "initial_state batch mismatches query");
  }
  if (!matches(decay, f32, {batch, seq, numHeadsV})) {
    return rewriter.notifyMatchFailure(op, "decay must be f32 [B, L, Hv]");
  }
  if (!matches(beta, f32, {batch, seq, numHeadsV})) {
    return rewriter.notifyMatchFailure(op, "beta must be f32 [B, L, Hv]");
  }
  if (!matches(aLog, f32, {numHeadsV})) {
    return rewriter.notifyMatchFailure(op, "a_log must be f32 [Hv]");
  }
  if (!matches(dtBias, f32, {numHeadsV})) {
    return rewriter.notifyMatchFailure(op, "dt_bias must be f32 [Hv]");
  }

  auto outputType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!outputType ||
      !matches(op->getResult(0), elem, {batch, seq, numHeadsV, headSizeV}) ||
      outputType.isDynamicDim(2) || outputType.isDynamicDim(3)) {
    return rewriter.notifyMatchFailure(
        op, "output must be [B, L, Hv, dv] with static Hv, dv");
  }
  auto finalType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(1).getType());
  if (!finalType ||
      !matches(op->getResult(1), f32,
               {stateType.getDimSize(0), numHeadsV, headSizeV, headSizeQk}) ||
      finalType.isDynamicDim(1) || finalType.isDynamicDim(2) ||
      finalType.isDynamicDim(3)) {
    return rewriter.notifyMatchFailure(
        op, "final_state must match initial_state shape");
  }

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure)) {
    return rewriter.notifyMatchFailure(op, "missing context argument");
  }

  mlir::Location loc = op->getLoc();
  mlir::Value outputInit = createEmptyTensor(rewriter, loc, outputType, qkv);
  mlir::Value finalInit =
      createEmptyTensor(rewriter, loc, finalType, initialState);

  auto state = mlir::OperationState(loc, "hip.gated_delta_net");
  state.addOperands({*ctxOrFailure, qkv, decay, beta, initialState, aLog,
                     dtBias, outputInit, finalInit});
  state.addTypes({outputType, finalType});
  state.addAttribute("scale", rewriter.getF32FloatAttr(scale));
  mlir::Operation *hipOp = rewriter.create(state);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

} // namespace

void populateGatedDeltaNetConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {
  patterns.add<GatedDeltaNetToHip>(ctx);
}

} // namespace hip
} // namespace mlir
