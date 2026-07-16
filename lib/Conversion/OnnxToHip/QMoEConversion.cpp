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

// Trace the router-gate chain feeding QMoE's `router_probs` operand so QMoE can
// recompute routing in fp32 internally. `router_probs` is actually the router
// LOGITS — QMoE applies softmax + top-k internally — produced upstream by:
//
//   %logits = MatMulNBits(%router_input, %W_q4, %scales, [%zp])  (router_proj)
//   [%logits = Add(%logits, %bias)]                              (optional bias)
//   [%probs  = Reshape(%logits, ...)]                            (optional)
//   %out     = QMoE(%hidden, %probs, ...)
//
// The walk skips shape-only Reshape ops and strips at most one bias Add, then
// matches the router_proj MatMulNBits. Both the ONNX form (onnx.Reshape /
// onnx.Add / onnx.Custom{MatMulNBits}) and the already-converted HIP form
// (tensor.collapse_shape|expand_shape / hip.add / hip.matmul_nbits) are
// accepted, since the onnx-to-hip patterns may fire in either order within a
// single greedy conversion pass. On success fills the router_proj operands +
// optional bias + quant params and returns true; on any structural mismatch
// returns false, and QMoE falls back to consuming the fp16 `router_probs` via
// hip_qmoe_topk_routing (unchanged behavior for MoE models whose router is not
// a MatMulNBits[+bias][+reshape] chain).
static bool traceRouterGate(mlir::Value routerProbs, mlir::Value &routerInput,
                            mlir::Value &gateWeight, mlir::Value &gateScales,
                            mlir::Value &gateZp, mlir::Value &gateBias,
                            int64_t &bits, int64_t &blockSize) {
  auto isNone = [](mlir::Value v) {
    return !v || mlir::isa<mlir::NoneType>(v.getType());
  };

  // Extract router_proj MatMulNBits operands from the already-converted
  // hip.matmul_nbits or the un-converted onnx.Custom{MatMulNBits} form.
  auto matchMatMulNBits = [&](mlir::Operation *d) -> bool {
    if (auto mm = mlir::dyn_cast<mlir::hip::MatMulNBitsOp>(d)) {
      routerInput = mm.getA();
      gateWeight = mm.getB();
      gateScales = mm.getScales();
      gateZp = isNone(mm.getZeroPoints()) ? mlir::Value{} : mm.getZeroPoints();
      bits = mm.getBits();
      blockSize = mm.getBlockSize();
      return true;
    }
    auto fnAttr = d->getAttrOfType<mlir::StringAttr>("function_name");
    auto domAttr = d->getAttrOfType<mlir::StringAttr>("domain_name");
    if (fnAttr && fnAttr.getValue() == "MatMulNBits" && domAttr &&
        domAttr.getValue() == "com.microsoft" && d->getNumOperands() >= 3) {
      auto bs = d->getAttrOfType<mlir::IntegerAttr>("block_size");
      if (!bs)
        return false;
      routerInput = d->getOperand(0);
      gateWeight = d->getOperand(1);
      gateScales = d->getOperand(2);
      gateZp = (d->getNumOperands() > 3 && !isNone(d->getOperand(3)))
                   ? d->getOperand(3)
                   : mlir::Value{};
      auto b = d->getAttrOfType<mlir::IntegerAttr>("bits");
      bits = b ? b.getSInt() : 4;
      blockSize = bs.getSInt();
      return true;
    }
    return false;
  };

  // Skip a shape-only reshape; return the data source (or v unchanged).
  auto stripReshape = [](mlir::Value v) -> mlir::Value {
    mlir::Operation *d = v.getDefiningOp();
    if (!d)
      return v;
    llvm::StringRef n = d->getName().getStringRef();
    if (n == "onnx.Reshape" || n == "tensor.collapse_shape" ||
        n == "tensor.expand_shape")
      return d->getOperand(0);
    return v;
  };

  // Match Add(proj, bias) (hip.add or onnx.Add) and split the larger-rank
  // operand (projection output) from the smaller/equal-rank operand (bias).
  auto matchAddBias = [](mlir::Value v, mlir::Value &projSide,
                         mlir::Value &biasSide) -> bool {
    mlir::Operation *d = v.getDefiningOp();
    if (!d)
      return false;
    mlir::Value a, b;
    if (auto add = mlir::dyn_cast<mlir::hip::AddOp>(d)) {
      a = add.getLhs();
      b = add.getRhs();
    } else if (d->getName().getStringRef() == "onnx.Add" &&
               d->getNumOperands() == 2) {
      a = d->getOperand(0);
      b = d->getOperand(1);
    } else {
      return false;
    }
    auto rankOf = [](mlir::Value x) -> int64_t {
      if (auto t = mlir::dyn_cast<mlir::ShapedType>(x.getType()))
        return t.hasRank() ? t.getRank() : -1;
      return -1;
    };
    // The bias broadcasts to the projection output, so rank(bias) <=
    // rank(proj). Ties default to (proj=a, bias=b) matching the common
    // Add(proj, bias) authoring convention.
    if (rankOf(b) <= rankOf(a)) {
      projSide = a;
      biasSide = b;
    } else {
      projSide = b;
      biasSide = a;
    }
    return true;
  };

  mlir::Value cur = routerProbs;
  gateBias = mlir::Value{};
  // Real chain is <= 3 ops (matmul, add, reshape); cap the walk so a
  // pathological IR shape cannot loop forever.
  for (int hop = 0; hop < 6; ++hop) {
    cur = stripReshape(cur);
    mlir::Operation *d = cur.getDefiningOp();
    if (!d)
      return false;
    if (matchMatMulNBits(d))
      return true;
    // Strip at most one bias add on the way up (record the first bias found).
    if (!gateBias) {
      mlir::Value projSide, biasSide;
      if (matchAddBias(cur, projSide, biasSide)) {
        gateBias = biasSide;
        cur = projSide;
        continue;
      }
    }
    return false; // unknown op — QMoE falls back to the fp16 router_probs
  }
  return false;
}

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

  // Recover the upstream router_proj (MatMulNBits[+bias][+reshape]) chain so
  // QMoE can recompute the gate in fp32 internally (see traceRouterGate). When
  // the trace fails these stay null and QMoE consumes the fp16 router_probs
  // unchanged. `routerGateBias` is the optional per-expert bias of an
  // onnx.Add/hip.add inserted between router_proj and the softmax/top-k.
  mlir::Value routerInput, routerGateWeight, routerGateScales, routerGateZp,
      routerGateBias;
  int64_t routerGateBits = 0, routerGateBlockSize = 0;
  traceRouterGate(routerProbs, routerInput, routerGateWeight, routerGateScales,
                  routerGateZp, routerGateBias, routerGateBits,
                  routerGateBlockSize);
  auto routerGateBitsAttr = rewriter.getI64IntegerAttr(routerGateBits);
  auto routerGateBlockSizeAttr =
      rewriter.getI64IntegerAttr(routerGateBlockSize);

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp = mlir::hip::QMoEOp::create(
      rewriter, loc, context, input, routerProbs, fc1Weights, fc1Scales,
      fc2Weights, fc2Scales, fc1Bias, fc2Bias, fc3Weights, fc3Scales, fc3Bias,
      fc1ZeroPoints, fc2ZeroPoints, fc3ZeroPoints, routerWeights, routerInput,
      routerGateWeight, routerGateScales, routerGateZp, routerGateBias, init,
      expertWeightBitsAttr, kAttr, blockSizeAttr, normalizeAttr,
      swigluFusionAttr, useSparseAttr, activationAlphaAttr, activationBetaAttr,
      swigluLimitAttr, activationTypeAttr, routerGateBitsAttr,
      routerGateBlockSizeAttr);
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
