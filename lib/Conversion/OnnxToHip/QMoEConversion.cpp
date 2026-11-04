/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"

#include <cmath>
#include <limits>

STATISTIC(NumQMoESigmoidRoutingFolds,
          "Number of QMoE routing subgraphs folded into sigmoid_bias routing");

namespace mlir {
namespace hip {
namespace {

static bool isMsQMoE(mlir::Operation *op) {
  auto funcNameAttr = op->getAttrOfType<mlir::StringAttr>("function_name");
  auto domainAttr = op->getAttrOfType<mlir::StringAttr>("domain_name");
  return funcNameAttr && funcNameAttr.getValue() == "QMoE" && domainAttr &&
         domainAttr.getValue() == "com.microsoft";
}

/// Routing recovered from the graph by QMoESigmoidRoutingFold.
struct SigmoidBiasRouting {
  mlir::Value logits;         // fp32 router logits
  mlir::Value correctionBias; // [num_experts] fp32
  float scale;                // routed_scaling_factor
};

//===----------------------------------------------------------------------===//
// ONNX QMoE -> HIP QMoE (com.microsoft custom op)
//===----------------------------------------------------------------------===//

/// Validates `op` and replaces it with `hip.qmoe`. With `sigmoid`, routing
/// reads the logits and correction bias instead of operands 1 and 14.
static mlir::LogicalResult convertQMoE(mlir::Operation *op,
                                       mlir::PatternRewriter &rewriter,
                                       const SigmoidBiasRouting *sigmoid);

struct QMoEToHip : public mlir::RewritePattern {
  QMoEToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isMsQMoE(op))
      return rewriter.notifyMatchFailure(op, "not a com.microsoft QMoE op");
    return convertQMoE(op, rewriter, /*sigmoid=*/nullptr);
  }
};

static mlir::LogicalResult convertQMoE(mlir::Operation *op,
                                       mlir::PatternRewriter &rewriter,
                                       const SigmoidBiasRouting *sigmoid) {
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
  auto normalizeAttr = rewriter.getI64IntegerAttr(
      sigmoid ? 1 : (normIntAttr ? normIntAttr.getSInt() : 0));

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

  mlir::Value correctionBias;
  auto routingTypeAttr = rewriter.getStringAttr("softmax");
  auto scaleAttr = rewriter.getF32FloatAttr(1.0f);
  if (sigmoid) {
    routerProbs = sigmoid->logits;
    routerWeights = mlir::Value{};
    correctionBias = sigmoid->correctionBias;
    routingTypeAttr = rewriter.getStringAttr("sigmoid_bias");
    scaleAttr = rewriter.getF32FloatAttr(sigmoid->scale);
  }

  auto rt = mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, rt, input);

  // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
  // result type == outs operand type.
  auto hipOp = mlir::hip::QMoEOp::create(
      rewriter, loc, context, input, routerProbs, fc1Weights, fc1Scales,
      fc2Weights, fc2Scales, fc1Bias, fc2Bias, fc3Weights, fc3Scales, fc3Bias,
      fc1ZeroPoints, fc2ZeroPoints, fc3ZeroPoints, routerWeights,
      correctionBias, init, expertWeightBitsAttr, kAttr, blockSizeAttr,
      normalizeAttr, swigluFusionAttr, useSparseAttr, activationAlphaAttr,
      activationBetaAttr, swigluLimitAttr, activationTypeAttr, routingTypeAttr,
      scaleAttr);
  rewriter.replaceOp(op, hipOp->getResults());
  return mlir::success();
}

//===----------------------------------------------------------------------===//
// Sigmoid + correction-bias routing fold
//===----------------------------------------------------------------------===//
//
// Sigmoid routers select experts on sigmoid scores plus a learned correction
// bias and weight them by the unbiased scores. com.microsoft QMoE
// only routes by softmax, so exports build the routing outside the op and hand
// it the result as two dense [T, E] scatters: router_probs is -inf except a
// constant at the selected experts (so QMoE's softmax top-k picks exactly
// those), and router_weights carries the selected weights.
//
// Before:
//   %s   = onnx.Sigmoid(%logits)                       // fp32 [.., E]
//   %sel = onnx.Add(%s, %bias)                         // %bias: [E] fp32
//   %v, %idx = onnx.TopK(%sel, k)
//   %g   = onnx.GatherElements(%s, %idx)  {axis = -1}
//   %w   = onnx.Mul(onnx.Div(%g, onnx.ReduceSum(%g, [-1])), c)
//   %rp  = onnx.ScatterElements(Expand(-inf), %idx, Expand(1.0))
//   %rw  = onnx.ScatterElements(Expand(0.0), %idx, onnx.Cast(%w))
//   %y   = onnx.Custom "QMoE"(%x, Reshape(%rp), ..., Reshape(%rw))
//            {k, normalize_routing_weights = 0}
// After:
//   %y = hip.qmoe(%ctx) ins(%x, %logits, ...) correction_bias(%bias : ...)
//            {routing_type = "sigmoid_bias", routed_scaling_factor = c,
//             normalize_routing_weights = 1}
//
// The router MatMul is kept as is, so selection still sees fp32 logits.

static mlir::Operation *defOf(mlir::Value v, llvm::StringRef name) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  return def && def->getName().getStringRef() == name ? def : nullptr;
}

static mlir::Value peelReshapes(mlir::Value v) {
  while (mlir::Operation *r = defOf(v, "onnx.Reshape"))
    v = r->getOperand(0);
  return v;
}

static mlir::Type elemOf(mlir::Value v) {
  return mlir::getElementTypeOrSelf(v.getType());
}

/// The single value of a scalar or splat constant.
static std::optional<double> getSplatFloat(mlir::Value v) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def)
    return std::nullopt;
  mlir::Attribute attr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    attr = cst.getValue();
  else if (def->getName().getStringRef() == "onnx.Constant") {
    if (auto f = def->getAttrOfType<mlir::FloatAttr>("value_float"))
      return f.getValueAsDouble();
    attr = def->getAttr("value");
  }
  auto dense = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(attr);
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::FloatType>(dense.getElementType()))
    return std::nullopt;
  return dense.getSplatValue<mlir::APFloat>().convertToDouble();
}

/// A tensor every element of which is one constant: a splat constant, or an
/// `Expand` / `ConstantOfShape` of one.
static std::optional<double> getFillValue(mlir::Value v) {
  if (mlir::Operation *e = defOf(v, "onnx.Expand"))
    return getSplatFloat(e->getOperand(0));
  if (mlir::Operation *c = defOf(v, "onnx.ConstantOfShape")) {
    auto dense = c->getAttrOfType<mlir::DenseElementsAttr>("value");
    if (dense && dense.isSplat() &&
        mlir::isa<mlir::FloatType>(dense.getElementType()))
      return dense.getSplatValue<mlir::APFloat>().convertToDouble();
    return std::nullopt;
  }
  return getSplatFloat(v);
}

static std::optional<int64_t> getScalarInt(mlir::Value v) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def)
    return std::nullopt;
  mlir::Attribute attr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    attr = cst.getValue();
  else if (def->getName().getStringRef() == "onnx.Constant") {
    if (auto i = def->getAttrOfType<mlir::IntegerAttr>("value_int"))
      return i.getValue().getSExtValue();
    if (auto ints = def->getAttrOfType<mlir::ArrayAttr>("value_ints");
        ints && ints.size() == 1)
      if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(ints[0]))
        return i.getValue().getSExtValue();
    attr = def->getAttr("value");
  }
  auto dense = mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(attr);
  if (!dense || dense.getNumElements() != 1 ||
      !dense.getElementType().isIntOrIndex())
    return std::nullopt;
  return (*dense.getValues<mlir::APInt>().begin()).getSExtValue();
}

static int64_t getIntAttrOr(mlir::Operation *op, llvm::StringRef name,
                            int64_t dflt) {
  auto attr = op->getAttrOfType<mlir::IntegerAttr>(name);
  return attr ? attr.getValue().getSExtValue() : dflt;
}

/// `axis` names the last dimension of the rank-`rank` operand.
static bool isLastAxis(int64_t axis, int64_t rank) {
  return axis == -1 || axis == rank - 1;
}

static int64_t rankOf(mlir::Value v) {
  auto ty = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  return ty ? ty.getRank() : -1;
}

/// `ScatterElements(data, idx, updates)` along the last axis, no reduction.
static mlir::Operation *matchLastAxisScatter(mlir::Value v) {
  mlir::Operation *s = defOf(v, "onnx.ScatterElements");
  if (!s || s->getNumOperands() != 3)
    return nullptr;
  if (!isLastAxis(getIntAttrOr(s, "axis", 0), rankOf(s->getOperand(0))))
    return nullptr;
  if (auto red = s->getAttrOfType<mlir::StringAttr>("reduction");
      red && red.getValue() != "none")
    return nullptr;
  return s;
}

/// `ReduceSum(x)` over the last axis with keepdims, axes as operand (opset 13+)
/// or attribute.
static bool isLastAxisKeepdimsSum(mlir::Operation *r, mlir::Value x) {
  if (!r || r->getName().getStringRef() != "onnx.ReduceSum" ||
      r->getOperand(0) != x || getIntAttrOr(r, "keepdims", 1) != 1)
    return false;
  std::optional<int64_t> axis;
  if (r->getNumOperands() > 1)
    axis = getScalarInt(r->getOperand(1));
  else if (auto axes = r->getAttrOfType<mlir::ArrayAttr>("axes");
           axes && axes.size() == 1)
    if (auto i = mlir::dyn_cast<mlir::IntegerAttr>(axes[0]))
      axis = i.getValue().getSExtValue();
  return axis && isLastAxis(*axis, rankOf(x));
}

struct QMoESigmoidRoutingFold : public mlir::RewritePattern {
  QMoESigmoidRoutingFold(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (!isMsQMoE(op) || op->getNumOperands() < 15)
      return rewriter.notifyMatchFailure(op, "not a QMoE with router_weights");
    if (getIntAttrOr(op, "normalize_routing_weights", 0) != 0)
      return rewriter.notifyMatchFailure(
          op, "normalize_routing_weights=1 would renormalize the weights");
    const int64_t k = getIntAttrOr(op, "k", 1);

    mlir::Value probsIn = op->getOperand(1);
    mlir::Value weightsIn = op->getOperand(14);
    if (!probsIn.hasOneUse() || !weightsIn.hasOneUse())
      return rewriter.notifyMatchFailure(op, "routing tensors have other uses");
    mlir::Operation *probsScatter = matchLastAxisScatter(peelReshapes(probsIn));
    mlir::Operation *weightsScatter =
        matchLastAxisScatter(peelReshapes(weightsIn));
    if (!probsScatter || !weightsScatter)
      return rewriter.notifyMatchFailure(op, "routing is not two scatters");
    mlir::Value idx = probsScatter->getOperand(1);
    if (weightsScatter->getOperand(1) != idx)
      return rewriter.notifyMatchFailure(op, "scatters use different indices");

    // router_probs must make QMoE's softmax top-k pick exactly `idx`.
    std::optional<double> unselected =
        getFillValue(probsScatter->getOperand(0));
    std::optional<double> selected = getFillValue(probsScatter->getOperand(2));
    if (!unselected || !selected || !std::isfinite(*selected) ||
        !(*unselected < *selected))
      return rewriter.notifyMatchFailure(op, "router_probs is not a selection");

    mlir::Operation *topk = defOf(idx, "onnx.TopK");
    if (!topk || topk->getNumResults() != 2 || topk->getResult(1) != idx)
      return rewriter.notifyMatchFailure(op, "indices are not TopK indices");
    mlir::Value selScore = topk->getOperand(0);
    if (!isLastAxis(getIntAttrOr(topk, "axis", -1), rankOf(selScore)) ||
        getIntAttrOr(topk, "largest", 1) != 1 ||
        getScalarInt(topk->getOperand(1)) != k)
      return rewriter.notifyMatchFailure(op, "TopK is not the QMoE top-k");

    mlir::Operation *add = defOf(selScore, "onnx.Add");
    if (!add)
      return rewriter.notifyMatchFailure(op, "selection score is not an Add");
    mlir::Value sig = add->getOperand(0);
    mlir::Value bias = add->getOperand(1);
    if (!defOf(sig, "onnx.Sigmoid"))
      std::swap(sig, bias);
    mlir::Operation *sigmoidOp = defOf(sig, "onnx.Sigmoid");
    if (!sigmoidOp)
      return rewriter.notifyMatchFailure(op, "selection is not sigmoid + bias");
    mlir::Value logits = sigmoidOp->getOperand(0);

    // Weights: Cast?(Mul(Div(G, ReduceSum(G)), c)), G = GatherElements(s, idx).
    mlir::Value w = weightsScatter->getOperand(2);
    if (mlir::Operation *cast = defOf(w, "onnx.Cast"))
      w = cast->getOperand(0);
    mlir::Operation *mul = defOf(w, "onnx.Mul");
    if (!mul)
      return rewriter.notifyMatchFailure(op, "weights are not scaled");
    mlir::Value divOut = mul->getOperand(0);
    std::optional<double> scale = getSplatFloat(mul->getOperand(1));
    if (!scale) {
      divOut = mul->getOperand(1);
      scale = getSplatFloat(mul->getOperand(0));
    }
    mlir::Operation *div = defOf(divOut, "onnx.Div");
    if (!scale || !std::isfinite(*scale) || !div)
      return rewriter.notifyMatchFailure(op, "weights are not s / sum * c");
    mlir::Value gathered = div->getOperand(0);
    mlir::Operation *gather = defOf(gathered, "onnx.GatherElements");
    if (!gather || gather->getOperand(0) != sig ||
        gather->getOperand(1) != idx ||
        !isLastAxis(getIntAttrOr(gather, "axis", 0), rankOf(sig)) ||
        !isLastAxisKeepdimsSum(div->getOperand(1).getDefiningOp(), gathered))
      return rewriter.notifyMatchFailure(
          op, "weights are not the normalized sigmoid of the selection");

    auto logitsTy = mlir::dyn_cast<mlir::RankedTensorType>(logits.getType());
    auto biasTy = mlir::dyn_cast<mlir::RankedTensorType>(bias.getType());
    auto inputTy =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!logitsTy || !biasTy || !inputTy || logitsTy.getRank() < 1 ||
        !logitsTy.getElementType().isF32() || !elemOf(sig).isF32() ||
        biasTy.getRank() != 1 || !biasTy.getElementType().isF32())
      return rewriter.notifyMatchFailure(
          op, "needs fp32 logits and a rank-1 fp32 correction bias");
    const int64_t experts = logitsTy.getDimSize(logitsTy.getRank() - 1);
    if (mlir::ShapedType::isDynamic(experts) ||
        biasTy.getDimSize(0) != experts || k > experts)
      return rewriter.notifyMatchFailure(op, "bias does not span the experts");
    if (logitsTy.hasStaticShape() && inputTy.hasStaticShape() &&
        logitsTy.getNumElements() / experts !=
            inputTy.getNumElements() /
                inputTy.getDimSize(inputTy.getRank() - 1))
      return rewriter.notifyMatchFailure(op, "logits rows != input tokens");

    SigmoidBiasRouting routing{logits, bias, static_cast<float>(*scale)};
    mlir::Operation *probsRoot = probsIn.getDefiningOp();
    mlir::Operation *weightsRoot = weightsIn.getDefiningOp();
    if (mlir::failed(convertQMoE(op, rewriter, &routing)))
      return mlir::failure();
    eraseDeadOnnxProducers(rewriter, {probsRoot, weightsRoot});
    ++NumQMoESigmoidRoutingFolds;
    return mlir::success();
  }

private:
  /// Erase the routing chain the fold made dead, so convertComputeOps does
  /// not lower it first. Only `onnx.*` ops are touched.
  static void eraseDeadOnnxProducers(mlir::PatternRewriter &rewriter,
                                     llvm::ArrayRef<mlir::Operation *> roots) {
    // A node shared by both scatters (TopK, the sigmoid) dies only after
    // both branches are gone, so live candidates are retried until no erase
    // makes progress.
    llvm::SetVector<mlir::Operation *> candidates;
    for (mlir::Operation *root : roots)
      if (root)
        candidates.insert(root);
    llvm::SmallPtrSet<mlir::Operation *, 32> erased;
    bool progress = true;
    while (progress) {
      progress = false;
      for (size_t i = 0; i < candidates.size(); ++i) {
        mlir::Operation *def = candidates[i];
        if (erased.contains(def) || !def->use_empty() ||
            !def->getName().getStringRef().starts_with("onnx."))
          continue;
        for (mlir::Value operand : def->getOperands())
          if (mlir::Operation *producer = operand.getDefiningOp())
            candidates.insert(producer);
        erased.insert(def);
        rewriter.eraseOp(def);
        progress = true;
      }
    }
  }
};

} // namespace

void populateQMoEConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<QMoEToHip>(ctx);
}

void populateQMoESigmoidRoutingFoldPatterns(RewritePatternSet &patterns,
                                            MLIRContext *ctx) {
  patterns.add<QMoESigmoidRoutingFold>(ctx);
}

} // namespace hip
} // namespace mlir
