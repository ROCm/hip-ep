/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DsaPrologueFold.cpp - Fold DSA neighbours into the op --------------===//
//
// Exporters surround a com.microsoft DynamicSparseAttention with
//
//   q, k, v = Split(qkv, [N*D, kvN*D, kvN*D], axis = -1)
//   q'  = MRotaryEmbedding(Reshape(SimplifiedLayerNormalization(
//             Reshape(q, [0, -1, D]), q_norm_weight), [0, -1, N*D]),
//             position_ids, cos_cache, sin_cache)
//   k'  = the same chain over k with k_norm_weight and kv_num_heads
//   out = Mul(DynamicSparseAttention(q', k', v, ...), Sigmoid(gate))
//
// which costs a strided copy of every Split slice, two norm and two rotary
// launches, a Sigmoid and a Mul around each attention call. Three patterns
// move that work into the attention op:
//   - the gate fold appends `gate` at kDsaGateIdx and replaces the Mul;
//   - the prologue fold passes q and k before their chains, appends the norm
//     weights, position ids and rotary caches at kDsaPrologueIdx and stamps
//     the epsilon, MRoPE section and layout;
//   - the packed fold passes qkv as the query with key and value absent.
// The kernels round to f16 wherever the folded ops do.
// DynamicSparseAttentionConversion is the only reader of the appended inputs.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "llvm/ADT/STLExtras.h"

#include <optional>

namespace mlir {
namespace hip {
namespace {

constexpr unsigned kKeyIdx = 1;
constexpr unsigned kValueIdx = 2;
constexpr unsigned kAuxKeyIdx = 5;

Operation *producer(Value v, StringRef name) {
  Operation *op = v.getDefiningOp();
  return op && op->getName().getStringRef() == name ? op : nullptr;
}

Operation *customProducer(Value v, StringRef function) {
  Operation *op = producer(v, "onnx.Custom");
  auto fn = op ? op->getAttrOfType<StringAttr>("function_name") : nullptr;
  return fn && fn.getValue() == function ? op : nullptr;
}

int64_t intAttr(Operation *op, StringRef name, int64_t defaultVal) {
  auto attr = op->getAttrOfType<IntegerAttr>(name);
  return attr ? attr.getValue().getSExtValue() : defaultVal;
}

bool isConstant(Value v, ArrayRef<int64_t> expected) {
  Operation *op = producer(v, "onnx.Constant");
  auto value = op ? op->getAttrOfType<DenseElementsAttr>("value") : nullptr;
  if (!value || !value.getElementType().isInteger(64) ||
      value.getNumElements() != static_cast<int64_t>(expected.size())) {
    return false;
  }
  return llvm::equal(
      llvm::map_range(value.getValues<APInt>(),
                      [](const APInt &v) { return v.getSExtValue(); }),
      expected);
}

Value optionalOperand(Operation *op, unsigned idx) {
  if (idx >= op->getNumOperands()) {
    return nullptr;
  }
  Value v = op->getOperand(idx);
  return isa<NoneType>(v.getType()) ? nullptr : v;
}

/// A DynamicSparseAttention whose aux key slot is None (its value fills the
/// slots a fold skips) and whose inputs fit the extension slots.
bool isFoldableDsa(Operation *op) {
  auto fn = op->getAttrOfType<StringAttr>("function_name");
  auto domain = op->getAttrOfType<StringAttr>("domain_name");
  return fn && fn.getValue() == "DynamicSparseAttention" && domain &&
         domain.getValue() == "com.microsoft" && op->getNumResults() == 3 &&
         op->getNumOperands() > kAuxKeyIdx + 1 &&
         op->getNumOperands() <= kDsaMaxInputs &&
         isa<NoneType>(op->getOperand(kAuxKeyIdx).getType());
}

/// `op` over `operands` with `extraAttrs` added, at the insertion point.
Operation *rebuild(PatternRewriter &rewriter, Operation *op,
                   ArrayRef<Value> operands,
                   ArrayRef<NamedAttribute> extraAttrs = {}) {
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(op->getResultTypes());
  state.addAttributes(op->getAttrs());
  state.addAttributes(extraAttrs);
  return rewriter.create(state);
}

/// out = Mul(DynamicSparseAttention.0, Sigmoid(g)) -> gate input g.
struct DsaGateFold : public RewritePattern {
  DsaGateFold(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isFoldableDsa(op) || optionalOperand(op, kDsaGateIdx)) {
      return rewriter.notifyMatchFailure(op, "not an ungated DSA");
    }
    Value out = op->getResult(0);
    if (!out.hasOneUse()) {
      return rewriter.notifyMatchFailure(op, "output has other users");
    }
    Operation *mul = *out.getUsers().begin();
    if (mul->getName().getStringRef() != "onnx.Mul" ||
        mul->getBlock() != op->getBlock() ||
        mul->getOperand(0) == mul->getOperand(1)) {
      return rewriter.notifyMatchFailure(op, "output is not multiplied");
    }
    Value other =
        mul->getOperand(0) == out ? mul->getOperand(1) : mul->getOperand(0);
    Operation *sigmoid = producer(other, "onnx.Sigmoid");
    if (!sigmoid) {
      return rewriter.notifyMatchFailure(op, "multiplier is not a Sigmoid");
    }
    Value gate = sigmoid->getOperand(0);
    if (mul->getResult(0).getType() != out.getType() ||
        gate.getType() != out.getType()) {
      return rewriter.notifyMatchFailure(op, "the gate must not broadcast");
    }

    // The gate may be computed after the attention; then the folded op takes
    // the Mul's place, after which the present caches must still be unread.
    Operation *def = gate.getDefiningOp();
    if (def && def->getBlock() != op->getBlock()) {
      return rewriter.notifyMatchFailure(op, "gate defined in another block");
    }
    if (def && op->isBeforeInBlock(def)) {
      for (Value present : op->getResults().drop_front()) {
        for (Operation *user : present.getUsers()) {
          if (user->getBlock() != mul->getBlock() ||
              !mul->isBeforeInBlock(user)) {
            return rewriter.notifyMatchFailure(
                op, "present cache read between the attention and the gate");
          }
        }
      }
      rewriter.setInsertionPoint(mul);
    }

    SmallVector<Value> operands = llvm::to_vector(op->getOperands());
    operands.resize(std::max<size_t>(operands.size(), kDsaGateIdx + 1),
                    op->getOperand(kAuxKeyIdx));
    operands[kDsaGateIdx] = gate;
    Operation *folded = rebuild(rewriter, op, operands);
    rewriter.replaceOp(mul, folded->getResult(0));
    rewriter.replaceOp(op, folded->getResults());
    if (sigmoid->use_empty()) {
      rewriter.eraseOp(sigmoid);
    }
    return success();
  }
};

/// One of the q / k chains MRotaryEmbedding(Reshape(SLN(Reshape(x)))).
struct NormRopeChain {
  Value input;
  Value weight;
  Value positionIds;
  Value cos;
  Value sin;
  APFloat epsilon = APFloat(0.0f);
  SmallVector<int64_t, 3> section;
  int64_t layout = 0;
  SmallVector<Operation *, 4> ops;
};

std::optional<NormRopeChain> matchChain(Value v, int64_t heads,
                                        int64_t headSize) {
  NormRopeChain chain;
  Operation *rope = customProducer(v, "MRotaryEmbedding");
  auto domain = rope ? rope->getAttrOfType<StringAttr>("domain_name") : nullptr;
  if (!rope || !domain || domain.getValue() != "com.microsoft" ||
      rope->getNumOperands() != 4 || rope->getNumResults() != 1 ||
      !v.hasOneUse() || intAttr(rope, "interleaved", 0) != 0 ||
      intAttr(rope, "is_packed_batching", 0) != 0 ||
      intAttr(rope, "num_heads", 0) != heads) {
    return std::nullopt;
  }
  if (auto scale = rope->getAttrOfType<FloatAttr>("scale");
      scale && scale.getValueAsDouble() != 1.0) {
    return std::nullopt;
  }
  chain.layout = intAttr(rope, "mrope_layout", 0);
  int64_t rotaryDim = intAttr(rope, "rotary_embedding_dim", 0);
  if (rotaryDim == 0) {
    rotaryDim = headSize;
  }
  auto sectionAttr = rope->getAttrOfType<ArrayAttr>("mrope_section");
  if ((chain.layout != 0 && chain.layout != 1) || rotaryDim <= 0 ||
      rotaryDim % 2 != 0 || rotaryDim > headSize || !sectionAttr ||
      sectionAttr.size() != 3) {
    return std::nullopt;
  }
  int64_t half = 0;
  for (Attribute a : sectionAttr) {
    auto s = dyn_cast<IntegerAttr>(a);
    if (!s || s.getValue().getSExtValue() < 0) {
      return std::nullopt;
    }
    chain.section.push_back(s.getValue().getSExtValue());
    half += chain.section.back();
  }
  if (2 * half != rotaryDim) {
    return std::nullopt;
  }
  chain.positionIds = rope->getOperand(1);
  chain.cos = rope->getOperand(2);
  chain.sin = rope->getOperand(3);

  Operation *outer = producer(rope->getOperand(0), "onnx.Reshape");
  if (!outer || intAttr(outer, "allowzero", 0) != 0 ||
      !outer->getResult(0).hasOneUse() ||
      !isConstant(outer->getOperand(1), {0, -1, heads * headSize})) {
    return std::nullopt;
  }
  Operation *norm =
      customProducer(outer->getOperand(0), "SimplifiedLayerNormalization");
  auto epsilon = norm ? norm->getAttrOfType<FloatAttr>("epsilon") : nullptr;
  if (!norm || !epsilon || norm->getNumOperands() != 2 ||
      intAttr(norm, "stash_type", 1) != 1 || !norm->getResult(0).hasOneUse() ||
      llvm::any_of(norm->getResults().drop_front(),
                   [](Value r) { return !r.use_empty(); })) {
    return std::nullopt;
  }
  const int64_t axis = intAttr(norm, "axis", -1);
  if (axis != -1 && axis != 2) {
    return std::nullopt;
  }
  chain.epsilon = epsilon.getValue();
  chain.weight = norm->getOperand(1);

  Operation *inner = producer(norm->getOperand(0), "onnx.Reshape");
  if (!inner || intAttr(inner, "allowzero", 0) != 0 ||
      !inner->getResult(0).hasOneUse() ||
      !isConstant(inner->getOperand(1), {0, -1, headSize})) {
    return std::nullopt;
  }
  chain.input = inner->getOperand(0);
  if (chain.input.getType() != v.getType()) {
    return std::nullopt;
  }
  chain.ops = {rope, outer, norm, inner};
  return chain;
}

/// q / k = MRotaryEmbedding(per-head SimplifiedLayerNormalization(x)) ->
/// x plus the prologue inputs.
struct DsaPrologueFold : public RewritePattern {
  DsaPrologueFold(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isFoldableDsa(op) || !optionalOperand(op, kKeyIdx) ||
        optionalOperand(op, kDsaPrologueIdx) ||
        op->hasAttr(kDsaNormEpsilonAttr)) {
      return rewriter.notifyMatchFailure(op, "not an unfolded unpacked DSA");
    }
    const int64_t numHeads = intAttr(op, "num_heads", 0);
    const int64_t kvNumHeads = intAttr(op, "kv_num_heads", 0);
    auto queryType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
    if (numHeads <= 0 || kvNumHeads <= 0 || !queryType ||
        queryType.getRank() != 3 || queryType.isDynamicDim(2) ||
        queryType.getDimSize(2) % numHeads != 0) {
      return rewriter.notifyMatchFailure(op, "unexpected query shape");
    }
    const int64_t headSize = queryType.getDimSize(2) / numHeads;
    std::optional<NormRopeChain> q =
        matchChain(op->getOperand(0), numHeads, headSize);
    std::optional<NormRopeChain> k =
        matchChain(op->getOperand(kKeyIdx), kvNumHeads, headSize);
    if (!q || !k) {
      return rewriter.notifyMatchFailure(op, "q/k are not norm+MRoPE chains");
    }
    if (q->positionIds != k->positionIds || q->cos != k->cos ||
        q->sin != k->sin || q->section != k->section ||
        q->layout != k->layout || !q->epsilon.bitwiseIsEqual(k->epsilon)) {
      return rewriter.notifyMatchFailure(
          op, "q and k chains must share the rotary inputs and attributes");
    }

    SmallVector<Value> operands = llvm::to_vector(op->getOperands());
    operands.resize(kDsaMaxInputs, op->getOperand(kAuxKeyIdx));
    operands[0] = q->input;
    operands[kKeyIdx] = k->input;
    llvm::copy(
        ArrayRef<Value>{q->weight, k->weight, q->positionIds, q->cos, q->sin},
        operands.begin() + kDsaPrologueIdx);
    Builder &b = rewriter;
    Operation *folded = rebuild(
        rewriter, op, operands,
        {b.getNamedAttr(kDsaNormEpsilonAttr,
                        b.getF32FloatAttr(q->epsilon.convertToFloat())),
         b.getNamedAttr(kDsaMropeSectionAttr,
                        b.getDenseI64ArrayAttr(q->section)),
         b.getNamedAttr(kDsaMropeLayoutAttr, b.getI64IntegerAttr(q->layout))});
    rewriter.replaceOp(op, folded->getResults());
    for (const NormRopeChain *chain : {&*q, &*k}) {
      for (Operation *dead : chain->ops) {
        rewriter.eraseOp(dead);
      }
    }
    return success();
  }
};

/// q, k, v = Split(qkv) -> packed query qkv.
struct DsaPackedFold : public RewritePattern {
  DsaPackedFold(MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (!isFoldableDsa(op) || !optionalOperand(op, kKeyIdx) ||
        !optionalOperand(op, kValueIdx)) {
      return rewriter.notifyMatchFailure(op, "not an unpacked DSA");
    }
    Operation *split = producer(op->getOperand(0), "onnx.Split");
    if (!split || split->getNumResults() != 3 ||
        op->getOperand(0) != split->getResult(0) ||
        op->getOperand(kKeyIdx) != split->getResult(1) ||
        op->getOperand(kValueIdx) != split->getResult(2)) {
      return rewriter.notifyMatchFailure(op, "q, k, v are not one Split");
    }
    Value qkv = split->getOperand(0);
    auto qkvType = dyn_cast<RankedTensorType>(qkv.getType());
    const int64_t axis = intAttr(split, "axis", 0);
    if (!qkvType || qkvType.getRank() != 3 || (axis != -1 && axis != 2)) {
      return rewriter.notifyMatchFailure(op, "Split is not over the last axis");
    }
    const int64_t numHeads = intAttr(op, "num_heads", 0);
    const int64_t kvNumHeads = intAttr(op, "kv_num_heads", 0);
    auto width = [](Value v) {
      auto type = dyn_cast<RankedTensorType>(v.getType());
      return type && type.getRank() == 3 ? type.getDimSize(2)
                                         : ShapedType::kDynamic;
    };
    const int64_t qWidth = width(split->getResult(0));
    const int64_t kvWidth = width(split->getResult(1));
    if (numHeads <= 0 || kvNumHeads <= 0 || ShapedType::isDynamic(qWidth) ||
        qWidth % numHeads != 0 || qWidth / numHeads * kvNumHeads != kvWidth ||
        width(split->getResult(2)) != kvWidth ||
        qkvType.getDimSize(2) != qWidth + 2 * kvWidth) {
      return rewriter.notifyMatchFailure(
          op, "Split widths are not N*D, kvN*D, kvN*D");
    }

    SmallVector<Value> operands = llvm::to_vector(op->getOperands());
    operands[0] = qkv;
    operands[kKeyIdx] = op->getOperand(kAuxKeyIdx);
    operands[kValueIdx] = op->getOperand(kAuxKeyIdx);
    Operation *folded = rebuild(rewriter, op, operands);
    rewriter.replaceOp(op, folded->getResults());
    if (split->use_empty()) {
      rewriter.eraseOp(split);
    }
    return success();
  }
};

} // namespace

void populateDsaPrologueFoldPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<DsaGateFold, DsaPrologueFold, DsaPackedFold>(ctx);
}

} // namespace hip
} // namespace mlir
