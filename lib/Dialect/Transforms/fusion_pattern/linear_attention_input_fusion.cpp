/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- linear_attention_input_fusion.cpp - LA input prep into the op ------===//
//
// At decode a gated-delta-net layer spends a dozen tiny launches turning its
// projections into linear-attention inputs. hip.linear_attention can take
// them raw and prepare them in one launch (wrap_linear_attention), so four
// independent patterns, all rooted at the op, fold them in:
//
//   Before:
//     %q  = tensor.extract_slice %qkv[0, 0, 0] [B, T, Hq*dk]
//     %k  = tensor.extract_slice %qkv[0, 0, Hq*dk] [B, T, Nk*dk]
//     %v  = tensor.extract_slice %qkv[0, 0, (Hq+Nk)*dk] [B, T, Hkv*dv]
//     %qn = collapse(hip.rms_norm(expand(%q), %w))           (likewise %kn)
//     %qs = hip.mul(%qn, splat<s>)
//     %g  = hip.cast<f16>(hip.mul(%A, hip.softplus(
//               hip.add(hip.cast<f32>(%a), %dt_bias))))
//     %b  = hip.sigmoid(%braw)
//     hip.linear_attention ins(%qs, %kn, %v) decay(%g) beta(%b)
//   After:
//     hip.linear_attention ins(%qkv) decay(%a) beta(%braw)
//         decay_softplus(%dt_bias, %A) qk_norm(%w)
//         {qk_norm_epsilon = eps, q_scale = s, beta_sigmoid}
//
// The op reproduces each folded kernel's rounding (see its description), so
// the rewrite is exact. Shape queries of the erased values only ask for the
// leading [B, T] axes, which they share with the new operands.
//
//===----------------------------------------------------------------------===//

#include "linear_attention_input_fusion.hpp"

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include <optional>

namespace hip {
namespace fusion_transform {
namespace {

using mlir::hip::AddOp;
using mlir::hip::CastOp;
using mlir::hip::ConstantOp;
using mlir::hip::LinearAttentionOp;
using mlir::hip::MulOp;
using mlir::hip::RmsNormOp;
using mlir::hip::SigmoidOp;
using mlir::hip::SoftplusOp;

// ONNX TensorProto element types used by hip.cast's `to`.
constexpr int64_t kOnnxFloat = 1;
constexpr int64_t kOnnxFloat16 = 10;

// True when every use is either `consumer` or a tensor.dim.
bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

template <typename OpT>
bool matchEither(mlir::Value lhs, mlir::Value rhs, OpT &op,
                 mlir::Value &other) {
  if ((op = lhs.getDefiningOp<OpT>())) {
    other = rhs;
    return true;
  }
  if ((op = rhs.getDefiningOp<OpT>())) {
    other = lhs;
    return true;
  }
  return false;
}

mlir::RankedTensorType rankedType(mlir::Value v) {
  return mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
}

// Value of an inline floating-point splat hip.constant, widened to f32.
std::optional<float> splatValue(mlir::Value v) {
  auto cst = v.getDefiningOp<ConstantOp>();
  if (!cst)
    return std::nullopt;
  auto dense =
      mlir::dyn_cast_if_present<mlir::DenseElementsAttr>(cst.getValueAttr());
  if (!dense || !dense.isSplat() ||
      !mlir::isa<mlir::FloatType>(dense.getElementType()))
    return std::nullopt;
  llvm::APFloat value = dense.getSplatValue<llvm::APFloat>();
  bool losesInfo = false;
  value.convert(llvm::APFloat::IEEEsingle(),
                llvm::APFloat::rmNearestTiesToEven, &losesInfo);
  return value.convertToFloat();
}

// Shape queries of `old` ask only for axes it shares with `replacement`:
// static axes, or dynamic ones ahead of the replacement's last axis.
bool shapeQueriesRetargetable(mlir::Value old, mlir::Value replacement) {
  auto oldType = rankedType(old);
  auto newType = rankedType(replacement);
  if (!oldType || !newType)
    return false;
  for (mlir::Operation *user : old.getUsers()) {
    auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(user);
    if (!dim)
      continue;
    std::optional<int64_t> index = dim.getConstantIndex();
    if (!index || *index >= oldType.getRank())
      return false;
    if (oldType.isDynamicDim(*index) && *index >= newType.getRank() - 1)
      return false;
  }
  return true;
}

void retargetShapeQueries(mlir::PatternRewriter &rewriter, mlir::Value old,
                          mlir::Value replacement) {
  auto oldType = mlir::cast<mlir::RankedTensorType>(old.getType());
  for (mlir::OpOperand &use : llvm::make_early_inc_range(old.getUses())) {
    auto dim = mlir::dyn_cast<mlir::tensor::DimOp>(use.getOwner());
    if (!dim)
      continue;
    int64_t index = *dim.getConstantIndex();
    rewriter.setInsertionPoint(dim);
    if (oldType.isDynamicDim(index))
      rewriter.modifyOpInPlace(dim, [&] { use.set(replacement); });
    else
      rewriter.replaceOp(dim, mlir::arith::ConstantIndexOp::create(
                                  rewriter, dim.getLoc(),
                                  oldType.getDimSize(index)));
  }
}

// Erase `chain` (consumers before producers, each left with no uses) and
// the destinations only they used.
void eraseChain(mlir::PatternRewriter &rewriter,
                llvm::ArrayRef<mlir::Operation *> chain) {
  llvm::SmallVector<mlir::Operation *> inits;
  for (mlir::Operation *op : chain) {
    if (auto dps = mlir::dyn_cast<mlir::DestinationStyleOpInterface>(op))
      for (mlir::Value init : dps.getDpsInits())
        if (mlir::Operation *def = init.getDefiningOp())
          inits.push_back(def);
    rewriter.eraseOp(op);
  }
  llvm::SmallPtrSet<mlir::Operation *, 8> dropped;
  for (mlir::Operation *def : inits)
    if (!dropped.contains(def) && mlir::isOpTriviallyDead(def)) {
      dropped.insert(def);
      rewriter.eraseOp(def);
    }
}

// The rewritable parts of a hip.linear_attention.
struct LaParts {
  mlir::Value query, key, value, decay, beta, decayBias, decayScale,
      qkNormWeight;
  mlir::FloatAttr qkNormEpsilon, qScale;
  mlir::UnitAttr betaSigmoid;

  explicit LaParts(LinearAttentionOp op)
      : query(op.getQuery()), key(op.getKey()), value(op.getValue()),
        decay(op.getDecay()), beta(op.getBeta()),
        decayBias(op.getDecayBias()), decayScale(op.getDecayScale()),
        qkNormWeight(op.getQkNormWeight()),
        qkNormEpsilon(op.getQkNormEpsilonAttr()), qScale(op.getQScaleAttr()),
        betaSigmoid(op.getBetaSigmoidAttr()) {}
};

void replaceWith(mlir::PatternRewriter &rewriter, LinearAttentionOp op,
                 const LaParts &p) {
  rewriter.setInsertionPoint(op);
  auto fused = LinearAttentionOp::create(
      rewriter, op.getLoc(), op->getResultTypes(), op.getCtx(), p.query,
      p.key, p.value, op.getPastState(), p.decay, p.beta, p.decayBias,
      p.decayScale, p.qkNormWeight, op.getOutput(), op.getPresentState(),
      op.getQNumHeadsAttr(), op.getKvNumHeadsAttr(), op.getScaleAttr(),
      op.getChunkSizeAttr(), op.getUpdateRuleAttr(), p.qkNormEpsilon,
      p.qScale, p.betaSigmoid);
  rewriter.replaceOp(op, fused->getResults());
}

bool isF16(mlir::Value v) {
  auto t = rankedType(v);
  return t && t.getElementType().isF16();
}

//===----------------------------------------------------------------------===//
// q/k head norms and q scale
//===----------------------------------------------------------------------===//

// collapse(hip.rms_norm(expand(x) : [..., H, d], weight [d])) with the
// expand splitting and the collapse merging only the last axis.
struct HeadNorm {
  mlir::Value input;
  mlir::Value weight;
  float epsilon;
  llvm::SmallVector<mlir::Operation *, 3> chain; // collapse, norm, expand
};

// q and k norms use one weight: the same value or equal splats.
bool sameWeight(mlir::Value a, mlir::Value b) {
  if (a == b)
    return true;
  std::optional<float> sa = splatValue(a);
  std::optional<float> sb = splatValue(b);
  return sa && sb && *sa == *sb && a.getType() == b.getType();
}

bool splitsLastAxisOnly(llvm::ArrayRef<mlir::ReassociationIndices> groups,
                        int64_t rank) {
  if ((int64_t)groups.size() != rank)
    return false;
  for (int64_t i = 0; i + 1 < rank; ++i)
    if (groups[i].size() != 1 || groups[i][0] != i)
      return false;
  return groups.back().size() == 2 && groups.back()[0] == rank - 1;
}

std::optional<HeadNorm> matchHeadNorm(mlir::Value v, mlir::Operation *consumer,
                                      int64_t dk, mlir::Value ctx) {
  auto collapse = v.getDefiningOp<mlir::tensor::CollapseShapeOp>();
  if (!collapse || !hasSingleValueUse(v, consumer))
    return std::nullopt;
  auto norm = collapse.getSrc().getDefiningOp<RmsNormOp>();
  if (!norm || norm.getCtx() != ctx ||
      !hasSingleValueUse(norm->getResult(0), collapse))
    return std::nullopt;
  auto expand = norm.getInput().getDefiningOp<mlir::tensor::ExpandShapeOp>();
  if (!expand || !hasSingleValueUse(expand.getResult(), norm))
    return std::nullopt;

  auto inType = rankedType(expand.getSrc());
  auto headType = rankedType(expand.getResult());
  if (!inType || !headType || !isF16(expand.getSrc()) ||
      collapse.getResult().getType() != inType ||
      headType.getDimSize(headType.getRank() - 1) != dk ||
      !splitsLastAxisOnly(collapse.getReassociationIndices(),
                          inType.getRank()) ||
      !splitsLastAxisOnly(expand.getReassociationIndices(), inType.getRank()))
    return std::nullopt;
  const int64_t axis = norm.getAxis();
  if (axis != -1 && axis != headType.getRank() - 1)
    return std::nullopt;
  auto weightType = rankedType(norm.getScale());
  if (!weightType || weightType.getRank() != 1 ||
      weightType.getDimSize(0) != dk || !isF16(norm.getScale()))
    return std::nullopt;

  HeadNorm match{expand.getSrc(), norm.getScale(),
                 norm.getEpsilon().convertToFloat(), {}};
  match.chain = {collapse, norm, expand};
  return match;
}

struct QkNormFusion : public mlir::OpRewritePattern<LinearAttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LinearAttentionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.isPackedQKV() || op.getQkNormWeight() ||
        op.getQScale().convertToFloat() != 1.f)
      return rewriter.notifyMatchFailure(op, "already prepared");
    auto stateType = rankedType(op.getPresentState());
    if (!stateType || stateType.getRank() != 4 ||
        stateType.isDynamicDim(2) || !isF16(op.getQuery()))
      return rewriter.notifyMatchFailure(op, "unsupported state / dtype");
    const int64_t dk = stateType.getDimSize(2);
    mlir::Value ctx = op.getCtx();

    // Optional q scale: hip.mul(qn, splat) with a one-element f16 constant.
    mlir::Value q = op.getQuery();
    MulOp qMul = q.getDefiningOp<MulOp>();
    std::optional<float> qScale;
    if (qMul && qMul.getCtx() == ctx && hasSingleValueUse(q, op)) {
      for (auto [s, x] : {std::pair{qMul.getRhs(), qMul.getLhs()},
                          std::pair{qMul.getLhs(), qMul.getRhs()}}) {
        auto sType = rankedType(s);
        if (sType && sType.getNumElements() == 1 && isF16(s) &&
            x.getType() == q.getType() && (qScale = splatValue(s))) {
          q = x;
          break;
        }
      }
    }
    if (!qScale)
      qMul = nullptr;

    std::optional<HeadNorm> qNorm =
        matchHeadNorm(q, qMul ? qMul.getOperation() : op.getOperation(), dk,
                      ctx);
    std::optional<HeadNorm> kNorm = matchHeadNorm(op.getKey(), op, dk, ctx);
    if (!qNorm || !kNorm)
      return rewriter.notifyMatchFailure(op, "q/k are not head-normalized");
    if (!sameWeight(qNorm->weight, kNorm->weight) ||
        qNorm->epsilon != kNorm->epsilon)
      return rewriter.notifyMatchFailure(op, "q/k norms differ");

    llvm::SmallVector<mlir::Operation *> qChain;
    if (qMul)
      qChain.push_back(qMul);
    qChain.append(qNorm->chain.begin(), qNorm->chain.end());
    for (auto [chain, input] :
         {std::pair{llvm::ArrayRef(qChain), qNorm->input},
          std::pair{llvm::ArrayRef(kNorm->chain), kNorm->input}})
      for (mlir::Operation *c : chain)
        if (!shapeQueriesRetargetable(c->getResult(0), input))
          return rewriter.notifyMatchFailure(op, "unexpected shape query");

    for (auto [chain, input] :
         {std::pair{llvm::ArrayRef(qChain), qNorm->input},
          std::pair{llvm::ArrayRef(kNorm->chain), kNorm->input}})
      for (mlir::Operation *c : chain)
        retargetShapeQueries(rewriter, c->getResult(0), input);

    LaParts parts(op);
    parts.query = qNorm->input;
    parts.key = kNorm->input;
    parts.qkNormWeight = qNorm->weight;
    parts.qkNormEpsilon = rewriter.getF32FloatAttr(qNorm->epsilon);
    if (qScale)
      parts.qScale = rewriter.getF32FloatAttr(*qScale);
    replaceWith(rewriter, op, parts);
    eraseChain(rewriter, qChain);
    eraseChain(rewriter, kNorm->chain);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Packed QKV
//===----------------------------------------------------------------------===//

// `v` = extract_slice(src)[0.., offset] covering src's leading axes whole.
// Returns src and the last-axis offset.
std::optional<std::pair<mlir::Value, int64_t>> lastAxisSlice(mlir::Value v) {
  auto slice = v.getDefiningOp<mlir::tensor::ExtractSliceOp>();
  if (!slice)
    return std::nullopt;
  mlir::Value src = slice.getSource();
  auto srcType = rankedType(src);
  auto resType = rankedType(v);
  if (!srcType || !resType || srcType.getRank() != resType.getRank() ||
      srcType.isDynamicDim(srcType.getRank() - 1) ||
      !mlir::areAllConstantIntValue(slice.getMixedStrides(), 1))
    return std::nullopt;
  const int64_t rank = srcType.getRank();
  llvm::SmallVector<mlir::OpFoldResult> offsets = slice.getMixedOffsets();
  llvm::SmallVector<mlir::OpFoldResult> sizes = slice.getMixedSizes();
  for (int64_t i = 0; i + 1 < rank; ++i) {
    if (!mlir::isConstantIntValue(offsets[i], 0))
      return std::nullopt;
    if (std::optional<int64_t> size = mlir::getConstantIntValue(sizes[i])) {
      if (srcType.isDynamicDim(i) || *size != srcType.getDimSize(i))
        return std::nullopt;
      continue;
    }
    auto size = llvm::dyn_cast_if_present<mlir::Value>(sizes[i]);
    auto dim = size ? size.getDefiningOp<mlir::tensor::DimOp>() : nullptr;
    if (!dim || dim.getSource() != src || dim.getConstantIndex() != i)
      return std::nullopt;
  }
  std::optional<int64_t> offset = mlir::getConstantIntValue(offsets.back());
  if (!offset)
    return std::nullopt;
  return std::pair{src, *offset};
}

struct PackedQkvFusion : public mlir::OpRewritePattern<LinearAttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LinearAttentionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op.isPackedQKV() || !isF16(op.getQuery()))
      return rewriter.notifyMatchFailure(op, "already packed / not f16");
    auto q = lastAxisSlice(op.getQuery());
    auto k = lastAxisSlice(op.getKey());
    auto v = lastAxisSlice(op.getValue());
    if (!q || !k || !v || q->first != k->first || q->first != v->first)
      return rewriter.notifyMatchFailure(op, "q/k/v are not slices of one "
                                             "tensor");
    mlir::Value src = q->first;
    auto lastDim = [](mlir::Value x) {
      auto t = mlir::cast<mlir::RankedTensorType>(x.getType());
      return t.getDimSize(t.getRank() - 1);
    };
    const int64_t qWidth = lastDim(op.getQuery());
    const int64_t kWidth = lastDim(op.getKey());
    const int64_t vWidth = lastDim(op.getValue());
    if (mlir::ShapedType::isDynamic(qWidth) ||
        mlir::ShapedType::isDynamic(kWidth) ||
        mlir::ShapedType::isDynamic(vWidth) || q->second != 0 ||
        k->second != qWidth || v->second != qWidth + kWidth ||
        lastDim(src) != qWidth + kWidth + vWidth)
      return rewriter.notifyMatchFailure(op, "slices are not q|k|v");

    for (mlir::Value x : {op.getQuery(), op.getKey(), op.getValue()})
      if (!shapeQueriesRetargetable(x, src))
        return rewriter.notifyMatchFailure(op, "unexpected shape query");
    for (mlir::Value x : {op.getQuery(), op.getKey(), op.getValue()})
      retargetShapeQueries(rewriter, x, src);

    LaParts parts(op);
    parts.query = src;
    parts.key = nullptr;
    parts.value = nullptr;
    replaceWith(rewriter, op, parts);
    return mlir::success();
  }
};

//===----------------------------------------------------------------------===//
// Softplus decay and sigmoid beta
//===----------------------------------------------------------------------===//

// An f32 [heads] vector, broadcast over [B, T].
bool isPerHeadVector(mlir::Value v, int64_t heads) {
  auto t = rankedType(v);
  return t && t.getRank() == 1 && t.getDimSize(0) == heads &&
         t.getElementType().isF32();
}

struct SoftplusDecayFusion : public mlir::OpRewritePattern<LinearAttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LinearAttentionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value decay = op.getDecay();
    if (!decay || op.getDecayBias() || !isF16(op.getQuery()))
      return rewriter.notifyMatchFailure(op, "no raw decay to fold");
    const int64_t heads = op.getKvNumHeads();
    auto decayType = rankedType(decay);
    if (!decayType || !isF16(decay) ||
        decayType.getDimSize(decayType.getRank() - 1) != heads)
      return rewriter.notifyMatchFailure(op, "decay is not per-head f16");
    mlir::Value ctx = op.getCtx();

    auto toF16 = decay.getDefiningOp<CastOp>();
    if (!toF16 || toF16.getTo() != kOnnxFloat16 ||
        !hasSingleValueUse(decay, op))
      return rewriter.notifyMatchFailure(op, "decay is not a cast to f16");
    MulOp mul = toF16.getInput().getDefiningOp<MulOp>();
    if (!mul || !hasSingleValueUse(mul->getResult(0), toF16))
      return rewriter.notifyMatchFailure(op, "no decay scale");
    SoftplusOp softplus;
    mlir::Value scale;
    if (!matchEither(mul.getLhs(), mul.getRhs(), softplus, scale) ||
        !hasSingleValueUse(softplus->getResult(0), mul))
      return rewriter.notifyMatchFailure(op, "no softplus");
    AddOp add = softplus.getX().getDefiningOp<AddOp>();
    if (!add || !hasSingleValueUse(add->getResult(0), softplus))
      return rewriter.notifyMatchFailure(op, "no decay bias");
    CastOp toF32;
    mlir::Value bias;
    if (!matchEither(add.getLhs(), add.getRhs(), toF32, bias) ||
        toF32.getTo() != kOnnxFloat ||
        !hasSingleValueUse(toF32->getResult(0), add))
      return rewriter.notifyMatchFailure(op, "no cast to f32");
    mlir::Value raw = toF32.getInput();

    if (raw.getType() != decay.getType() || !isPerHeadVector(bias, heads) ||
        !isPerHeadVector(scale, heads))
      return rewriter.notifyMatchFailure(op, "decay operands mismatch");
    mlir::Operation *chain[] = {toF16, mul, softplus, add, toF32};
    for (mlir::Operation *c : chain) {
      if (c->getOperand(0) != ctx)
        return rewriter.notifyMatchFailure(op, "ops must share a context");
      auto cType = rankedType(c->getResult(0));
      if (!cType || cType.getShape() != decayType.getShape())
        return rewriter.notifyMatchFailure(op, "decay chain broadcasts");
      if (!shapeQueriesRetargetable(c->getResult(0), raw))
        return rewriter.notifyMatchFailure(op, "unexpected shape query");
    }

    for (mlir::Operation *c : chain)
      retargetShapeQueries(rewriter, c->getResult(0), raw);
    LaParts parts(op);
    parts.decay = raw;
    parts.decayBias = bias;
    parts.decayScale = scale;
    replaceWith(rewriter, op, parts);
    eraseChain(rewriter, chain);
    return mlir::success();
  }
};

struct SigmoidBetaFusion : public mlir::OpRewritePattern<LinearAttentionOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(LinearAttentionOp op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Value beta = op.getBeta();
    if (!beta || op.getBetaSigmoid() || !isF16(op.getQuery()) || !isF16(beta))
      return rewriter.notifyMatchFailure(op, "no raw beta to fold");
    auto sigmoid = beta.getDefiningOp<SigmoidOp>();
    if (!sigmoid || sigmoid.getCtx() != op.getCtx() ||
        !hasSingleValueUse(beta, op) || sigmoid.getX().getType() != beta.getType() ||
        !shapeQueriesRetargetable(beta, sigmoid.getX()))
      return rewriter.notifyMatchFailure(op, "beta is not a sigmoid");

    mlir::Value raw = sigmoid.getX();
    retargetShapeQueries(rewriter, beta, raw);
    LaParts parts(op);
    parts.beta = raw;
    parts.betaSigmoid = rewriter.getUnitAttr();
    replaceWith(rewriter, op, parts);
    mlir::Operation *chain[] = {sigmoid};
    eraseChain(rewriter, chain);
    return mlir::success();
  }
};

} // namespace

void populateLinearAttentionInputFusionPatterns(
    mlir::RewritePatternSet &patterns, mlir::PatternBenefit benefit) {
  patterns.add<QkNormFusion, PackedQkvFusion, SoftplusDecayFusion,
               SigmoidBetaFusion>(patterns.getContext(), benefit);
}

} // namespace fusion_transform
} // namespace hip
