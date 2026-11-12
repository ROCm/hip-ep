/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DecomposedAttentionFusion.cpp - Primitive SDPA -> MultiHeadAttention
//===//
//
// PyTorch exports scaled-dot-product attention without a fused op as
// primitives over the heads-major layout:
//
//   q4 = Transpose(Reshape(q, [B, Sq, N, H]), perm = [0, 2, 1, 3])   BNSH
//   kT = Transpose(Reshape(k, [B, Skv, N, H]), perm = [0, 2, 3, 1])  BNHS
//   v4 = Transpose(Reshape(v, [B, Skv, N, H]), perm = [0, 2, 1, 3])  BNSH
//   P  = Softmax(MatMul(q4 * cq, kT * ck) * cs, axis = -1)
//   O  = Transpose(MatMul(P, v4), perm = [0, 2, 1, 3])               BSNH
//
// The scale may sit on either projection (diffusers splits 1/sqrt(H) into
// H^-1/4 on each), on the scores, or nowhere, and may be spelled Mul, Div or
// Mul(Reciprocal(c)). K^T may also arrive as Transpose(BNSH, [0, 1, 3, 2]) or
// already folded into the score MatMul as `hipdnn.transB`.
//
// Lowered as written, the scores and the probabilities are materialised in
// DRAM. At Sq = Skv = 4096 with 8 heads and batch 2 each is 512 MiB of fp16,
// so the two MatMuls and the Softmax move gigabytes per layer for a matrix that
// exists only to be reduced. com.microsoft.MultiHeadAttention carries the same
// computation with Q, K and V in their projected [B, S, N*H] layout, and its
// runtime runs a flash kernel that keeps the softmax state in registers. The
// rewrite targets that op:
//
//   O = Reshape(MultiHeadAttention(Reshape(q), Reshape(k), Reshape(v)),
//               [B, Sq, N, H])
//
// Every Reshape here is between two row-major views of the same bytes, so the
// projections feed MultiHeadAttention without a copy.
//
// Only shapes the flash path runs are taken (fp16, static, head dim <= 256,
// Sq > 1); anything else keeps its primitive lowering. Rooted on onnx.Softmax
// and run in the pre-lowering set, before convertComputeOps rewrites the
// chain's members into hip.* ops independently.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cmath>
#include <optional>

#undef DEBUG_TYPE
#define DEBUG_TYPE "decomposed-attention-fusion"

STATISTIC(NumDecomposedAttentionFused,
          "Number of primitive attention chains rewritten to "
          "com.microsoft.MultiHeadAttention");

namespace mlir {
namespace hip {

namespace {

// The flash path's head-dimension ceiling (wrap_multi_head_attention).
constexpr int64_t kMaxFlashHeadDim = 256;

bool isOnnxOp(mlir::Operation *op, llvm::StringRef name) {
  return op && op->getName().getStringRef() == name;
}

/// A single-element floating-point constant, looking through the value
/// preserving wrappers the pre-lowering set and exporters put around scale
/// literals.
std::optional<double> getScalarConstant(mlir::Value v) {
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (isOnnxOp(def, "onnx.Constant")) {
    auto attr =
        mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(def->getAttr("value"));
    if (!attr || attr.getNumElements() != 1 ||
        !mlir::isa<mlir::FloatType>(attr.getElementType()))
      return std::nullopt;
    return (*attr.getValues<llvm::APFloat>().begin()).convertToDouble();
  }
  if ((isOnnxOp(def, "onnx.Cast") || isOnnxOp(def, "onnx.CastLike")) &&
      def->getNumOperands() >= 1)
    return getScalarConstant(def->getOperand(0));
  if (isOnnxOp(def, "onnx.Reciprocal") && def->getNumOperands() == 1) {
    auto inner = getScalarConstant(def->getOperand(0));
    if (!inner || *inner == 0.0)
      return std::nullopt;
    return 1.0 / *inner;
  }
  if (isOnnxOp(def, "onnx.Sqrt") && def->getNumOperands() == 1) {
    auto inner = getScalarConstant(def->getOperand(0));
    if (!inner || *inner < 0.0)
      return std::nullopt;
    return std::sqrt(*inner);
  }
  return std::nullopt;
}

std::optional<llvm::SmallVector<int64_t>> getTransposePerm(mlir::Operation *op,
                                                           int64_t rank) {
  llvm::SmallVector<int64_t> perm;
  if (auto dense = op->getAttrOfType<mlir::DenseIntElementsAttr>("perm")) {
    for (const llvm::APInt &v : dense.getValues<llvm::APInt>())
      perm.push_back(v.getSExtValue());
  } else if (auto array = op->getAttrOfType<mlir::ArrayAttr>("perm")) {
    for (mlir::Attribute elem : array) {
      auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(elem);
      if (!intAttr)
        return std::nullopt;
      perm.push_back(intAttr.getValue().getSExtValue());
    }
  } else {
    // ONNX default: reverse the dimensions.
    for (int64_t i : llvm::seq<int64_t>(0, rank))
      perm.push_back(rank - 1 - i);
  }
  if (static_cast<int64_t>(perm.size()) != rank)
    return std::nullopt;
  return perm;
}

/// One operand of the attention, traced back through the layout and scale
/// ops between it and the tensor that actually holds its data. `perm` is the
/// composite permutation: dimension i of the traced value is dimension
/// perm[i] of `base`.
struct TracedOperand {
  mlir::Value base;
  llvm::SmallVector<int64_t> perm;
  double scale = 1.0;
};

/// A scalar Mul / Div that does not change its tensor operand's type, as
/// (tensor operand, factor).
std::optional<std::pair<mlir::Value, double>>
matchScalarScale(mlir::Operation *op) {
  if (!op || op->getNumOperands() != 2 || op->getNumResults() != 1)
    return std::nullopt;
  mlir::Type resultType = op->getResult(0).getType();
  if (isOnnxOp(op, "onnx.Mul")) {
    for (unsigned i : {0u, 1u}) {
      mlir::Value tensor = op->getOperand(i);
      if (tensor.getType() != resultType)
        continue;
      if (auto c = getScalarConstant(op->getOperand(1 - i)))
        return std::make_pair(tensor, *c);
    }
    return std::nullopt;
  }
  if (isOnnxOp(op, "onnx.Div") && op->getOperand(0).getType() == resultType) {
    auto c = getScalarConstant(op->getOperand(1));
    if (c && *c != 0.0)
      return std::make_pair(op->getOperand(0), 1.0 / *c);
  }
  return std::nullopt;
}

TracedOperand traceOperand(mlir::Value v, int64_t rank) {
  TracedOperand t;
  t.base = v;
  for (int64_t i : llvm::seq<int64_t>(0, rank))
    t.perm.push_back(i);
  while (mlir::Operation *def = t.base.getDefiningOp()) {
    if (isOnnxOp(def, "onnx.Transpose") && def->getNumOperands() == 1) {
      auto perm = getTransposePerm(def, rank);
      if (!perm)
        break;
      llvm::SmallVector<int64_t> composed;
      for (int64_t d : t.perm)
        composed.push_back((*perm)[d]);
      t.perm = std::move(composed);
      t.base = def->getOperand(0);
      continue;
    }
    if (auto scaled = matchScalarScale(def)) {
      t.scale *= scaled->second;
      t.base = scaled->first;
      continue;
    }
    break;
  }
  return t;
}

/// A static rank-4 fp16 tensor type, or null.
mlir::RankedTensorType staticF16Rank4(mlir::Value v) {
  auto type = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!type || type.getRank() != 4 || !type.hasStaticShape() ||
      !type.getElementType().isF16())
    return nullptr;
  return type;
}

bool isPerm(llvm::ArrayRef<int64_t> perm, llvm::ArrayRef<int64_t> expected) {
  return perm == expected;
}

mlir::IntegerAttr si64Attr(mlir::MLIRContext *ctx, int64_t v) {
  return mlir::IntegerAttr::get(
      mlir::IntegerType::get(ctx, 64, mlir::IntegerType::Signed), v);
}

/// `v` viewed as `shape`. Reuses `v`, or the input of the Reshape producing
/// it, when either already has that shape; otherwise emits an onnx.Reshape.
mlir::Value reshapeTo(mlir::PatternRewriter &rewriter, mlir::Location loc,
                      mlir::Value v, llvm::ArrayRef<int64_t> shape) {
  auto type = mlir::cast<mlir::RankedTensorType>(v.getType());
  auto target = mlir::RankedTensorType::get(shape, type.getElementType());
  if (type == target)
    return v;
  if (mlir::Operation *def = v.getDefiningOp();
      isOnnxOp(def, "onnx.Reshape") && def->getOperand(0).getType() == target)
    return def->getOperand(0);

  auto shapeType = mlir::RankedTensorType::get(
      {static_cast<int64_t>(shape.size())}, rewriter.getI64Type());
  mlir::OperationState constState(loc, "onnx.Constant");
  constState.addTypes(shapeType);
  constState.addAttribute("value",
                          mlir::DenseElementsAttr::get(shapeType, shape));
  mlir::Value shapeValue = rewriter.create(constState)->getResult(0);

  mlir::OperationState state(loc, "onnx.Reshape");
  state.addOperands({v, shapeValue});
  state.addTypes(target);
  state.addAttribute("allowzero", si64Attr(rewriter.getContext(), 0));
  return rewriter.create(state)->getResult(0);
}

/// Append `v`'s producer and then the producers along its data operand, up to
/// the first of `stops` (the values the replacement still reads). Constants
/// are left to the pass's final DCE: they may be shared with other sites.
void collectChain(mlir::Value v, llvm::ArrayRef<mlir::Value> stops,
                  llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  // Longer than any matched chain (MatMul, Softmax, two scales, Mul,
  // Transpose, Transpose, Reshape); only bounds a walk that missed its stop.
  constexpr int kMaxChain = 10;
  for (int i = 0; i < kMaxChain && v && !llvm::is_contained(stops, v); ++i) {
    mlir::Operation *def = v.getDefiningOp();
    if (!def || def->getNumOperands() == 0 || isOnnxOp(def, "onnx.Constant") ||
        llvm::is_contained(ops, def))
      return;
    ops.push_back(def);
    mlir::Value next = def->getOperand(0);
    if (auto scaled = matchScalarScale(def))
      next = scaled->first;
    v = next;
  }
}

/// Erase every op in `ops` that is, or becomes, dead. The chains may share
/// producers, so this runs to a fixed point instead of walking each once.
void eraseDead(mlir::PatternRewriter &rewriter,
               llvm::SmallVectorImpl<mlir::Operation *> &ops) {
  bool changed = true;
  while (changed) {
    changed = false;
    for (mlir::Operation *&op : ops) {
      if (op && op->use_empty()) {
        rewriter.eraseOp(op);
        op = nullptr;
        changed = true;
      }
    }
  }
}

struct DecomposedAttentionToMultiHeadAttention : public mlir::RewritePattern {
  DecomposedAttentionToMultiHeadAttention(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Softmax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *softmax,
                  mlir::PatternRewriter &rewriter) const override {
    if (softmax->getNumOperands() != 1 || softmax->getNumResults() != 1)
      return rewriter.notifyMatchFailure(softmax, "softmax.arity");
    auto scoresType = staticF16Rank4(softmax->getOperand(0));
    if (!scoresType)
      return rewriter.notifyMatchFailure(softmax, "scores.not_static_f16_4d");
    if (auto axis = softmax->getAttrOfType<mlir::IntegerAttr>("axis")) {
      const int64_t a = axis.getValue().getSExtValue();
      if (a != -1 && a != 3)
        return rewriter.notifyMatchFailure(softmax, "softmax.not_last_axis");
    }
    mlir::Value probs = softmax->getResult(0);
    if (!probs.hasOneUse())
      return rewriter.notifyMatchFailure(softmax, "probs.not_one_use");

    // Scores: a score scale may sit between the MatMul and the Softmax.
    double scale = 1.0;
    mlir::Value scores = softmax->getOperand(0);
    while (mlir::Operation *def = scores.getDefiningOp()) {
      auto scaled = matchScalarScale(def);
      if (!scaled || !scores.hasOneUse())
        break;
      scale *= scaled->second;
      scores = scaled->first;
    }
    mlir::Operation *qk = scores.getDefiningOp();
    if (!isOnnxOp(qk, "onnx.MatMul") || qk->getNumOperands() != 2 ||
        !scores.hasOneUse())
      return rewriter.notifyMatchFailure(softmax, "scores.not_matmul");
    if (qk->hasAttr("hipdnn.transA"))
      return rewriter.notifyMatchFailure(softmax, "scores.transA");
    auto transBAttr = qk->getAttrOfType<mlir::IntegerAttr>("hipdnn.transB");
    const bool kTransposed = transBAttr && transBAttr.getInt() != 0;

    mlir::Operation *pv = *probs.getUsers().begin();
    if (!isOnnxOp(pv, "onnx.MatMul") || pv->getNumOperands() != 2 ||
        pv->getOperand(0) != probs || pv->hasAttr("hipdnn.transA") ||
        pv->hasAttr("hipdnn.transB"))
      return rewriter.notifyMatchFailure(softmax, "probs.not_value_matmul");

    TracedOperand q = traceOperand(qk->getOperand(0), 4);
    TracedOperand k = traceOperand(qk->getOperand(1), 4);
    TracedOperand v = traceOperand(pv->getOperand(1), 4);
    // Q and V must be BNSH views of BSNH bases; K^T a BNHS view, or a BNSH
    // view when the score MatMul already carries the transpose.
    const llvm::SmallVector<int64_t> bsnhToBnsh = {0, 2, 1, 3};
    const llvm::SmallVector<int64_t> bsnhToBnhs = {0, 2, 3, 1};
    if (!isPerm(q.perm, bsnhToBnsh) || !isPerm(v.perm, bsnhToBnsh) ||
        !isPerm(k.perm, kTransposed ? bsnhToBnsh : bsnhToBnhs))
      return rewriter.notifyMatchFailure(softmax, "qkv.layout");
    if (v.scale != 1.0)
      return rewriter.notifyMatchFailure(softmax, "value.scaled");
    scale *= q.scale * k.scale;

    auto qType = staticF16Rank4(q.base);
    auto kType = staticF16Rank4(k.base);
    auto vType = staticF16Rank4(v.base);
    if (!qType || !kType || !vType)
      return rewriter.notifyMatchFailure(softmax, "qkv.not_static_f16_4d");
    const int64_t batch = qType.getDimSize(0);
    const int64_t sq = qType.getDimSize(1);
    const int64_t heads = qType.getDimSize(2);
    const int64_t headDim = qType.getDimSize(3);
    const int64_t skv = kType.getDimSize(1);
    if (kType.getShape() != vType.getShape() || kType.getDimSize(0) != batch ||
        kType.getDimSize(2) != heads || kType.getDimSize(3) != headDim)
      return rewriter.notifyMatchFailure(softmax, "qkv.shape_mismatch");
    if (scoresType.getShape() !=
        llvm::ArrayRef<int64_t>({batch, heads, sq, skv}))
      return rewriter.notifyMatchFailure(softmax, "scores.shape_mismatch");
    if (sq <= 1 || headDim > kMaxFlashHeadDim)
      return rewriter.notifyMatchFailure(softmax, "shape.not_flash_eligible");
    if (!std::isfinite(scale) || scale <= 0.0)
      return rewriter.notifyMatchFailure(softmax, "scale.not_positive");

    mlir::Value out = pv->getResult(0);
    auto outType = staticF16Rank4(out);
    if (!outType || outType.getShape() !=
                        llvm::ArrayRef<int64_t>({batch, heads, sq, headDim}))
      return rewriter.notifyMatchFailure(softmax, "output.shape");
    // The common consumer turns BNSH back into BSNH, which is the
    // attention's native output layout; drop that Transpose rather than
    // emit an inverse one.
    mlir::Operation *outTranspose = nullptr;
    if (out.hasOneUse()) {
      mlir::Operation *user = *out.getUsers().begin();
      if (isOnnxOp(user, "onnx.Transpose")) {
        auto perm = getTransposePerm(user, 4);
        if (perm && isPerm(*perm, bsnhToBnsh))
          outTranspose = user;
      }
    }

    mlir::Location loc = softmax->getLoc();
    mlir::MLIRContext *ctx = rewriter.getContext();
    rewriter.setInsertionPoint(pv);
    const int64_t hidden = heads * headDim;
    mlir::Value q3 = reshapeTo(rewriter, loc, q.base, {batch, sq, hidden});
    mlir::Value k3 = reshapeTo(rewriter, loc, k.base, {batch, skv, hidden});
    mlir::Value v3 = reshapeTo(rewriter, loc, v.base, {batch, skv, hidden});

    // Everything between the replacement's inputs and the output, collected
    // before any of it is touched.
    const mlir::Value stops[] = {q3, k3, v3};
    llvm::SmallVector<mlir::Operation *> chain = {pv};
    collectChain(probs, stops, chain);
    collectChain(qk->getOperand(1), stops, chain);
    collectChain(pv->getOperand(1), stops, chain);

    auto mhaType =
        mlir::RankedTensorType::get({batch, sq, hidden}, rewriter.getF16Type());
    mlir::OperationState mhaState(loc, "onnx.Custom");
    mhaState.addOperands({q3, k3, v3});
    mhaState.addTypes(mhaType);
    mhaState.addAttribute("function_name",
                          rewriter.getStringAttr("MultiHeadAttention"));
    mhaState.addAttribute("domain_name",
                          rewriter.getStringAttr("com.microsoft"));
    mhaState.addAttribute("num_heads", si64Attr(ctx, heads));
    mhaState.addAttribute("scale",
                          rewriter.getF32FloatAttr(static_cast<float>(scale)));
    mlir::Value attention = rewriter.create(mhaState)->getResult(0);
    mlir::Value bsnh =
        reshapeTo(rewriter, loc, attention, {batch, sq, heads, headDim});

    if (outTranspose) {
      rewriter.replaceOp(outTranspose, bsnh);
    } else {
      mlir::OperationState trState(loc, "onnx.Transpose");
      trState.addOperands(bsnh);
      trState.addTypes(outType);
      trState.addAttribute("perm", rewriter.getI64ArrayAttr(bsnhToBnsh));
      rewriter.replaceAllUsesWith(out, rewriter.create(trState)->getResult(0));
    }

    // The chain is dead now. Erase it here: unregistered ops are never
    // trivially dead, and leaving it would lower a score matrix nobody reads.
    eraseDead(rewriter, chain);

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] fused B=" << batch
                            << " N=" << heads << " Sq=" << sq << " Skv=" << skv
                            << " H=" << headDim << " scale=" << scale << "\n");
    ++NumDecomposedAttentionFused;
    return mlir::success();
  }
};

} // namespace

void populateDecomposedAttentionFusionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<DecomposedAttentionToMultiHeadAttention>(ctx);
}

} // namespace hip
} // namespace mlir
