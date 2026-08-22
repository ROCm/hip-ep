/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AttentionWindowFold.cpp - Recover a sliding window from the mask ---===//
//
// `onnx.Attention` had no way to declare a sliding window until opset 25, which
// added `left_window_size` / `right_window_size`. At opset 23 and 24 the op's
// attributes are only `is_causal`, `kv_num_heads`, `q_num_heads`,
// `qk_matmul_output_mode`, `scale`, `softcap` and `softmax_precision`, so a
// model whose attention is windowed has nowhere to say so and bakes the window
// into the additive float mask instead: the mask is 0 for the key positions a
// query may attend to and a large negative value everywhere else. The op then
// reaches the runtime with `local_window_size == -1`, so the decomposed path
// scores and copies the whole key range even though everything older than the
// window is already -inf in the scores.
//
// So this rewrite resolves the window in two ways, and the order matters: an
// explicit `left_window_size` is read directly, and only when there is none
// does it recover the window from the mask's producing subgraph. The declared
// attribute is the model telling us; the mask walk is inference about what the
// model happens to have built, which is strictly weaker evidence and is here to
// serve the opset-24 exports that have no alternative. Either way the result is
// stamped as `hipdnn.local_window_size`, which OnnxAttentionConversion reads
// onto `hip.gqa`.
//
// The two spellings do not share a convention, which is the easiest thing to
// get wrong here. Opset 25 defines `left_window_size = L` as L keys *preceding*
// the current one, so the attended count is L + 1 -- the spec's own example has
// `left_window_size=2` admitting "the current key and two preceding keys".
// `hipdnn.local_window_size = N` means N attended keys *including* the current
// one, matching `com.microsoft.GroupQueryAttention` (whose own definition moved
// to include the current token in onnxruntime PR 25927) and the runtime's
// `kv_lo = abs_q - local_window_size + 1`. Hence the `+ 1` on the attribute
// path. A HuggingFace `sliding_window: 1024` is `left_window_size = 1023` is
// `hipdnn.local_window_size = 1024`.
//
// What is matched
// ---------------
// Gemma-4's 25 windowed layers and its 5 global layers share one mask shape and
// differ by exactly one node, which is what makes this recognizable:
//
//   windowed:  keep = And(pad, Or(win, seg))
//   global:    keep = And(pad, Or(GreaterOrEqual(P,Q), seg))
//   both:      mask = Where(keep, 0.0, -65504.0)
//
//   with  win = And(GreaterOrEqual(P,Q), Less(Sub(P,Q), 1024))
//         seg = the same-image-block term described below
//
// The window leg is `And(P >= Q, P - Q < W)` where the compare and the
// subtraction reference the SAME two values in the SAME order. That identity is
// the whole proof, and it needs no knowledge of what P and Q are: the causal
// conjunct `P >= Q` fixes P as the query index and Q as the key index (the
// reverse reading would keep only future keys, which no decoder mask does), so
// `P - Q < W` reads `q - k < W`, i.e. `k >= q - W + 1`. That is exactly the
// runtime's `kv_lo = abs_q - local_window_size + 1`, so W transfers with no
// off-by-one. Where both index tensors are recognizable unsqueezes their axes
// are checked as corroboration.
//
// How the legs combine
// --------------------
// The keep-condition is a monotone And/Or tree, and the bound propagates
// through it the only way it can:
//
//   * AND: a conjunct can only remove keeps, so an unrecognized conjunct is
//     harmless and any bounded conjunct bounds the whole node (min of the
//     bounds). This is what absorbs the padding leg for free.
//   * OR: a disjunct can add keeps, so EVERY disjunct must be bounded or the
//     node is unbounded (max of the bounds).
//
// The segment exception
// ---------------------
// Gemma-4's mask ORs in a same-image-block bidirectional term,
// `And(Equal(a,b), a >= 0)`, built from `Equal(input_ids, <image token>)`
// through a CumSum block index. It is identically false for a text-only prompt,
// so narrowing is then exact. With images it keeps keys in the query's own
// image block, and can therefore reach past the window only if a single image
// block is longer than the window -- 256 soft tokens per image against a
// 1024 window here, a 4x margin. That bound is a property of the input rather
// than of the graph, so it cannot be proven here; this rewrite recognizes the
// term explicitly and accepts it, rather than accepting anything it does not
// understand. A leg that is neither windowed nor this exact shape makes the
// whole match fail and the op keep its full range.
//
// Implementation notes
// --------------------
//   * Roots on `onnx.Attention`. Idempotent: bails if the attribute is already
//     set, so the greedy `ExistingOps` pre-lowering loop quiesces.
//   * Must run BEFORE `convertComputeOps`, whose greedy pattern set would have
//     rewritten the mask's producers into `hip.*` ops in an order this cannot
//     rely on. Pre-lowering still sees generic `onnx.*` with inline constants.
//   * Recognition is conservative throughout: every unhandled shape declines
//     with a breadcrumb and leaves the op scoring its full key range, which is
//     what it does today.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <optional>

#define DEBUG_TYPE "attention-window-fold"

STATISTIC(NumAttentionWindowStamps,
          "Number of onnx.Attention ops whose sliding window was recovered "
          "from the additive mask subgraph");

namespace mlir {
namespace hip {

namespace {

/// Recursion and work caps. The real subgraphs sit at depth 7 with a few dozen
/// nodes; these are loose enough not to bind and tight enough that a
/// pathological graph cannot make the pre-lowering loop quadratic.
constexpr int kMaxDepth = 16;
constexpr int kMaxVisited = 128;

/// The mask value must be negative enough to be a "drop" marker rather than a
/// bias. fp16's most negative finite value (-65504) is the usual choice; -inf
/// and float's -3.4e38 also appear.
constexpr double kDropThreshold = -1.0e4;

static llvm::StringRef opName(mlir::Value v) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  return def ? def->getName().getStringRef() : llvm::StringRef();
}

/// Return `v`'s value when it is an inline integer scalar: `onnx.Constant` with
/// either a rank-0/single-element `value` tensor or the `value_int` attribute
/// (both forms appear in the same graph -- initializers import as the former,
/// exporter-synthesized literals as the latter), or an `arith.constant`.
static std::optional<int64_t> getInlineScalarInt(mlir::Value v) {
  if (!v)
    return std::nullopt;
  mlir::Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (defOp->getName().getStringRef() == "onnx.Constant") {
    if (auto asInt = defOp->getAttrOfType<mlir::IntegerAttr>("value_int"))
      return asInt.getValue().getSExtValue();
  }

  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (defOp->getName().getStringRef() == "onnx.Constant")
    dense = defOp->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!dense)
    return std::nullopt;

  auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
  if (!tensorTy || tensorTy.getRank() > 1 || tensorTy.getNumElements() != 1)
    return std::nullopt;
  if (!tensorTy.getElementType().isIntOrIndex())
    return std::nullopt;
  return (*dense.getValues<mlir::APInt>().begin()).getSExtValue();
}

/// Return `v`'s value when it is an inline floating-point scalar. Used only to
/// identify the Where's keep/drop literals.
static std::optional<double> getInlineScalarFloat(mlir::Value v) {
  if (!v)
    return std::nullopt;
  mlir::Operation *defOp = v.getDefiningOp();
  if (!defOp)
    return std::nullopt;

  if (defOp->getName().getStringRef() == "onnx.Constant") {
    if (auto asFloat = defOp->getAttrOfType<mlir::FloatAttr>("value_float"))
      return asFloat.getValueAsDouble();
  }

  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (defOp->getName().getStringRef() == "onnx.Constant")
    dense = defOp->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!dense)
    return std::nullopt;

  auto tensorTy = mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
  if (!tensorTy || tensorTy.getRank() > 1 || tensorTy.getNumElements() != 1)
    return std::nullopt;
  if (!mlir::isa<mlir::FloatType>(tensorTy.getElementType()))
    return std::nullopt;
  return (*dense.getValues<mlir::APFloat>().begin()).convertToDouble();
}

/// Axis of an `onnx.Unsqueeze`, from the opset-13 `axes` operand or the older
/// `axes` attribute, when it is a single inline value. Used only to corroborate
/// which of the two index tensors varies along the query axis.
static std::optional<int64_t> getUnsqueezeAxis(mlir::Value v) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def || def->getName().getStringRef() != "onnx.Unsqueeze")
    return std::nullopt;
  if (def->getNumOperands() > 1)
    return getInlineScalarInt(def->getOperand(1));
  if (auto arr = def->getAttrOfType<mlir::ArrayAttr>("axes")) {
    if (arr.size() != 1)
      return std::nullopt;
    if (auto intAttr = mlir::dyn_cast<mlir::IntegerAttr>(arr[0]))
      return intAttr.getValue().getSExtValue();
  }
  return std::nullopt;
}

/// Match a causal keep-compare, `p >= q`, in either spelling.
static bool matchCausalCompare(mlir::Value v, mlir::Value &p, mlir::Value &q) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def || def->getNumOperands() != 2)
    return false;
  llvm::StringRef name = def->getName().getStringRef();
  if (name == "onnx.GreaterOrEqual") {
    p = def->getOperand(0);
    q = def->getOperand(1);
    return true;
  }
  if (name == "onnx.LessOrEqual") { // q <= p
    p = def->getOperand(1);
    q = def->getOperand(0);
    return true;
  }
  return false;
}

/// Match an upper bound on `p - q`, returning the window it implies: the number
/// of key positions the bound admits, counting the query's own position.
/// `p - q < W` admits W of them; `p - q <= W` admits W + 1.
static bool matchDistanceBound(mlir::Value v, mlir::Value &p, mlir::Value &q,
                               int64_t &window) {
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def || def->getNumOperands() != 2)
    return false;
  llvm::StringRef name = def->getName().getStringRef();

  // Which operand holds the difference and which the limit, and whether the
  // comparison is strict.
  mlir::Value diffSide, limitSide;
  bool strict;
  if (name == "onnx.Less") { // diff < W
    diffSide = def->getOperand(0);
    limitSide = def->getOperand(1);
    strict = true;
  } else if (name == "onnx.LessOrEqual") { // diff <= W
    diffSide = def->getOperand(0);
    limitSide = def->getOperand(1);
    strict = false;
  } else if (name == "onnx.Greater") { // W > diff
    diffSide = def->getOperand(1);
    limitSide = def->getOperand(0);
    strict = true;
  } else if (name == "onnx.GreaterOrEqual") { // W >= diff
    diffSide = def->getOperand(1);
    limitSide = def->getOperand(0);
    strict = false;
  } else {
    return false;
  }

  if (opName(diffSide) != "onnx.Sub")
    return false;
  mlir::Operation *sub = diffSide.getDefiningOp();
  if (sub->getNumOperands() != 2)
    return false;

  auto limit = getInlineScalarInt(limitSide);
  if (!limit)
    return false;

  p = sub->getOperand(0);
  q = sub->getOperand(1);
  window = strict ? *limit : *limit + 1;
  return true;
}

/// Match the same-segment bidirectional term, `And(Equal(a,b), a >= c)` with a
/// constant `c`. See "The segment exception" above for why this is accepted
/// rather than proven.
static bool matchSegmentTerm(mlir::Operation *andOp) {
  if (andOp->getNumOperands() != 2)
    return false;
  for (int i = 0; i < 2; ++i) {
    mlir::Value eq = andOp->getOperand(i);
    mlir::Value ge = andOp->getOperand(1 - i);
    if (opName(eq) != "onnx.Equal")
      continue;
    mlir::Operation *eqOp = eq.getDefiningOp();
    mlir::Operation *geOp = ge.getDefiningOp();
    if (!geOp || eqOp->getNumOperands() != 2 || geOp->getNumOperands() != 2)
      continue;
    llvm::StringRef geName = geOp->getName().getStringRef();
    if (geName != "onnx.GreaterOrEqual" && geName != "onnx.Greater")
      continue;
    if (!getInlineScalarInt(geOp->getOperand(1)))
      continue;
    // The "is in a segment at all" test must be on one of the two values the
    // equality compares, or these are unrelated conditions.
    mlir::Value probe = geOp->getOperand(0);
    if (probe == eqOp->getOperand(0) || probe == eqOp->getOperand(1))
      return true;
  }
  return false;
}

/// What a boolean leg of the keep-condition tells us about how far back a kept
/// key can be.
enum class LegKind {
  /// No information. Safe as an AND conjunct (it can only remove keeps), fatal
  /// as an OR disjunct.
  Unbounded,
  /// Keeping implies the key is within `window` positions of the query.
  Windowed,
  /// The recognized same-segment term, accepted under the documented
  /// image-block bound.
  Segment,
};

struct LegInfo {
  LegKind kind = LegKind::Unbounded;
  int64_t window = 0;

  static LegInfo unbounded() { return {}; }
  static LegInfo windowed(int64_t w) { return {LegKind::Windowed, w}; }
  static LegInfo segment() { return {LegKind::Segment, 0}; }
};

/// Classify the keep-condition rooted at `v`. `visited` bounds total work.
static LegInfo classifyKeep(mlir::Value v, int depth, int &visited) {
  if (depth > kMaxDepth || ++visited > kMaxVisited)
    return LegInfo::unbounded();
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def)
    return LegInfo::unbounded();
  llvm::StringRef name = def->getName().getStringRef();

  // Value-preserving wrappers: a bool-to-bool cast keeps the predicate, and an
  // int-to-bool cast is a nonzero test that we would classify as unbounded
  // either way, so descending is always safe.
  if (name == "onnx.Cast" || name == "onnx.CastLike" || name == "onnx.Identity")
    return classifyKeep(def->getOperand(0), depth + 1, visited);

  if (name == "onnx.And") {
    if (def->getNumOperands() != 2)
      return LegInfo::unbounded();
    // Try the two leaf shapes first: both are And-rooted, and recursing into
    // their halves would only find unbounded compares.
    if (matchSegmentTerm(def))
      return LegInfo::segment();

    for (int i = 0; i < 2; ++i) {
      mlir::Value causal = def->getOperand(i);
      mlir::Value bound = def->getOperand(1 - i);
      mlir::Value cp, cq, bp, bq;
      int64_t window = 0;
      if (!matchCausalCompare(causal, cp, cq))
        continue;
      if (!matchDistanceBound(bound, bp, bq, window))
        continue;
      // The proof: the causal compare and the subtraction must reference the
      // same two values in the same order, which is what makes `p` the query
      // index and `q` the key index without having to identify either.
      if (cp != bp || cq != bq || window <= 0)
        continue;
      // Corroboration where the shapes allow it. Broadcast into [.., q, k]
      // puts the query-varying tensor's data on the lower axis, so it is the
      // one unsqueezed at the HIGHER axis. Only a clear inversion is rejected;
      // an unreadable axis leaves the causal argument standing on its own.
      auto pAxis = getUnsqueezeAxis(cp);
      auto qAxis = getUnsqueezeAxis(cq);
      if (pAxis && qAxis && *pAxis < *qAxis)
        continue;
      return LegInfo::windowed(window);
    }

    // A general conjunction: any bounded conjunct bounds the whole node, and
    // the tightest one wins.
    LegInfo a = classifyKeep(def->getOperand(0), depth + 1, visited);
    LegInfo b = classifyKeep(def->getOperand(1), depth + 1, visited);
    if (a.kind == LegKind::Windowed && b.kind == LegKind::Windowed)
      return LegInfo::windowed(std::min(a.window, b.window));
    if (a.kind == LegKind::Windowed)
      return a;
    if (b.kind == LegKind::Windowed)
      return b;
    if (a.kind == LegKind::Segment || b.kind == LegKind::Segment)
      return LegInfo::segment();
    return LegInfo::unbounded();
  }

  if (name == "onnx.Or") {
    if (def->getNumOperands() != 2)
      return LegInfo::unbounded();
    LegInfo a = classifyKeep(def->getOperand(0), depth + 1, visited);
    LegInfo b = classifyKeep(def->getOperand(1), depth + 1, visited);
    // A disjunct adds keeps, so an unrecognized one leaves the whole node
    // unbounded no matter how tight the other side is.
    if (a.kind == LegKind::Unbounded || b.kind == LegKind::Unbounded)
      return LegInfo::unbounded();
    if (a.kind == LegKind::Windowed && b.kind == LegKind::Windowed)
      return LegInfo::windowed(std::max(a.window, b.window));
    if (a.kind == LegKind::Windowed)
      return a;
    if (b.kind == LegKind::Windowed)
      return b;
    return LegInfo::segment();
  }

  return LegInfo::unbounded();
}

/// Walk from the mask value down to the `onnx.Where` that turns the boolean
/// keep-condition into the additive 0 / large-negative bias, through the layout
/// ops that do not change values.
static mlir::Operation *findSelect(mlir::Value mask) {
  mlir::Value cur = mask;
  for (int depth = 0; depth < kMaxDepth; ++depth) {
    mlir::Operation *def = cur ? cur.getDefiningOp() : nullptr;
    if (!def)
      return nullptr;
    llvm::StringRef name = def->getName().getStringRef();
    if (name == "onnx.Where")
      return def;
    if (name == "onnx.Cast" || name == "onnx.CastLike" ||
        name == "onnx.Identity" || name == "onnx.Unsqueeze" ||
        name == "onnx.Squeeze" || name == "onnx.Reshape" ||
        name == "onnx.Expand") {
      cur = def->getOperand(0);
      continue;
    }
    return nullptr;
  }
  return nullptr;
}

struct AttentionStampMaskWindow : public mlir::RewritePattern {
  AttentionStampMaskWindow(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Attention", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->hasAttr("hipdnn.local_window_size"))
      return rewriter.notifyMatchFailure(op, "window.already_stamped");

    // Opset 25's declared window wins over anything inferred from the mask.
    // Only a non-negative left bound is a window: the -1 default means "left
    // side unbounded", which says nothing about the mask and so must fall
    // through to the walk rather than suppress it.
    if (auto left = op->getAttrOfType<mlir::IntegerAttr>("left_window_size")) {
      int64_t l = left.getValue().getSExtValue();
      if (l >= 0) {
        // A right bound that admits keys past the current position is a
        // bidirectional window, which `hipdnn.local_window_size` cannot express
        // -- it is a causal left extent only. Decline rather than narrow by
        // half a specification. Absent, -1 and 0 are all fine: 0 is "current
        // key only" on the right, and with no bound at all there is nothing to
        // the right of a decode step's query anyway.
        if (auto right =
                op->getAttrOfType<mlir::IntegerAttr>("right_window_size")) {
          int64_t r = right.getValue().getSExtValue();
          if (r > 0)
            return rewriter.notifyMatchFailure(op, "window.bidirectional");
        }
        // +1: opset 25 counts preceding keys, hipdnn counts attended keys.
        rewriter.modifyOpInPlace(op, [&] {
          op->setAttr("hipdnn.local_window_size",
                      rewriter.getI64IntegerAttr(l + 1));
        });
        return mlir::success();
      }
    }

    // Operand 3 is attn_mask. Absent or None means there is no mask to read a
    // window out of.
    if (op->getNumOperands() < 4)
      return rewriter.notifyMatchFailure(op, "window.no_mask_operand");
    mlir::Value mask = op->getOperand(3);
    if (!mask || mlir::isa<mlir::NoneType>(mask.getType()))
      return rewriter.notifyMatchFailure(op, "window.mask_is_none");

    mlir::Operation *select = findSelect(mask);
    if (!select || select->getNumOperands() != 3)
      return rewriter.notifyMatchFailure(op, "window.no_select_producer");

    // Polarity: the true arm must be the neutral bias and the false arm the
    // drop marker. The inverted spelling would need the boolean function
    // negated, which turns every And into an Or and is not handled.
    auto keepVal = getInlineScalarFloat(select->getOperand(1));
    auto dropVal = getInlineScalarFloat(select->getOperand(2));
    if (!keepVal || !dropVal)
      return rewriter.notifyMatchFailure(op, "window.select_arms_not_const");
    if (*keepVal != 0.0 || *dropVal > kDropThreshold)
      return rewriter.notifyMatchFailure(op, "window.select_arms_not_0_neg");

    int visited = 0;
    LegInfo info = classifyKeep(select->getOperand(0), /*depth=*/0, visited);
    if (info.kind != LegKind::Windowed)
      return rewriter.notifyMatchFailure(op, "window.keep_cond_unbounded");

    rewriter.modifyOpInPlace(op, [&] {
      op->setAttr("hipdnn.local_window_size",
                  rewriter.getI64IntegerAttr(info.window));
    });

    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] recovered local_window_size="
                            << info.window << " from the mask subgraph ("
                            << visited << " nodes visited)\n");
    ++NumAttentionWindowStamps;
    return mlir::success();
  }
};

} // namespace

void populateAttentionWindowFoldPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx) {
  patterns.add<AttentionStampMaskWindow>(ctx);
}

} // namespace hip
} // namespace mlir
