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
// So this rewrite recovers the window from the mask's producing subgraph and
// stamps it as `hipdnn.local_window_size`, which OnnxAttentionConversion reads
// onto `hip.gqa`. This is inference about what the model happens to have built,
// not the model telling us, and it is here to serve the opset-24 exports that
// have no way to tell us. Reading a declared `left_window_size` where one
// exists is a separate and strictly easier change, deliberately not made here:
// no export on hand has the attribute, so it could not be tested against
// anything real.
//
// `hipdnn.local_window_size = N` means N attended keys *including* the current
// one, matching `com.microsoft.GroupQueryAttention` (whose own definition moved
// to include the current token in onnxruntime PR 25927) and the runtime's
// `kv_lo = abs_q - local_window_size + 1`. A HuggingFace `sliding_window: 1024`
// is `hipdnn.local_window_size = 1024`, and the mask's `P - Q < 1024` below
// carries that same count, so W transfers with no adjustment.
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
// subtraction reference the SAME two values in the SAME order. That identity
// fixes the two roles relative to each other: the causal conjunct `P >= Q`
// makes P the query side and Q the key side, since the reverse reading would
// keep only future keys, which no decoder mask does. So `P - Q < W` reads
// `q - k < W`, i.e. `k >= q - W + 1`, which is exactly the runtime's
// `kv_lo = abs_q - local_window_size + 1`.
//
// Identity alone is not enough, though, and this is the part worth being
// careful about. It says `P - Q < W` bounds the difference of two tensors; the
// runtime then narrows KV cache reads by ABSOLUTE POSITION. For the second to
// follow from the first, P and Q have to be positions on one common scale --
// same origin, unit stride. Structurally identical arithmetic over tensors that
// were scaled or re-based would match just as well and yield a W in the wrong
// units, and the narrowing would then drop keys the model wanted.
//
// So P and Q must additionally be drawn from the SAME position sequence: both
// must reach a common `onnx.CumSum` or `onnx.Range` within
// `kPositionProvenanceDepth`. That is the standard way a decoder builds
// positions -- a cumulative count over the attention mask, or a plain range --
// and sharing the node is what makes the subtraction a genuine distance rather
// than a difference of two unrelated quantities. On the Gemma-4 export the key
// side is `Unsqueeze(CumSum(attention_mask))` and the query side is
// `Unsqueeze(Slice(CumSum(attention_mask)))`, the same CumSum reached at depths
// 1 and 2, so the bound is small. Where both index tensors are recognizable
// unsqueezes their axes are checked as further corroboration.
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
// `And(Equal(a,b), a >= 0)`. It is identically false for a text-only prompt, so
// narrowing is then exact. With images it keeps keys in the query's own image
// block, and can therefore reach past the window only if a single image block
// is longer than the window -- 256 soft tokens per image against a 1024 window
// here, a 4x margin. That bound is a property of the input rather than of the
// graph, so it cannot be proven here; this rewrite recognizes the term
// explicitly and accepts it, rather than accepting anything it does not
// understand. A leg that is neither windowed nor this exact shape makes the
// whole match fail and the op keep its full range.
//
// `And(Equal(a,b), Cmp(a, const))` on its own is too weak to accept, because it
// is also how a same-document mask or a prefix-LM mask is spelled, and those
// reach arbitrarily far back. What separates the block-index reading is where
// the compared values come from: a block index is a running count, so an
// `onnx.CumSum` is an ancestor of the equality's operands. Requiring that
// ancestor is what makes the leg identifiable rather than merely plausible --
// a document id read straight out of an input tensor has no CumSum behind it
// and is declined. On the Gemma-4 export the CumSum sits 3 and 4 nodes above
// the two `Equal` operands respectively, so `kSegmentProvenanceDepth` only has
// to be a small constant.
//
// Both legs therefore ask for a CumSum, for different reasons and -- worth
// knowing when reading a dump -- not the same one. The export has exactly two:
// a block-index CumSum over the image-block starts, which is the one the
// segment leg reaches, and a position CumSum over `attention_mask`, which is
// the one the window leg's P and Q share. Finding the position CumSum behind a
// segment leg would say nothing about block extent, which is why the segment
// check looks above the `Equal` operands specifically rather than anywhere in
// the mask region.
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
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <optional>

// DEBUG_TYPE comes from OnnxToHipUtils.h ("convert-onnx-to-hip"), shared with
// the rest of this directory rather than defined per file here, so that this
// change does not also have to move the header's definition.

STATISTIC(NumAttentionWindowStamps,
          "Number of onnx.Attention ops stamped with a sliding window "
          "recovered from the additive mask subgraph");
STATISTIC(NumAttentionKeyBiasFolds,
          "Number of onnx.Attention masks reduced to a per-key bias row");

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

/// How far above an `onnx.Equal` operand an `onnx.CumSum` may sit and still
/// count as its provenance. The Gemma-4 export needs 4; the slack absorbs the
/// layout ops an exporter may add without admitting an unrelated CumSum from
/// elsewhere in the graph.
constexpr int kSegmentProvenanceDepth = 6;

/// How far above the window leg's `P` / `Q` a shared position source may sit.
/// The Gemma-4 export needs 2 on the query side (through an `onnx.Slice` that
/// takes the trailing `sq` entries) and 1 on the key side.
constexpr int kPositionProvenanceDepth = 6;

/// Collect the position sources reachable above `v`.
///
/// A position source is an `onnx.CumSum` (the cumulative count over an
/// attention mask that a decoder builds `position_ids` from) or an
/// `onnx.Range`. Both are treated as leaves: the walk stops there rather than
/// descending into what fed them, because it is the identity of the node that
/// matters, not its inputs.
static void
collectPositionSources(mlir::Value v, int depth, int &visited,
                       llvm::SmallPtrSetImpl<mlir::Operation *> &out) {
  if (!v || depth > kPositionProvenanceDepth || ++visited > kMaxVisited)
    return;
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return;
  llvm::StringRef name = def->getName().getStringRef();
  if (name == "onnx.CumSum" || name == "onnx.Range") {
    out.insert(def);
    return;
  }
  for (mlir::Value operand : def->getOperands())
    collectPositionSources(operand, depth + 1, visited, out);
}

/// Are `p` and `q` drawn from the same position sequence?
///
/// See "The window leg" in the file header. The same-value identity between the
/// causal compare and the subtraction fixes which side is the query and which
/// the key, but it says nothing about the units of `p - q`, and the runtime
/// narrows the KV cache by absolute position. Requiring both to reach one
/// common position source is what makes the difference a position distance: two
/// tensors sliced from a single monotone position sequence share an origin and
/// a stride, so subtracting them measures positions.
///
/// A common ancestor of any op type would be far too weak -- two unrelated
/// index tensors routinely share a `Shape` -- so the shared node must itself be
/// a recognized position source.
static bool hasSharedPositionSource(mlir::Value p, mlir::Value q,
                                    int &visited) {
  llvm::SmallPtrSet<mlir::Operation *, 4> pSources;
  collectPositionSources(p, /*depth=*/0, visited, pSources);
  if (pSources.empty())
    return false;
  llvm::SmallPtrSet<mlir::Operation *, 4> qSources;
  collectPositionSources(q, /*depth=*/0, visited, qSources);
  for (mlir::Operation *src : pSources)
    if (qSources.contains(src))
      return true;
  return false;
}

/// Is an `onnx.CumSum` an ancestor of `v` within `kSegmentProvenanceDepth`?
///
/// This is the provenance check described under "The segment exception": it is
/// what distinguishes a block index (a running count, hence a CumSum) from a
/// document id or prefix-LM boundary read straight out of an input, which wear
/// the same `Equal` spelling but bound nothing. `visited` is the caller's
/// budget, so a wide graph cannot turn this into a quadratic walk.
static bool hasCumSumAncestor(mlir::Value v, int depth, int &visited) {
  if (depth > kSegmentProvenanceDepth || ++visited > kMaxVisited)
    return false;
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  if (!def)
    return false;
  if (def->getName().getStringRef() == "onnx.CumSum")
    return true;
  for (mlir::Value operand : def->getOperands())
    if (hasCumSumAncestor(operand, depth + 1, visited))
      return true;
  return false;
}

/// Match the same-segment bidirectional term, `And(Equal(a,b), a >= c)` with a
/// constant `c` and a CumSum behind the equality. See "The segment exception"
/// above for why the shape is accepted rather than proven, and why the CumSum
/// is required rather than merely documented.
static bool matchSegmentTerm(mlir::Operation *andOp, int &visited) {
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
    if (probe != eqOp->getOperand(0) && probe != eqOp->getOperand(1))
      continue;
    // Provenance: one side of the equality must be a running count. Either
    // side satisfies it -- the export derives both from the same CumSum, at
    // different depths, and which one is shallower is an exporter detail.
    if (hasCumSumAncestor(eqOp->getOperand(0), /*depth=*/0, visited) ||
        hasCumSumAncestor(eqOp->getOperand(1), /*depth=*/0, visited))
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
    if (matchSegmentTerm(def, visited))
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
      // Half the proof: the causal compare and the subtraction must reference
      // the same two values in the same order, which is what makes `p` the
      // query index and `q` the key index without having to identify either.
      if (cp != bp || cq != bq || window <= 0)
        continue;
      // The other half: they must be positions on a common scale, or `p - q`
      // is not a position distance and W is not in units of key positions.
      if (!hasSharedPositionSource(cp, cq, visited))
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

//===----------------------------------------------------------------------===//
// Key-padding fold
//===----------------------------------------------------------------------===//
//
// A causal decoder export with `is_causal = 1` very often still ships the
// whole keep-condition in its additive mask:
//
//   keep = And(pad, causal [, window])
//   mask = Unsqueeze(Cast_f16(Where(keep, 0.0, -65504.0)), [1])
//
//   with  pad    = Cast_bool(Unsqueeze(attention_mask, [1]))    [B, 1, T]
//         causal = GreaterOrEqual(P, Q)                         [B, S, T]
//         window = Less(Sub(P, Q), W)
//
// The causal conjunct repeats what `is_causal = 1` already applies, and the
// window conjunct what `hipdnn.local_window_size` carries. Only `pad` adds
// information, and it varies along the key axis alone. Rebuilding the mask
// from that conjunct gives the same scores with a [B, 1, 1, T] mask instead
// of [B, 1, S, T]:
//
//   mask' = Unsqueeze(Cast_f16(Where(pad, 0.0, -65504.0)), [1])  [B, 1, 1, T]
//
// and the op is stamped `hipdnn.key_bias_mask`, which tells
// OnnxAttentionConversion to pass the row as hip.gqa's per-key `key_bias`
// instead of a full `attention_bias`. That is what lets the fused flash
// kernels serve the op: they add one bias value per key and keep their own
// causal and window masking, where a full [S, T] bias would have to be
// streamed per query row.
//
// Why dropping the causal and window conjuncts is exact. Both compare P and Q
// drawn from one position sequence (the same provenance test the window fold
// uses), while the runtime masks by absolute index. The two agree for every
// query that is itself a real token and every key `pad` keeps, as long as the
// attention mask's zeros are contiguous at one end -- left or right padding,
// which is what generators emit: such a key at absolute index j and query at
// absolute index i have positions j - L and i - L for the same pad count L,
// so P >= Q is i >= j and P - Q < W is i - j < W. A key `pad` drops is
// dropped by mask' too, whatever its position. The rows of padded queries may
// change (a right-padded query's position is clamped, a left-padded one's row
// is fully masked either way), but `pad` drops those positions as keys from
// every row, so no real token's output depends on them. This is the same
// assumption the window stamp already makes when it narrows the KV range by
// absolute position.
//
// Recognition is all-or-nothing, unlike the window fold: every conjunct must
// be understood, because every conjunct but `pad` is dropped. An Or anywhere,
// a second per-key leg, or a leg that varies along the query axis without
// being a recognized causal or window compare declines and leaves the mask
// exactly as it was.

/// Is `v` a boolean conjunct that varies only along the batch and key axes of
/// a rank-`rank` keep-condition? Every axis between the first and the last
/// must be a static 1. That is a statement about the value's type, which is
/// what makes it safe: whatever produced it, broadcasting it into [B, .., T]
/// cannot depend on the query.
/// The ORT importer spells ONNX BOOL as ui8, so an And operand of either type
/// is a boolean.
static bool isPerKeyLeg(mlir::Value v, int64_t rank) {
  auto t = mlir::dyn_cast<mlir::RankedTensorType>(v.getType());
  if (!t || t.getRank() != rank || rank < 2)
    return false;
  mlir::Type elem = t.getElementType();
  if (!elem.isInteger(1) && !elem.isUnsignedInteger(8))
    return false;
  for (int64_t i : llvm::seq<int64_t>(1, rank - 1))
    if (t.getDimSize(i) != 1)
      return false;
  return true;
}

/// Split an And-tree into its conjuncts. Casts from i1 are looked through; a
/// cast from any other type, ui8 included since it may be real uint8 data, is
/// a nonzero test and stays a leaf.
static bool collectConjuncts(mlir::Value v, int depth,
                             llvm::SmallVectorImpl<mlir::Value> &out) {
  if (depth > kMaxDepth || out.size() > 8)
    return false;
  mlir::Operation *def = v ? v.getDefiningOp() : nullptr;
  llvm::StringRef name = def ? def->getName().getStringRef() : "";
  if (name == "onnx.And" && def->getNumOperands() == 2)
    return collectConjuncts(def->getOperand(0), depth + 1, out) &&
           collectConjuncts(def->getOperand(1), depth + 1, out);
  if ((name == "onnx.Cast" || name == "onnx.Identity") &&
      def->getNumOperands() == 1) {
    auto in = mlir::dyn_cast<mlir::ShapedType>(def->getOperand(0).getType());
    if (in && in.getElementType().isInteger(1))
      return collectConjuncts(def->getOperand(0), depth + 1, out);
  }
  if (!llvm::is_contained(out, v))
    out.push_back(v);
  return true;
}

/// Inline axes of an `onnx.Unsqueeze`, from the operand or the attribute.
static std::optional<llvm::SmallVector<int64_t>>
getUnsqueezeAxes(mlir::Operation *def) {
  llvm::SmallVector<int64_t> axes;
  if (def->getNumOperands() > 1) {
    mlir::Operation *c = def->getOperand(1).getDefiningOp();
    mlir::DenseElementsAttr dense;
    if (auto cst = mlir::dyn_cast_or_null<mlir::arith::ConstantOp>(c))
      dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
    else if (c && c->getName().getStringRef() == "onnx.Constant")
      dense = c->getAttrOfType<mlir::DenseElementsAttr>("value");
    if (!dense || !dense.getElementType().isInteger())
      return std::nullopt;
    for (const llvm::APInt &a : dense.getValues<llvm::APInt>())
      axes.push_back(a.getSExtValue());
    return axes;
  }
  auto arr = def->getAttrOfType<mlir::ArrayAttr>("axes");
  if (!arr)
    return std::nullopt;
  for (mlir::Attribute a : arr) {
    auto i = mlir::dyn_cast<mlir::IntegerAttr>(a);
    if (!i)
      return std::nullopt;
    axes.push_back(i.getValue().getSExtValue());
  }
  return axes;
}

/// The result type of `def` (a value-preserving op on the mask's path) when
/// its data operand is narrowed to `in`. Only the ops whose result shape
/// follows from the operand's are handled; anything else declines.
static mlir::RankedTensorType retypeOnPath(mlir::Operation *def,
                                           mlir::RankedTensorType in) {
  auto out =
      mlir::dyn_cast<mlir::RankedTensorType>(def->getResult(0).getType());
  if (!out)
    return {};
  llvm::StringRef name = def->getName().getStringRef();
  if (name == "onnx.Cast" || name == "onnx.Identity")
    return mlir::RankedTensorType::get(in.getShape(), out.getElementType());
  if (name != "onnx.Unsqueeze")
    return {};
  auto axes = getUnsqueezeAxes(def);
  if (!axes)
    return {};
  const int64_t rank = in.getRank() + static_cast<int64_t>(axes->size());
  if (out.getRank() != rank)
    return {};
  llvm::SmallVector<bool> inserted(rank, false);
  for (int64_t a : *axes) {
    if (a < 0)
      a += rank;
    // A trailing axis would move the key extent off the last dimension.
    if (a < 0 || a >= rank - 1 || inserted[a])
      return {};
    inserted[a] = true;
  }
  llvm::SmallVector<int64_t> shape;
  int64_t src = 0;
  for (int64_t i : llvm::seq<int64_t>(rank))
    shape.push_back(inserted[i] ? 1 : in.getDimSize(src++));
  return mlir::RankedTensorType::get(shape, out.getElementType());
}

/// Clone `def` with its data operand replaced by `in` and its result retyped
/// to `ty`, reusing an identical clone another Attention already made: the
/// layers of a model share one mask, and each would otherwise rebuild it.
/// The clone goes right before `def`, which every operand already dominates
/// (`in` is an ancestor of `def`'s data operand or was placed before it). A
/// reused op must precede `def` for the same reason.
static mlir::Value cloneOnto(mlir::PatternRewriter &rewriter,
                             mlir::Operation *def, mlir::Value in,
                             mlir::RankedTensorType ty) {
  for (mlir::Operation *user : in.getUsers()) {
    if (user == def || user->getName() != def->getName() ||
        user->getNumOperands() != def->getNumOperands() ||
        user->getOperand(0) != in || user->getResult(0).getType() != ty ||
        user->getAttrDictionary() != def->getAttrDictionary() ||
        user->getBlock() != def->getBlock() || !user->isBeforeInBlock(def))
      continue;
    bool same = true;
    for (unsigned i : llvm::seq<unsigned>(1, def->getNumOperands()))
      same = same && user->getOperand(i) == def->getOperand(i);
    if (same)
      return user->getResult(0);
  }
  mlir::OpBuilder::InsertionGuard guard(rewriter);
  rewriter.setInsertionPoint(def);
  mlir::Operation *clone = rewriter.clone(*def);
  clone->setOperand(0, in);
  clone->getResult(0).setType(ty);
  return clone->getResult(0);
}

struct AttentionFoldKeyPaddingMask : public mlir::RewritePattern {
  AttentionFoldKeyPaddingMask(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Attention", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->hasAttr("hipdnn.key_bias_mask"))
      return rewriter.notifyMatchFailure(op, "keybias.already_folded");
    auto causalAttr = op->getAttrOfType<mlir::IntegerAttr>("is_causal");
    if (!causalAttr || causalAttr.getValue().getSExtValue() != 1)
      return rewriter.notifyMatchFailure(op, "keybias.not_is_causal");
    if (op->getNumOperands() < 4)
      return rewriter.notifyMatchFailure(op, "keybias.no_mask_operand");
    mlir::Value mask = op->getOperand(3);
    if (!mask || mlir::isa<mlir::NoneType>(mask.getType()))
      return rewriter.notifyMatchFailure(op, "keybias.mask_is_none");

    // The path from the mask down to its Where, outermost first. Only ops
    // whose result shape follows from their operand's can be replayed onto
    // the narrowed keep-condition.
    llvm::SmallVector<mlir::Operation *> path;
    mlir::Operation *select = nullptr;
    for (mlir::Value cur = mask; !select && path.size() < 4;) {
      mlir::Operation *def = cur.getDefiningOp();
      if (!def)
        break;
      llvm::StringRef name = def->getName().getStringRef();
      if (name == "onnx.Where") {
        select = def;
      } else if (name == "onnx.Cast" || name == "onnx.Identity" ||
                 name == "onnx.Unsqueeze") {
        path.push_back(def);
        cur = def->getOperand(0);
      } else {
        break;
      }
    }
    if (!select || select->getNumOperands() != 3)
      return rewriter.notifyMatchFailure(op, "keybias.no_select_producer");
    auto keepVal = getInlineScalarFloat(select->getOperand(1));
    auto dropVal = getInlineScalarFloat(select->getOperand(2));
    if (!keepVal || !dropVal || *keepVal != 0.0 || *dropVal > kDropThreshold)
      return rewriter.notifyMatchFailure(op, "keybias.select_arms_not_0_neg");
    for (int i : {1, 2}) {
      auto arm = mlir::dyn_cast<mlir::RankedTensorType>(
          select->getOperand(i).getType());
      if (!arm || arm.getNumElements() != 1)
        return rewriter.notifyMatchFailure(op, "keybias.select_arm_not_scalar");
    }

    mlir::Value keep = select->getOperand(0);
    auto keepTy = mlir::dyn_cast<mlir::RankedTensorType>(keep.getType());
    if (!keepTy || keepTy.getRank() < 2)
      return rewriter.notifyMatchFailure(op, "keybias.keep_unranked");
    llvm::SmallVector<mlir::Value> conjuncts;
    if (!collectConjuncts(keep, /*depth=*/0, conjuncts))
      return rewriter.notifyMatchFailure(op, "keybias.keep_too_deep");

    // Classify every conjunct. Causal compares are kept by (P, Q) so a window
    // bound can be checked against one of them.
    mlir::Value keyLeg;
    llvm::SmallVector<std::pair<mlir::Value, mlir::Value>> causal;
    llvm::SmallVector<std::tuple<mlir::Value, mlir::Value, int64_t>> bounds;
    for (mlir::Value c : conjuncts) {
      mlir::Value p, q;
      int64_t w = 0;
      int visited = 0;
      // The bound first: `W >= Sub(P, Q)` is also a GreaterOrEqual.
      if (matchDistanceBound(c, p, q, w)) {
        bounds.emplace_back(p, q, w);
      } else if (matchCausalCompare(c, p, q)) {
        auto pAxis = getUnsqueezeAxis(p);
        auto qAxis = getUnsqueezeAxis(q);
        if (!hasSharedPositionSource(p, q, visited) ||
            (pAxis && qAxis && *pAxis < *qAxis))
          return rewriter.notifyMatchFailure(op, "keybias.causal_unproven");
        causal.emplace_back(p, q);
      } else if (isPerKeyLeg(c, keepTy.getRank()) && !keyLeg) {
        keyLeg = c;
      } else {
        return rewriter.notifyMatchFailure(op, "keybias.unrecognized_conjunct");
      }
    }
    if (!keyLeg)
      return rewriter.notifyMatchFailure(op, "keybias.no_key_leg");
    std::optional<int64_t> window;
    for (auto [p, q, w] : bounds) {
      // The causal compare over the same (P, Q) is what orients the bound;
      // without one, `P - Q < W` could equally bound the future.
      if (w <= 0 || !llvm::is_contained(causal, std::make_pair(p, q)))
        return rewriter.notifyMatchFailure(op, "keybias.window_unproven");
      window = window ? std::min(*window, w) : w;
    }
    if (auto stamped =
            op->getAttrOfType<mlir::IntegerAttr>("hipdnn.local_window_size")) {
      if (stamped.getValue().getSExtValue() != window.value_or(-1))
        return rewriter.notifyMatchFailure(op, "keybias.window_disagrees");
    }

    // Replay the Where and the layout path onto the key leg, innermost first.
    auto keyTy = mlir::cast<mlir::RankedTensorType>(keyLeg.getType());
    const int64_t last = keyTy.getRank() - 1;
    if (keyTy.getDimSize(last) == 1 && keepTy.getDimSize(last) != 1)
      return rewriter.notifyMatchFailure(op, "keybias.key_leg_broadcast");
    auto selTy =
        mlir::dyn_cast<mlir::RankedTensorType>(select->getResult(0).getType());
    if (!selTy || selTy.getRank() != keyTy.getRank())
      return rewriter.notifyMatchFailure(op, "keybias.select_rank");
    llvm::SmallVector<mlir::RankedTensorType> types;
    mlir::RankedTensorType curTy =
        mlir::RankedTensorType::get(keyTy.getShape(), selTy.getElementType());
    types.push_back(curTy);
    for (mlir::Operation *def : llvm::reverse(path)) {
      curTy = retypeOnPath(def, curTy);
      if (!curTy)
        return rewriter.notifyMatchFailure(op, "keybias.path_not_replayable");
      types.push_back(curTy);
    }
    auto maskTy = mlir::dyn_cast<mlir::RankedTensorType>(mask.getType());
    if (!maskTy || curTy.getRank() != 4 || maskTy.getRank() != 4 ||
        curTy.getDimSize(1) != 1 || curTy.getDimSize(2) != 1 ||
        curTy.getElementType() != maskTy.getElementType())
      return rewriter.notifyMatchFailure(op, "keybias.result_not_b11t");
    // hip.gqa's key_bias is an fp16 row.
    if (!maskTy.getElementType().isF16())
      return rewriter.notifyMatchFailure(op, "keybias.mask_not_f16");

    mlir::Value v = cloneOnto(rewriter, select, keyLeg, types.front());
    for (auto [def, ty] :
         llvm::zip(llvm::reverse(path), llvm::drop_begin(types)))
      v = cloneOnto(rewriter, def, v, ty);

    rewriter.modifyOpInPlace(op, [&] {
      op->setOperand(3, v);
      op->setAttr("hipdnn.key_bias_mask", rewriter.getUnitAttr());
      if (window)
        op->setAttr("hipdnn.local_window_size",
                    rewriter.getI64IntegerAttr(*window));
    });
    LLVM_DEBUG(llvm::dbgs()
               << "[" DEBUG_TYPE "] folded the attention mask to a "
               << "per-key bias row (" << conjuncts.size()
               << " conjuncts, window " << window.value_or(-1) << ")\n");
    ++NumAttentionKeyBiasFolds;
    return mlir::success();
  }
};

} // namespace

void populateAttentionWindowFoldPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx) {
  patterns.add<AttentionStampMaskWindow>(ctx);
  patterns.add<AttentionFoldKeyPaddingMask>(ctx);
}

} // namespace hip
} // namespace mlir
