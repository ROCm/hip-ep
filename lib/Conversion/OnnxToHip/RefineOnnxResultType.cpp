/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RefineOnnxResultType.cpp - ONNX result-type rules + meet -----------===//
//
// Implementation of `refineResultTypeFromOperands` (the per-op-name
// rules) and `meetRankedTypes` (the lattice meet helper).
//
// Adding a new op:
//   * Pointwise unary or pointwise broadcast group: append the name to
//     the `kPointwise` table -- no other change needed.
//   * Otherwise: add a `compute<Op>Result` helper alongside the
//     existing ones and add a case in `refineResultTypeFromOperands`.
//
// All rules return null Type whenever they cannot make progress (any
// operand still unranked, attribute missing, etc.). Element types are
// read from the existing result type for unary / broadcast / concat /
// slice -- the importer always sets the right element type even when
// the shape is a rank-0 placeholder.
//
// Why op-name dispatch (and not an interface like
// `--hip-infer-shapes` uses):
//   * Every upstream shape-inference design is built on op
//     registration (onnx-mlir's `ShapeInferenceOpInterface`, TOSA's
//     `InferShapedTypeOpInterface`, torch-mlir's per-op rules,
//     `--hip-infer-shapes`'s `ReifyRankedShapedTypeOpInterface` via
//     TableGen sub-bases like `Hip_DpsOp_Broadcast`). The op carries
//     its category in its definition; the pass walks ops by interface.
//   * `OnnxStubDialect` deliberately leaves ONNX ops unregistered (no
//     fork of onnx-mlir's 200+ op classes). For an unregistered op,
//     `Operation*` exposes only the op name, positional operands, and
//     a generic attribute dict -- no traits, no interfaces.
//   * Pure structural heuristics are unsafe: `onnx.ReduceSum` (rank
//     changes via `keepdims`), `onnx.Reshape` (shape from operand[1]),
//     and `onnx.Cast` (changes elem type but not shape) are
//     indistinguishable structurally but have very different shape
//     rules. The whitelist is the smallest viable safety belt.
//
// What we do borrow from upstream: the broadcast math itself is
// delegated to `mlir::OpTrait::util::getBroadcastedShape`, the same
// helper used by `lib/Dialect/IR/HipShapeUtils.cpp::reifyBroadcastShape`
// and TOSA's `tosa::AddOp::inferReturnTypeComponents`. Single source
// of truth for the per-dim merge rules.
//
//===----------------------------------------------------------------------===//

#include "RefineOnnxResultType.h"

#include "mlir/Dialect/Traits.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hip {

namespace {

//===----------------------------------------------------------------------===//
// Pointwise op-name table and rule helpers.
//
// The whitelist is load-bearing: it is the safety belt against
// promoting ops whose result rank / shape is a function of attributes
// rather than operand types. Without it, a generic "result == max-rank
// operand shape" rule would silently re-promote `onnx.ReduceSum` with
// `keepdims = 0` from rank-0 back to the input's rank, and would
// inherit the data operand's shape on `onnx.Reshape` (which is driven
// by the constant `shape` operand). New ops join the whitelist only
// after a per-op rule that reads the relevant attributes.
//===----------------------------------------------------------------------===//

/// ONNX pointwise ops whose result shape is the numpy-style align-
/// right broadcast of all operand shapes. Unary ops are the
/// degenerate N=1 case (broadcast of one operand == that operand's
/// shape) and share this table.
static constexpr StringLiteral kPointwise[] = {
    // Unary  (N == 1):
    "onnx.Identity", "onnx.Cast", "onnx.CastLike", "onnx.Tanh", "onnx.Sigmoid",
    "onnx.Relu", "onnx.Gelu", "onnx.Erf", "onnx.Softmax",
    // Binary / N-ary broadcast:
    "onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div", "onnx.Min", "onnx.Max",
    "onnx.Where", "onnx.Equal", "onnx.Greater", "onnx.Less"};

/// Rule 1: pointwise. Result shape is the numpy-style align-right
/// broadcast of all operand shapes; output rank = max operand rank
/// with leading-1 padding for shorter operands. Subsumes the unary
/// case at N == 1: a single operand of shape `[d0, d1, ...]` is
/// broadcast to itself and yields the same shape.
///
/// The per-dim merge is delegated to upstream
/// `mlir::OpTrait::util::getBroadcastedShape` -- the same helper
/// `lib/Dialect/IR/HipShapeUtils.cpp::reifyBroadcastShape` and
/// `tosa::AddOp::inferReturnTypeComponents` use. Single source of
/// truth for the broadcast contract; if upstream tightens the
/// dynamic-vs-static rules we follow automatically. The helper is
/// binary, so we fold left over the operand list.
///
/// Element type is preserved from the existing result type so element-
/// type-changing ops (Cast / CastLike) and i1-result ops (Equal /
/// Greater / Less) are handled without inspecting attributes.
///
/// On broadcast failure we return null (the verifier should already
/// have caught it; the rules library bails defensively rather than
/// fabricate a wrong shape).
///
/// Before:  %r = "onnx.Add"(%a, %b)
///              : (tensor<2x3xf16>, tensor<f16>) -> tensor<f16>
/// After:   %r = "onnx.Add"(%a, %b)
///              : (tensor<2x3xf16>, tensor<f16>) -> tensor<2x3xf16>
static RankedTensorType computePointwiseResult(Operation *op,
                                               RankedTensorType existing) {
  SmallVector<int64_t> result;
  bool seeded = false;
  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    // Skip non-tensor operands and rank-0 placeholders -- the latter
    // are the importer's "unranked" stand-in; broadcasting against
    // them would clamp the result rank to 0.
    if (!t || t.getRank() == 0)
      continue;
    if (!seeded) {
      result.assign(t.getShape().begin(), t.getShape().end());
      seeded = true;
      continue;
    }
    SmallVector<int64_t> next;
    if (!OpTrait::util::getBroadcastedShape(result, t.getShape(), next))
      return {};
    result = std::move(next);
  }
  if (!seeded)
    return {};
  return RankedTensorType::get(result, existing.getElementType());
}

/// Rule 2: Concat. Result rank == operand rank; axis dim is the sum of
/// per-operand axis dims (kDynamic if any is dynamic); non-axis dims
/// take the agreement value across operands (kDynamic on disagreement
/// or any-operand-dynamic). Negative axis is wrap-corrected.
///
/// Before:  %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
///              : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
/// After:   %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
///              : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<2x7xf16>
static RankedTensorType computeConcatResult(Operation *op,
                                            RankedTensorType existing) {
  // Rank from any ranked operand. ONNX requires identical rank across
  // Concat inputs, so taking the max is equivalent for valid IR; rank-0
  // placeholders are absorbed.
  int64_t rank = 0;
  for (Value v : op->getOperands()) {
    if (auto t = dyn_cast<RankedTensorType>(v.getType()))
      rank = std::max(rank, t.getRank());
  }
  if (rank == 0)
    return {};

  // ONNX `axis` is a signed integer attribute (`si64`); use `getSInt`
  // to extract with the correct sign-extension. Same idiom as
  // ConcatConversion.cpp.
  int64_t axis = 0;
  if (auto a = op->getAttrOfType<IntegerAttr>("axis"))
    axis = a.getSInt();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return {};

  // 0 is the "unseeded" sentinel; ONNX dim values are always positive
  // or kDynamic in practice, so 0 cannot be confused with a real dim.
  // Any slot left at 0 after the merge means no operand contributed --
  // rewritten to kDynamic at the end.
  SmallVector<int64_t> shape(rank, 0);

  // If any operand is unranked or rank-mismatched (e.g. a rank-0
  // placeholder beside the true-rank operand), force the axis dim
  // dynamic up front -- we cannot reason about its contribution.
  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != rank) {
      shape[axis] = ShapedType::kDynamic;
      break;
    }
  }

  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != rank)
      continue;
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      int64_t di = t.getDimSize(i);
      if (shape[i] == ShapedType::kDynamic)
        continue;
      if (di == ShapedType::kDynamic) {
        shape[i] = ShapedType::kDynamic;
        continue;
      }
      if (i == axis) {
        // Sum along the concat axis; 0 means unseeded.
        shape[i] = (shape[i] == 0) ? di : shape[i] + di;
      } else {
        // Non-axis: require agreement, else kDynamic.
        if (shape[i] == 0)
          shape[i] = di;
        else if (shape[i] != di)
          shape[i] = ShapedType::kDynamic;
      }
    }
  }

  // Any dim still at the unseeded sentinel: no ranked operand
  // contributed. Defensively rewrite to dynamic.
  for (int64_t i : llvm::seq<int64_t>(0, rank)) {
    if (shape[i] == 0)
      shape[i] = ShapedType::kDynamic;
  }

  return RankedTensorType::get(shape, existing.getElementType());
}

/// Rule 3: Slice. Rank-preserving; all dims kDynamic. Conservative --
/// a precise rule would constant-fold starts / ends / axes / steps,
/// but Slice is rare in the contexts where this rules library is
/// consulted and the all-dynamic-rank-N result is enough for
/// downstream conversion to make progress.
///
/// Before:  %r = "onnx.Slice"(%x, ...)
///              : (tensor<2x3xf16>, ...) -> tensor<f16>
/// After:   %r = "onnx.Slice"(%x, ...)
///              : (tensor<2x3xf16>, ...) -> tensor<?x?xf16>
static RankedTensorType computeSliceResult(Operation *op,
                                           RankedTensorType existing) {
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  if (!inputType)
    return {};
  SmallVector<int64_t> shape(inputType.getRank(), ShapedType::kDynamic);
  return RankedTensorType::get(shape, existing.getElementType());
}

/// Rule 4: LayerNormalization, result 0 only. Returns null for results
/// 1 / 2 (mean / inv-std) so the caller leaves them alone -- this
/// rules library does not yet model their reduced-rank shapes.
/// Result 0 is shape-preserving (same as the data operand), so it
/// reuses the pointwise rule.
///
/// Before:  %r = "onnx.LayerNormalization"(%x, %scale, %bias)
///              : (tensor<2x3xf16>, ...) -> tensor<f16>
/// After:   %r = "onnx.LayerNormalization"(%x, %scale, %bias)
///              : (tensor<2x3xf16>, ...) -> tensor<2x3xf16>
static RankedTensorType computeLayerNormResult(Operation *op,
                                               unsigned resultIdx,
                                               RankedTensorType existing) {
  if (resultIdx != 0)
    return {};
  return computePointwiseResult(op, existing);
}

} // namespace

RankedTensorType refineResultTypeFromOperands(Operation *op,
                                              unsigned resultIdx) {
  if (resultIdx >= op->getNumResults() || op->getNumOperands() == 0)
    return {};
  auto existing =
      dyn_cast<RankedTensorType>(op->getResult(resultIdx).getType());
  if (!existing)
    return {};

  StringRef name = op->getName().getStringRef();
  if (llvm::is_contained(kPointwise, name))
    return computePointwiseResult(op, existing);
  if (name == "onnx.Concat")
    return computeConcatResult(op, existing);
  if (name == "onnx.Slice")
    return computeSliceResult(op, existing);
  if (name == "onnx.LayerNormalization")
    return computeLayerNormResult(op, resultIdx, existing);

  // No rule -- return null so the caller leaves the op's result type
  // alone. The omission is intentional for ops whose result rank is a
  // function of attributes, not operand types (e.g. onnx.ReduceSum
  // with `keepdims = 0`, onnx.Reshape with a constant `shape`); the
  // rules library would have to read those attributes to produce a
  // safe candidate.
  return {};
}

RankedTensorType meetRankedTypes(RankedTensorType old,
                                 RankedTensorType candidate) {
  if (!candidate)
    return old;
  if (!old)
    return candidate;

  if (old.getElementType() != candidate.getElementType())
    return {};

  int64_t oldRank = old.getRank();
  int64_t newRank = candidate.getRank();

  // Rank promotion: rank-0 placeholder + ranked candidate -> candidate.
  if (oldRank == 0 && newRank > 0)
    return candidate;

  // Same rank: per-dim meet.
  if (oldRank == newRank) {
    ArrayRef<int64_t> oldShape = old.getShape();
    ArrayRef<int64_t> newShape = candidate.getShape();
    SmallVector<int64_t> merged(oldRank);
    for (int64_t i : llvm::seq<int64_t>(0, oldRank)) {
      int64_t a = oldShape[i];
      int64_t b = newShape[i];
      if (a == ShapedType::kDynamic)
        merged[i] = b;
      else if (b == ShapedType::kDynamic)
        merged[i] = a;
      else if (a == b)
        merged[i] = a;
      else
        return {}; // Two statics disagree.
    }
    return RankedTensorType::get(merged, old.getElementType());
  }

  // Any other rank disagreement is a conflict (we don't shrink rank,
  // and we don't promote from rank-N to rank-M for N > 0).
  return {};
}

} // namespace hip
} // namespace mlir
