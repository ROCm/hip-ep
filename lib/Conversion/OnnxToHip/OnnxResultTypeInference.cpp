/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxResultTypeInference.cpp - rules library + dialect fallback -===//
//
// FallbackModel implementation of `OnnxResultTypeInferenceInterface`.
// `OnnxResultTypeInferenceFallback::computeResultType` switches on
// `op->getName()` and delegates to per-op rule helpers. Returns null
// Type for any op outside the rules library; callers (e.g.
// `OnnxLoopOutlinePass`'s body op refinement) MUST treat null as "leave
// the op alone" -- the safety belt against false promotion of
// rank-changing ops we have not yet reasoned about.
//
// Rule groups currently in the library:
//
//   1. Pointwise unary -- result shape from operand[0]; element type
//      preserved from the existing result type so element-type-changing
//      ops (Cast, CastLike) are handled without inspecting attributes.
//      Ops: Identity, Cast, CastLike, Tanh, Sigmoid, Relu, Gelu, Erf,
//           Softmax.
//
//   2. Pointwise broadcast (N >= 2 operands) -- numpy-style align-right
//      broadcast over operand shapes, output rank = max operand rank,
//      per-dim take the max (kDynamic if any operand is dynamic). The
//      element type is preserved from the existing result type so
//      comparison ops (Equal, Greater, Less) keep their i1 result.
//      Ops: Add, Sub, Mul, Div, Min, Max, Where, Equal, Greater, Less.
//
//   3. Concat -- axis dim is the sum of per-operand axis dims
//      (kDynamic if any is dynamic); non-axis dims agree-or-dynamic.
//
//   4. Slice -- rank-preserving, all dims kDynamic. A precise rule
//      would constant-fold starts/ends/axes/steps, but Slice is rare
//      in the contexts where this rules library is consulted today.
//
//   5. LayerNormalization -- result 0 has operand[0]'s shape; results
//      1 / 2 (mean, inv-std) are not handled.
//
// Adding a new op:
//   * "Pointwise unary" or "pointwise broadcast" group: add the name
//     to the corresponding `is*` predicate; nothing else needed.
//   * Otherwise add a `compute<Op>Result` helper alongside
//     `computeConcatResult` and dispatch to it from the switch.
//
// IR before/after (canonical case -- onnx.Concat in a Loop body):
//
//   // Before (cloned from onnx.Loop body, ONNX shape inference did
//   // not recurse so the result was a rank-0 placeholder):
//   %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
//        : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
//
//   // After (Concat rule sums the axis dim; non-axis dims agree):
//   %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
//        : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<2x7xf16>
//
//===----------------------------------------------------------------------===//

#include "hip/Conversion/OnnxToHip/OnnxResultTypeInference.h"
#include "hip/InitAllPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/TypeID.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

namespace mlir {
namespace hip {

namespace {

//===----------------------------------------------------------------------===//
// Op-name predicates for the two "uniform rule" groups.
//===----------------------------------------------------------------------===//

bool isPointwiseUnary(StringRef name) {
  return llvm::is_contained(
      ArrayRef<StringLiteral>{"onnx.Identity", "onnx.Cast", "onnx.CastLike",
                              "onnx.Tanh", "onnx.Sigmoid", "onnx.Relu",
                              "onnx.Gelu", "onnx.Erf", "onnx.Softmax"},
      name);
}

bool isPointwiseBroadcast(StringRef name) {
  return llvm::is_contained(
      ArrayRef<StringLiteral>{"onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div",
                              "onnx.Min", "onnx.Max", "onnx.Where",
                              "onnx.Equal", "onnx.Greater", "onnx.Less"},
      name);
}

//===----------------------------------------------------------------------===//
// Per-op rule helpers.
//===----------------------------------------------------------------------===//

/// Rule 1: pointwise unary. Result shape == operand[0] shape; result
/// element type preserved from the existing result type.
///
/// Before:  %r = "onnx.Tanh"(%x) : (tensor<2x3xf16>) -> tensor<f16>
/// After:   %r = "onnx.Tanh"(%x) : (tensor<2x3xf16>) -> tensor<2x3xf16>
Type computeUnaryResult(Operation *op, unsigned resultIdx) {
  auto inputType = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto resultType =
      dyn_cast<RankedTensorType>(op->getResult(resultIdx).getType());
  if (!inputType || !resultType)
    return {};
  return RankedTensorType::get(inputType.getShape(),
                               resultType.getElementType());
}

/// Rule 2: pointwise broadcast (N >= 2 operands). Numpy-style: align
/// operand shapes right, output rank = max operand rank, per-dim take
/// the max of operand dims at that position (kDynamic when any operand
/// is dynamic at that position).
///
/// Before:  %r = "onnx.Add"(%a, %b)
///              : (tensor<2x3xf16>, tensor<f16>) -> tensor<f16>
/// After:   %r = "onnx.Add"(%a, %b)
///              : (tensor<2x3xf16>, tensor<f16>) -> tensor<2x3xf16>
Type computeBroadcastResult(Operation *op, unsigned resultIdx) {
  auto resultType =
      dyn_cast<RankedTensorType>(op->getResult(resultIdx).getType());
  if (!resultType)
    return {};

  // First pass: output rank = max of operand ranks. None-typed and
  // non-ranked operands contribute no shape information (typical for
  // `Where`'s splat condition or operands still in flight through
  // conversion); skip them.
  int64_t maxRank = 0;
  for (Value v : op->getOperands()) {
    if (auto t = dyn_cast<RankedTensorType>(v.getType()))
      maxRank = std::max(maxRank, t.getRank());
  }
  if (maxRank == 0)
    return {};

  // Second pass: per-dim broadcast. shape[outIdx] starts at 1 (the
  // implicit broadcast neutral). For each operand, align right and
  // merge.
  SmallVector<int64_t> shape(maxRank, 1);
  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t)
      continue;
    int64_t r = t.getRank();
    int64_t off = maxRank - r;
    for (int64_t i : llvm::seq<int64_t>(0, r)) {
      int64_t outIdx = off + i;
      int64_t d = t.getDimSize(i);
      if (d == ShapedType::kDynamic) {
        shape[outIdx] = ShapedType::kDynamic;
      } else if (shape[outIdx] == 1) {
        // First non-1 contribution at this position wins.
        shape[outIdx] = d;
      } else if (shape[outIdx] != d && d != 1 &&
                 shape[outIdx] != ShapedType::kDynamic) {
        // Conflicting concrete dims at the same position is invalid
        // broadcast IR; fall back to dynamic rather than abort.
        shape[outIdx] = ShapedType::kDynamic;
      }
    }
  }
  return RankedTensorType::get(shape, resultType.getElementType());
}

/// Rule 3: Concat. Result rank == operand rank; axis dim is the sum of
/// per-operand axis dims (kDynamic if any is dynamic); non-axis dims
/// take the agreement value across operands (kDynamic on disagreement
/// or any-operand-dynamic).
///
/// Before:  %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
///              : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
/// After:   %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
///              : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<2x7xf16>
Type computeConcatResult(Operation *op, unsigned resultIdx) {
  auto resultType =
      dyn_cast<RankedTensorType>(op->getResult(resultIdx).getType());
  if (!resultType)
    return {};

  // Rank from the first ranked operand. ONNX requires identical rank
  // across Concat inputs, so taking the max is equivalent for valid IR.
  int64_t rank = 0;
  for (Value v : op->getOperands()) {
    if (auto t = dyn_cast<RankedTensorType>(v.getType()))
      rank = std::max(rank, t.getRank());
  }
  if (rank == 0)
    return {};

  // axis attribute, with negative-index wrap.
  int64_t axis = 0;
  if (auto a = op->getAttrOfType<IntegerAttr>("axis"))
    axis = a.getInt();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return {};

  // 0 is the "unseeded" sentinel: a real ONNX dim is always > 0 or
  // kDynamic. Any dim left at 0 after the merge means no operand
  // contributed to it (impossible for valid Concat IR but defensively
  // rewritten to kDynamic below).
  SmallVector<int64_t> shape(rank, 0);

  // If any operand is unranked or rank-mismatched (e.g. operand[0] is
  // a rank-0 placeholder while operand[1] carries the true rank), we
  // cannot reason about its contribution to the axis dim -- force the
  // axis dim dynamic up front.
  bool axisIsUnknown = false;
  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != rank) {
      axisIsUnknown = true;
      break;
    }
  }
  if (axisIsUnknown)
    shape[axis] = ShapedType::kDynamic;

  for (Value v : op->getOperands()) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (!t || t.getRank() != rank)
      continue;
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      int64_t di = t.getDimSize(i);
      if (shape[i] == ShapedType::kDynamic) {
        // kDynamic is absorbing on both axes; nothing further to merge.
        continue;
      }
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

  return RankedTensorType::get(shape, resultType.getElementType());
}

/// Rule 4: Slice. Rank-preserving; all dims kDynamic.
///
/// Before:  %r = "onnx.Slice"(%x, ...) : (tensor<2x3xf16>, ...) -> tensor<f16>
/// After:   %r = "onnx.Slice"(%x, ...)
///              : (tensor<2x3xf16>, ...) -> tensor<?x?xf16>
Type computeSliceResult(Operation *op, unsigned resultIdx) {
  auto t0 = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
  auto resultType =
      dyn_cast<RankedTensorType>(op->getResult(resultIdx).getType());
  if (!t0 || !resultType)
    return {};
  SmallVector<int64_t> shape(t0.getRank(), ShapedType::kDynamic);
  return RankedTensorType::get(shape, resultType.getElementType());
}

/// Rule 5: LayerNormalization, result 0 only. Returns null for
/// resultIdx > 0 (mean / inv-std) so the caller leaves them alone.
///
/// Before:  %r = "onnx.LayerNormalization"(%x, %scale, %bias)
///              : (tensor<2x3xf16>, ...) -> tensor<f16>
/// After:   %r = "onnx.LayerNormalization"(%x, %scale, %bias)
///              : (tensor<2x3xf16>, ...) -> tensor<2x3xf16>
Type computeLayerNormResult(Operation *op, unsigned resultIdx) {
  if (resultIdx != 0)
    return {};
  return computeUnaryResult(op, resultIdx);
}

//===----------------------------------------------------------------------===//
// Singleton FallbackModel + dispatch switch.
//===----------------------------------------------------------------------===//

struct OnnxResultTypeInferenceFallback
    : public OnnxResultTypeInferenceInterface::FallbackModel<
          OnnxResultTypeInferenceFallback> {
  Type computeResultType(Operation *op, unsigned resultIdx) const {
    if (resultIdx >= op->getNumResults() || op->getNumOperands() == 0)
      return {};
    StringRef name = op->getName().getStringRef();

    if (isPointwiseUnary(name))
      return computeUnaryResult(op, resultIdx);
    if (isPointwiseBroadcast(name))
      return computeBroadcastResult(op, resultIdx);
    if (name == "onnx.Concat")
      return computeConcatResult(op, resultIdx);
    if (name == "onnx.Slice")
      return computeSliceResult(op, resultIdx);
    if (name == "onnx.LayerNormalization")
      return computeLayerNormResult(op, resultIdx);

    // No rule -- safety belt: caller must treat null as "leave the
    // op alone".
    return {};
  }
};

} // namespace

void registerOnnxResultTypeInferenceFallback() {
  // Function-local static keeps the FallbackModel alive for the lifetime
  // of the process; OnnxStubDialect stores a non-owning void*. The
  // registry is a DenseMap that overwrites on insert, so re-registering
  // the same (interface TypeID, fallback ptr) pair is a no-op.
  static OnnxResultTypeInferenceFallback fallback;
  ::hip::compiler::detail::OnnxStubDialect::registerInterfaceFallback(
      TypeID::get<OnnxResultTypeInferenceInterface>(), &fallback);
}

} // namespace hip
} // namespace mlir
