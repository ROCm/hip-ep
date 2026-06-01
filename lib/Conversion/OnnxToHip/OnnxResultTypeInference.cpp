/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OnnxResultTypeInference.cpp - rules library + dialect fallback -===//
//
// FallbackModel implementation of `OnnxResultTypeInferenceInterface`.
//
// `OnnxResultTypeInferenceFallback::computeResultType` is the dispatch
// point: it switches on `op->getName()` and delegates to per-op rule
// helpers for each ONNX op the rules library knows about. Returns null
// Type for every op outside the rules library; callers (e.g.
// `OnnxLoopOutlinePass`'s body op refinement) MUST treat null as "leave
// the op alone" -- this is the safety belt against false promotion of
// rank-changing ops we have not yet reasoned about.
//
// Rules currently in the library (group + ops):
//
//   1. Pointwise unary (1 operand; result shape == operand[0] shape;
//      element type taken from the EXISTING result type so element-type-
//      changing ops like Cast / CastLike are handled without inspecting
//      attributes -- the cloned IR's result element type is correct;
//      only the rank/shape may be stale):
//        Identity, Cast, CastLike, Tanh, Sigmoid, Relu, Gelu, Erf,
//        Softmax
//
//   2. Pointwise broadcast (N >= 2 operands; result rank == max operand
//      rank; per-dim numpy-style broadcast: align right, take max of
//      operand dims (or 1 if missing), kDynamic when ANY operand has
//      dynamic dim at that position; element type taken from the
//      existing result type so comparison ops like Equal / Greater /
//      Less keep their i1 result element type):
//        Add, Sub, Mul, Div, Min, Max, Where, Equal, Greater, Less
//
//   3. Concat (axis dim is sum of operand axis dims, kDynamic if any
//      operand axis dim is dynamic; non-axis dims agreement-or-dynamic):
//        Concat
//
//   4. Slice (rank-preserving, all dims kDynamic -- a precise rule
//      would parse starts/ends/axes/steps inputs but Slice is rare in
//      Loop bodies and the rank is the load-bearing fact for the
//      conversion patterns):
//        Slice
//
//   5. LayerNormalization (result 0 has operand[0] shape; results 1
//      and 2 -- mean and inv-std -- are not handled, the rule returns
//      null for resultIdx > 0 so the caller leaves the op alone):
//        LayerNormalization
//
// Adding a new op:
//   * If it fits "pointwise unary" or "pointwise broadcast", add the
//     name to the corresponding `is*` predicate -- nothing else needed.
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
//   // After (the Concat rule sums the axis dim; non-axis dims agree):
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

bool isPointwiseUnary(::llvm::StringRef name) {
  return llvm::is_contained(
      ::llvm::ArrayRef<::llvm::StringLiteral>{
          "onnx.Identity", "onnx.Cast", "onnx.CastLike", "onnx.Tanh",
          "onnx.Sigmoid", "onnx.Relu", "onnx.Gelu", "onnx.Erf", "onnx.Softmax"},
      name);
}

bool isPointwiseBroadcast(::llvm::StringRef name) {
  return llvm::is_contained(
      ::llvm::ArrayRef<::llvm::StringLiteral>{
          "onnx.Add", "onnx.Sub", "onnx.Mul", "onnx.Div", "onnx.Min",
          "onnx.Max", "onnx.Where", "onnx.Equal", "onnx.Greater", "onnx.Less"},
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
::mlir::Type computeUnaryResult(::mlir::Operation *op, unsigned resultIdx) {
  auto inputType =
      ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getOperand(0).getType());
  auto resultType = ::mlir::dyn_cast<::mlir::RankedTensorType>(
      op->getResult(resultIdx).getType());
  if (!inputType || !resultType)
    return {};
  return ::mlir::RankedTensorType::get(inputType.getShape(),
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
::mlir::Type computeBroadcastResult(::mlir::Operation *op, unsigned resultIdx) {
  auto resultType = ::mlir::dyn_cast<::mlir::RankedTensorType>(
      op->getResult(resultIdx).getType());
  if (!resultType)
    return {};

  // First pass: determine the output rank as the max of operand ranks.
  // None-typed and non-ranked operands are skipped (they contribute no
  // shape information; common for `Where`'s condition when it is a
  // splat / for ops in the middle of conversion where one operand is
  // still an opaque value).
  int64_t maxRank = 0;
  for (::mlir::Value v : op->getOperands()) {
    if (auto t = ::mlir::dyn_cast<::mlir::RankedTensorType>(v.getType()))
      maxRank = std::max(maxRank, t.getRank());
  }
  if (maxRank == 0)
    return {};

  // Second pass: per-dim broadcast. shape[outIdx] starts at 1 (the
  // implicit broadcast neutral). For each operand, align right and
  // merge.
  ::llvm::SmallVector<int64_t> shape(maxRank, 1);
  for (::mlir::Value v : op->getOperands()) {
    auto t = ::mlir::dyn_cast<::mlir::RankedTensorType>(v.getType());
    if (!t)
      continue;
    int64_t r = t.getRank();
    int64_t off = maxRank - r;
    for (int64_t i = 0; i < r; ++i) {
      int64_t outIdx = off + i;
      int64_t d = t.getDimSize(i);
      if (d == ::mlir::ShapedType::kDynamic) {
        shape[outIdx] = ::mlir::ShapedType::kDynamic;
      } else if (shape[outIdx] == 1) {
        // First non-1 contribution at this position wins.
        shape[outIdx] = d;
      } else if (shape[outIdx] != d && d != 1 &&
                 shape[outIdx] != ::mlir::ShapedType::kDynamic) {
        // Conflicting concrete dims at the same position: invalid IR
        // for broadcast, but be conservative rather than abort.
        shape[outIdx] = ::mlir::ShapedType::kDynamic;
      }
    }
  }
  return ::mlir::RankedTensorType::get(shape, resultType.getElementType());
}

/// Rule 3: Concat. Result rank == operand rank; axis dim is the sum of
/// per-operand axis dims (kDynamic if any is dynamic); non-axis dims
/// take the agreement value across operands (kDynamic when operands
/// disagree or any is dynamic).
///
/// Before:  %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
///              : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<f16>
/// After:   %r = "onnx.Concat"(%a, %b) {axis = 1 : si64}
///              : (tensor<2x3xf16>, tensor<2x4xf16>) -> tensor<2x7xf16>
::mlir::Type computeConcatResult(::mlir::Operation *op, unsigned resultIdx) {
  auto resultType = ::mlir::dyn_cast<::mlir::RankedTensorType>(
      op->getResult(resultIdx).getType());
  if (!resultType)
    return {};

  // Determine rank from the first ranked operand. ONNX requires all
  // Concat inputs to have identical rank, so taking the max would be
  // equivalent for valid IR.
  int64_t rank = 0;
  for (::mlir::Value v : op->getOperands()) {
    if (auto t = ::mlir::dyn_cast<::mlir::RankedTensorType>(v.getType()))
      rank = std::max(rank, t.getRank());
  }
  if (rank == 0)
    return {};

  // axis attribute, with negative-index wrap.
  int64_t axis = 0;
  if (auto a = op->getAttrOfType<::mlir::IntegerAttr>("axis"))
    axis = a.getInt();
  if (axis < 0)
    axis += rank;
  if (axis < 0 || axis >= rank)
    return {};

  // We use 0 as a "not yet seeded" sentinel (a real ONNX dim is always
  // > 0 or kDynamic; never 0 except for the empty-tensor edge case
  // which Concat cannot produce). After the merge loop any dim still
  // at 0 means no operand contributed to it -- impossible for valid
  // Concat IR but defensively rewritten to kDynamic.
  ::llvm::SmallVector<int64_t> shape(rank, 0);

  // If ANY operand is unranked or has a rank that doesn't match the
  // max (e.g. operand[0] is a rank-0 placeholder while operand[1]
  // carries the true rank-N type), we cannot reason about that
  // operand's contribution to the axis dim -- force the axis dim
  // dynamic up front.
  bool axisIsUnknown = false;
  for (::mlir::Value v : op->getOperands()) {
    auto t = ::mlir::dyn_cast<::mlir::RankedTensorType>(v.getType());
    if (!t || t.getRank() != rank) {
      axisIsUnknown = true;
      break;
    }
  }
  if (axisIsUnknown)
    shape[axis] = ::mlir::ShapedType::kDynamic;

  for (::mlir::Value v : op->getOperands()) {
    auto t = ::mlir::dyn_cast<::mlir::RankedTensorType>(v.getType());
    if (!t || t.getRank() != rank)
      continue;
    for (int64_t i = 0; i < rank; ++i) {
      int64_t di = t.getDimSize(i);
      if (shape[i] == ::mlir::ShapedType::kDynamic) {
        // Already absorbed kDynamic earlier this loop; nothing further
        // to merge -- kDynamic is the absorbing element on both axes.
        continue;
      }
      if (di == ::mlir::ShapedType::kDynamic) {
        shape[i] = ::mlir::ShapedType::kDynamic;
        continue;
      }
      if (i == axis) {
        // Sum along the concat axis; 0 means we haven't seeded yet.
        shape[i] = (shape[i] == 0) ? di : shape[i] + di;
      } else {
        // Non-axis dim: agreement, otherwise dynamic.
        if (shape[i] == 0)
          shape[i] = di;
        else if (shape[i] != di)
          shape[i] = ::mlir::ShapedType::kDynamic;
      }
    }
  }

  // Any dim still at the unseeded sentinel: no ranked operand
  // contributed. Treat as dynamic to keep the helper defensive against
  // pathologically-typed Concat IR (e.g. all operands rank-mismatched
  // or non-tensor).
  for (int64_t i = 0; i < rank; ++i) {
    if (shape[i] == 0)
      shape[i] = ::mlir::ShapedType::kDynamic;
  }

  return ::mlir::RankedTensorType::get(shape, resultType.getElementType());
}

/// Rule 4: Slice. Rank-preserving; all dims kDynamic. A precise rule
/// would constant-fold starts/ends/axes/steps, but Slice is rare in
/// Loop bodies and the rank is the load-bearing fact for downstream
/// rank-aware patterns.
///
/// Before:  %r = "onnx.Slice"(%x, ...) : (tensor<2x3xf16>, ...) -> tensor<f16>
/// After:   %r = "onnx.Slice"(%x, ...) : (tensor<2x3xf16>, ...) ->
/// tensor<?x?xf16>
::mlir::Type computeSliceResult(::mlir::Operation *op, unsigned resultIdx) {
  auto t0 =
      ::mlir::dyn_cast<::mlir::RankedTensorType>(op->getOperand(0).getType());
  auto resultType = ::mlir::dyn_cast<::mlir::RankedTensorType>(
      op->getResult(resultIdx).getType());
  if (!t0 || !resultType)
    return {};
  ::llvm::SmallVector<int64_t> shape(t0.getRank(),
                                     ::mlir::ShapedType::kDynamic);
  return ::mlir::RankedTensorType::get(shape, resultType.getElementType());
}

/// Rule 5: LayerNormalization, result 0 only. Result 0 has operand[0]'s
/// shape; results 1 (mean) and 2 (inv-std) are not handled (the rule
/// returns null for resultIdx > 0 so the caller leaves the op alone).
///
/// Before:  %r = "onnx.LayerNormalization"(%x, %scale, %bias)
///              : (tensor<2x3xf16>, ...) -> tensor<f16>
/// After:   %r = "onnx.LayerNormalization"(%x, %scale, %bias)
///              : (tensor<2x3xf16>, ...) -> tensor<2x3xf16>
::mlir::Type computeLayerNormResult(::mlir::Operation *op, unsigned resultIdx) {
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
  ::mlir::Type computeResultType(::mlir::Operation *op,
                                 unsigned resultIdx) const {
    if (resultIdx >= op->getNumResults() || op->getNumOperands() == 0)
      return {};
    ::llvm::StringRef name = op->getName().getStringRef();

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
  // of the process; OnnxStubDialect just stores a non-owning void*.
  // `registerInterfaceFallback` is idempotent: re-registering the same
  // (interface TypeID, fallback ptr) pair is a no-op since the registry
  // is a DenseMap that overwrites on insert.
  static OnnxResultTypeInferenceFallback s_fallback;
  ::hip::compiler::detail::OnnxStubDialect::registerInterfaceFallback(
      ::mlir::TypeID::get<OnnxResultTypeInferenceInterface>(), &s_fallback);
}

} // namespace hip
} // namespace mlir
