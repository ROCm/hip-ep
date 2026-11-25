/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlShapeRule.h - Deriving a result's extents ------------*- C++ -*-===//
//
// How an operation's result extents follow from its operands' is a property of
// the operation, so each operation states it, beside its verifier. This is what
// a rule is handed to state it with.
//
// A rule reads its operands' extents, builds the result's, and says so. It
// never asks what the capture observed, because the point of the derivation is
// to reach an expression the observation cannot supply -- except where GGML
// only gives the boundaries and not the way they were drawn, which is the one
// thing `relayout` is for.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_IR_GGMLSHAPERULE_H
#define HIP_DIALECT_GGML_IR_GGMLSHAPERULE_H

#include "hip/Dialect/Ggml/IR/GgmlShapeArithmetic.h"

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LLVM.h"

#include <cstdint>

#include <optional>

namespace mlir {
namespace ggml {

/// GGML tensors are rank four, always.
inline constexpr unsigned kShapeRank = 4;

using Shape = SmallVector<shape_expr::Expr, kShapeRank>;

/// The extents a value has and the byte strides it walks them with.
struct Walk {
  Shape extents;
  Shape strides;
};

/// What GGML requires of an operation's operands.
///
/// Most extents follow from what an operation does, and a rule derives them.
/// A few are chosen by whoever built the graph and recorded nowhere -- GGML's
/// `ggml_repeat_4d` takes four plain integers and keeps no tensor to trace
/// them to. What is left of them is the preconditions the operations that
/// consume the result assert, which say that one operand's extent is another
/// operand's extent. An operation states its own here; the pass carries them
/// to the operation that needs them.
class ShapeConstraints {
public:
  virtual ~ShapeConstraints() = default;

  /// GGML requires the operand at `slot` to have, at `dimension`, the extent
  /// the operand at `otherSlot` has at `otherDimension`.
  virtual void equate(int64_t slot, unsigned dimension, int64_t otherSlot,
                      unsigned otherDimension) = 0;
};

/// What a shape rule may read and what it produces.
///
/// The derivation owns the arithmetic, so a rule adds and divides through it
/// and a failure -- a division that does not divide, a product of two varying
/// extents -- is collected once rather than threaded through every step.
class ShapeDerivation {
public:
  virtual ~ShapeDerivation() = default;

  /// The extents already derived for the operand at one GGML slot, or an empty
  /// shape when the operation was not given that operand.
  virtual const Shape &operandShape(int64_t slot) const = 0;
  virtual bool hasOperand(int64_t slot) const = 0;
  /// The byte strides already derived for that operand.
  virtual const Shape &operandStrides(int64_t slot) const = 0;

  /// The extents the capture observed for the result. A rule needs these only
  /// when GGML records where the dimension boundaries ended up rather than how
  /// they were drawn.
  virtual ArrayRef<int64_t> observedResultExtents() const = 0;
  /// The byte strides the capture observed for the result, in the same order.
  virtual ArrayRef<int64_t> observedResultStrides() const = 0;

  /// Redraws `source`'s boundaries to land on `observed`, which is what a
  /// reshape or a zero-offset view does: every element keeps its linear index
  /// and only the boundaries move. Fails when no assignment of the source's
  /// varying extents reaches them.
  virtual std::optional<Shape> relayout(const Shape &source,
                                        ArrayRef<int64_t> observed) = 0;

  /// Reads an observed walk against the source it walks.
  ///
  /// A view's extents and strides are not derived from anything -- GGML
  /// records the four numbers whoever built the graph asked for. What can be
  /// recovered is which of them carry a symbol, and the evidence for that is
  /// the stride: a dimension that steps by one of the source's strides is
  /// walking that source dimension, and spans either the whole of it or a
  /// constant part.
  virtual std::optional<Walk> recoverWalk(const Shape &sourceExtents,
                                          const Shape &sourceStrides,
                                          ArrayRef<int64_t> extents,
                                          ArrayRef<int64_t> strides) = 0;

  virtual shape_expr::Arithmetic &arithmetic() = 0;

  /// The extents the result has. A rule calls this exactly once.
  virtual void setResultShape(Shape shape) = 0;
  /// The byte strides the result has, when relabelling axes carried the
  /// source's rather than the result's own layout implying them.
  virtual void setResultStrides(Shape strides) = 0;

  shape_expr::Expr constant(int64_t value) const {
    return shape_expr::constant(value);
  }
};

} // namespace ggml
} // namespace mlir

#include "hip/Dialect/Ggml/IR/GgmlShapeRuleInterface.h.inc"

#endif // HIP_DIALECT_GGML_IR_GGMLSHAPERULE_H
