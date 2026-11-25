/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- OpsShared.h - What more than one operation needs ----------------===//
//
// An operation's verifier and its shape rule are written beside it, one file
// each. What several of them need in common is here, and nothing else is: a
// helper only one operation uses belongs in that operation's file, where the
// reason for it is legible.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_IR_OPS_OPSSHARED_H
#define HIP_DIALECT_GGML_IR_OPS_OPSSHARED_H

#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlOps.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeRule.h"

#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/MathExtras.h"

#include <array>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>

namespace mlir::ggml::shared {

/// What GGML asserts of an operation's operands: extent `dimension` of the
/// operand at `slot` is extent `otherDimension` of the one at `otherSlot`.
/// Said where the capture saw the two equal and more than one (an extent of
/// one is a broadcast, or no evidence).
inline void equateCaptured(Value operand, int64_t slot, unsigned dimension,
                           Value other, int64_t otherSlot,
                           unsigned otherDimension,
                           ShapeConstraints &constraints) {
  auto a = operand ? dyn_cast<RankedTensorType>(operand.getType()) : nullptr;
  auto b = other ? dyn_cast<RankedTensorType>(other.getType()) : nullptr;
  if (!a || !b || a.getRank() != kGgmlRank || b.getRank() != kGgmlRank)
    return;
  // The tensor type spells GGML's dimensions outermost first.
  const int64_t x = a.getDimSize(kGgmlRank - 1 - dimension);
  const int64_t y = b.getDimSize(kGgmlRank - 1 - otherDimension);
  if (ShapedType::isDynamic(x) || x != y || x <= 1)
    return;
  constraints.equate(slot, dimension, otherSlot, otherDimension);
}

/// Elementwise and normalizing operations keep one operand's extents. GGML
/// broadcasts the second operand into the first, so the result never takes its
/// extents from the one being repeated.
inline LogicalResult sameShapeAs(Operation *op, ShapeDerivation &derivation,
                                 int64_t slot) {
  if (!derivation.hasOperand(slot))
    return op->emitOpError() << "expects an operand at GGML slot " << slot;
  derivation.setResultShape(derivation.operandShape(slot));
  return success();
}

/// A reshape, a contiguous copy, and a zero-offset view all keep every element
/// at the same linear index and only move the dimension boundaries. GGML
/// records where the boundaries ended up, not how they were drawn, so this is
/// the one rule that reads what the capture observed.
inline LogicalResult redrawBoundaries(Operation *op,
                                      ShapeDerivation &derivation) {
  if (!derivation.hasOperand(0))
    return op->emitOpError() << "expects an operand";
  std::optional<Shape> resultShape = derivation.relayout(
      derivation.operandShape(0), derivation.observedResultExtents());
  if (!resultShape) {
    return op->emitOpError() << "cannot recover its extents: "
                             << derivation.arithmetic().getFailure();
  }
  derivation.setResultShape(std::move(*resultShape));
  return success();
}

inline LogicalResult verifyFiniteFloat(Operation *op, FloatAttr attribute,
                                       llvm::StringRef name) {
  if (!attribute.getValue().isFinite()) {
    return op->emitOpError() << "requires `" << name << "` to be finite";
  }
  return success();
}

inline LogicalResult verifyFiniteNonNegativeFloat(Operation *op,
                                                  FloatAttr attribute,
                                                  llvm::StringRef name) {
  if (failed(verifyFiniteFloat(op, attribute, name))) {
    return failure();
  }
  if (attribute.getValueAsDouble() < 0.0) {
    return op->emitOpError() << "requires `" << name << "` to be non-negative";
  }
  return success();
}

inline LogicalResult verifyFinitePositiveFloat(Operation *op,
                                               FloatAttr attribute,
                                               llvm::StringRef name) {
  if (failed(verifyFiniteFloat(op, attribute, name))) {
    return failure();
  }
  if (!(attribute.getValueAsDouble() > 0.0)) {
    return op->emitOpError() << "requires `" << name << "` to be positive";
  }
  return success();
}

inline bool compatibleDimensions(int64_t lhs, int64_t rhs) {
  return ShapedType::isDynamic(lhs) || ShapedType::isDynamic(rhs) || lhs == rhs;
}

/// Whether two tensors hold the same elements in the same arrangement.
///
/// This is what an operation means when it says its result matches its input:
/// the extents and the element type, and nothing about how the bytes are laid
/// out. Comparing the types themselves would also demand the same strides,
/// which no operation here intends -- a freshly allocated result is dense even
/// when its input is a strided view of something else.
inline bool haveSameShapeAndElementType(RankedTensorType lhs,
                                        RankedTensorType rhs) {
  return lhs.getShape() == rhs.getShape() &&
         lhs.getElementType() == rhs.getElementType();
}

struct CardinalitySummary {
  int64_t staticProduct = 1;
  unsigned dynamicDimensions = 0;
  bool overflow = false;
};

inline CardinalitySummary summarizeCardinality(RankedTensorType type) {
  CardinalitySummary summary;
  const bool isEmpty = llvm::is_contained(type.getShape(), int64_t{0});
  if (isEmpty)
    summary.staticProduct = 0;
  for (int64_t dimension : type.getShape()) {
    if (ShapedType::isDynamic(dimension)) {
      ++summary.dynamicDimensions;
      continue;
    }
    if (isEmpty)
      continue;
    if (summary.staticProduct >
        std::numeric_limits<int64_t>::max() / dimension) {
      summary.overflow = true;
      continue;
    }
    summary.staticProduct *= dimension;
  }
  return summary;
}

inline LogicalResult verifyAffineExpression(Operation *op, Attribute attribute,
                                            llvm::StringRef name,
                                            bool allowZeroConstant) {
  std::optional<ShapeExprView> view = getShapeExprView(attribute);
  if (!view) {
    return op->emitOpError() << "requires `" << name
                             << "` to be an integer or a shape expression";
  }
  // A varying quantity is bounded by the symbols it names rather than by any
  // one value, so only a constant can be checked for sign here.
  if (view->isConstant() &&
      (view->offset < 0 || (!allowZeroConstant && view->offset == 0))) {
    return op->emitOpError()
           << "requires integer `" << name << "` to be "
           << (allowZeroConstant ? "non-negative" : "positive");
  }
  return success();
}

inline LogicalResult verifyElementwiseBinary(Operation *op, Value lhs,
                                             Value rhs, Value result) {
  auto lhsType = cast<RankedTensorType>(lhs.getType());
  auto rhsType = cast<RankedTensorType>(rhs.getType());
  auto resultType = cast<RankedTensorType>(result.getType());
  if (lhsType.getRank() != 4 || rhsType.getRank() != 4 ||
      resultType.getRank() != 4) {
    return op->emitOpError("requires rank-4 tensors in the raw GGML layout");
  }
  if (!haveSameShapeAndElementType(lhsType, resultType)) {
    return op->emitOpError(
        "requires the result shape and element type to match the lhs");
  }
  if (rhsType.getElementType() != lhsType.getElementType()) {
    return op->emitOpError("requires lhs and rhs element types to match");
  }
  if (!lhsType.getElementType().isIntOrFloat()) {
    return op->emitOpError("requires builtin integer or floating-point data");
  }

  for (int64_t dimension = 0; dimension < 4; ++dimension) {
    const int64_t lhsExtent = lhsType.getDimSize(dimension);
    const int64_t rhsExtent = rhsType.getDimSize(dimension);
    if (ShapedType::isDynamic(lhsExtent) || ShapedType::isDynamic(rhsExtent)) {
      continue;
    }
    if (rhsExtent <= 0 || lhsExtent % rhsExtent != 0) {
      return op->emitOpError() << "requires rhs dimension " << dimension
                               << " to divide the corresponding lhs dimension";
    }
  }
  return success();
}

inline LogicalResult verifyElementwiseUnary(Operation *op, Value input,
                                            Value result) {
  auto inputType = cast<RankedTensorType>(input.getType());
  auto resultType = cast<RankedTensorType>(result.getType());
  if (inputType.getRank() != 4 || resultType.getRank() != 4) {
    return op->emitOpError("requires rank-4 tensors in the raw GGML layout");
  }
  if (!haveSameShapeAndElementType(inputType, resultType)) {
    return op->emitOpError(
        "requires matching input and result shapes and element types");
  }
  if (!inputType.getElementType().isIntOrFloat()) {
    return op->emitOpError("requires builtin integer or floating-point data");
  }
  return success();
}

/// The elements a layout holds.
///
/// Succeeds with a null attribute when the product has no affine form -- two
/// varying extents multiplied -- which a caller must not read as a mismatch.
/// Fails when the product overflows, which is a graph no target can run.
inline FailureOr<Attribute> countElements(LayoutAttr layout, std::string &why,
                                          Operation *op = nullptr) {
  shape_expr::Arithmetic arithmetic;
  declareModuleProducts(op, arithmetic);
  shape_expr::Expr total = shape_expr::constant(1);
  for (Attribute extent : layout.getExtents()) {
    std::optional<shape_expr::Expr> view = getShapeExprView(extent);
    if (!view)
      return Attribute();
    total = arithmetic.multiply(total, *view);
  }
  if (arithmetic.hasFailure()) {
    // The reason is owned by `arithmetic`, which is about to go.
    why = arithmetic.getFailure().str();
    if (arithmetic.getFailureKind() ==
        shape_expr::Arithmetic::FailureKind::Overflow)
      return failure();
    return Attribute();
  }
  return shape_expr::toAttribute(layout.getContext(), total);
}

} // namespace mlir::ggml::shared

/// The rule most operations follow: the result is one operand's shape.
#define GGML_SAME_SHAPE_AS(Op, slot)                                           \
  LogicalResult Op::deriveResultShape(ShapeDerivation &derivation) {           \
    return ::mlir::ggml::shared::sameShapeAs(*this, derivation, slot);         \
  }

#endif // HIP_DIALECT_GGML_IR_OPS_OPSSHARED_H
