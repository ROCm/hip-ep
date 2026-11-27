/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_DIALECT_GGML_IR_GGMLSHAPEEXPR_H
#define HIP_DIALECT_GGML_IR_GGMLSHAPEEXPR_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LLVM.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/LogicalResult.h"

#include <optional>
#include <utility>

// The interface and the attributes implementing it are declared with the
// dialect, so that one printer and parser serve every ggml attribute.
#include "hip/Dialect/Ggml/IR/GgmlDialect.h"

namespace mlir::ggml {

/// Maps a symbol name to its value for one request, or nullopt when the symbol
/// is unknown.
using SymbolLookup =
    llvm::function_ref<std::optional<int64_t>(llvm::StringRef)>;

/// A shape expression is either a constant, spelled as an `IntegerAttr`, or one
/// of the dialect's expression attributes. Reading through these functions
/// rather than casting keeps a consumer working when another expression form is
/// added. Integer constants must be signless (excluding i1) or index values
/// representable as signed int64.
bool isShapeExpr(Attribute attribute);

/// Evaluates `attribute` under `lookup`. Fails when the attribute is not a
/// shape expression, when `lookup` does not know a symbol it names, or when the
/// arithmetic overflows a signed 64-bit integer.
FailureOr<int64_t> evaluateShapeExpr(Attribute attribute, SymbolLookup lookup);

/// Appends the symbols `attribute` names, without duplicates. A constant
/// appends nothing; an attribute that is not a shape expression is ignored, so
/// callers that care should check `isShapeExpr` first.
void getShapeExprSymbols(Attribute attribute,
                         SmallVectorImpl<llvm::StringRef> &symbols);

/// Returns the value `attribute` always takes, or nullopt when it varies or is
/// not a shape expression.
std::optional<int64_t> getShapeExprConstant(Attribute attribute);

/// Builds the canonical spelling of `offset` plus `terms`: an `IntegerAttr`
/// when no term survives, and an `AffineAttr` otherwise. Terms with a zero
/// coefficient are dropped and repeated symbols are summed, so callers may pass
/// whatever their arithmetic produced. Returns null if combining coefficients
/// in input order overflows int64.
Attribute
makeShapeExpr(MLIRContext *context,
              llvm::ArrayRef<std::pair<llvm::StringRef, int64_t>> terms,
              int64_t offset);

/// One shape expression read back in normalized form, whatever spelled it.
///
/// Terms carry nonzero coefficients, name each symbol once, and are sorted, so
/// two views compare equal exactly when they denote the same function. Reading
/// through a view rather than matching an attribute keeps a consumer working
/// across every spelling the compiler has emitted and any form added later.
struct ShapeExprView {
  llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 2> terms;
  int64_t offset = 0;

  bool isConstant() const { return terms.empty(); }

  /// Returns the one scaled symbol this expression names, or nullopt when it is
  /// constant or names several.
  std::optional<std::pair<llvm::StringRef, int64_t>> getSingleTerm() const {
    if (terms.size() != 1)
      return std::nullopt;
    return terms.front();
  }

  /// True when both expressions vary with the request identically, whatever
  /// constants they add.
  bool hasSameTerms(const ShapeExprView &other) const {
    return terms == other.terms;
  }
};

/// Normalizes `attribute` into a view, or returns nullopt when it does not
/// spell a shape expression. Accepts an integer constant, the dialect's
/// expression attributes, and the dictionary encodings emitted before those
/// existed.
std::optional<ShapeExprView> getShapeExprView(Attribute attribute);

/// Whether two recognized expressions denote the same affine function,
/// regardless of integer width or legacy/current attribute spelling. An
/// invalid expression never compares equal, including to itself.
bool haveEquivalentShapeExprs(Attribute lhs, Attribute rhs);

} // namespace mlir::ggml

#endif // HIP_DIALECT_GGML_IR_GGMLSHAPEEXPR_H
