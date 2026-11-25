/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlShapeArithmetic.h - Affine extent arithmetic -------*- C++ -*-===//
//
// The arithmetic shape derivation needs. Expressions are kept as values rather
// than attributes so that a rule can add and scale them without interning every
// intermediate, and are converted once at the end.
//
// Failure is sticky rather than returned. A shape rule is a short sequence of
// operations that either all make sense or do not, so threading a result
// through each step would obscure the rule without catching anything the caller
// does not already check once at the end.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_DIALECT_GGML_IR_GGMLSHAPEARITHMETIC_H
#define HIP_DIALECT_GGML_IR_GGMLSHAPEARITHMETIC_H

#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>
#include <string>

namespace mlir {
namespace ggml {
namespace shape_expr {

/// An extent as a function of the request: a sum of scaled symbols plus a
/// constant. Terms carry nonzero coefficients, name each symbol once, and stay
/// sorted, so two expressions compare equal exactly when they denote the same
/// function.
using Expr = ShapeExprView;

inline Expr constant(int64_t value) { return Expr{{}, value}; }

inline Expr symbol(llvm::StringRef name) { return Expr{{{name, 1}}, 0}; }

inline bool isConstant(const Expr &expression) {
  return expression.terms.empty();
}

/// Affine arithmetic that remembers the first thing it could not do.
///
/// Non-affine and unresolved extents stop derivation without proving a shape
/// mismatch. Overflow is separate so verifiers can reject it without depending
/// on diagnostic wording. The first failure and its message persist until
/// reset.
class Arithmetic {
public:
  enum class FailureKind { None, NonAffine, Unresolved, Overflow };

  bool hasFailure() const { return failureKind != FailureKind::None; }
  FailureKind getFailureKind() const { return failureKind; }
  llvm::StringRef getFailure() const { return failure; }
  void clearFailure() {
    failure.clear();
    failureKind = FailureKind::None;
  }

  /// Records that the graph does not admit an affine extent here.
  Expr nonAffine(const llvm::Twine &reason) {
    return record(FailureKind::NonAffine, reason);
  }

  /// Records that these rules could not decide the extent.
  Expr unresolved(const llvm::Twine &reason) {
    return record(FailureKind::Unresolved, reason);
  }

  /// Records that checked arithmetic exceeded the signed extent range.
  Expr overflow(const llvm::Twine &reason) {
    return record(FailureKind::Overflow, reason);
  }

  Expr add(const Expr &left, const Expr &right) {
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 4> terms(
        left.terms.begin(), left.terms.end());
    terms.append(right.terms.begin(), right.terms.end());
    int64_t offset = 0;
    if (llvm::AddOverflow(left.offset, right.offset, offset))
      return overflow("extent sum overflows int64");
    return normalize(terms, offset);
  }

  Expr scale(const Expr &expression, int64_t factor) {
    if (factor == 0)
      return constant(0);
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 4> terms;
    for (auto [name, coefficient] : expression.terms) {
      int64_t scaled = 0;
      if (llvm::MulOverflow(coefficient, factor, scaled))
        return overflow("extent product overflows int64");
      terms.emplace_back(name, scaled);
    }
    int64_t offset = 0;
    if (llvm::MulOverflow(expression.offset, factor, offset))
      return overflow("extent product overflows int64");
    return normalize(terms, offset);
  }

  /// Declares `product` the product of symbols `a` and `b`: an image's
  /// blocks are its block columns times its block rows. The graph may then
  /// multiply those two, and divide the product by either.
  void declareProduct(llvm::StringRef product, llvm::StringRef a,
                      llvm::StringRef b) {
    products.push_back({product, a, b});
  }
  bool hasProducts() const { return !products.empty(); }

  /// Multiplies two extents. A product of two request-dependent extents is not
  /// affine, and saying so is the point: approximating it would put a wrong
  /// shape in the artifact - unless it is a declared product of one scaled
  /// symbol by another.
  Expr multiply(const Expr &left, const Expr &right) {
    if (isConstant(left))
      return scale(right, left.offset);
    if (isConstant(right))
      return scale(left, right.offset);
    if (std::optional<Expr> product = symbolProduct(left, right))
      return *product;
    // Sums multiply term by term, each pair of symbols a declared product:
    // (columns + 1) x rows is blocks + rows.
    if (!products.empty()) {
      Expr result = normalize({}, 0);
      const auto parts = [&](const Expr &e) {
        llvm::SmallVector<Expr, 4> split;
        for (auto [name, coefficient] : e.terms)
          split.push_back(normalize({{name, coefficient}}, 0));
        if (e.offset != 0)
          split.push_back(normalize({}, e.offset));
        return split;
      };
      for (const Expr &l : parts(left))
        for (const Expr &r : parts(right)) {
          std::optional<Expr> term;
          if (isConstant(l))
            term = scale(r, l.offset);
          else if (isConstant(r))
            term = scale(l, r.offset);
          else
            term = symbolProduct(l, r);
          if (!term)
            return nonAffine(
                "product of two request-dependent extents is not affine");
          result = add(result, *term);
        }
      return result;
    }
    return nonAffine("product of two request-dependent extents is not affine");
  }

  /// Two scaled symbols whose product is declared: that product, scaled.
  std::optional<Expr> symbolProduct(const Expr &left, const Expr &right) {
    const auto l = left.getSingleTerm(), r = right.getSingleTerm();
    if (!l || !r || left.offset != 0 || right.offset != 0)
      return std::nullopt;
    std::optional<llvm::StringRef> product = productOf(l->first, r->first);
    if (!product)
      product = regrouped(l->first, r->first);
    if (!product)
      product = regrouped(r->first, l->first);
    if (!product)
      return std::nullopt;
    int64_t coefficient = 0;
    if (llvm::MulOverflow(l->second, r->second, coefficient)) {
      overflow("extent product overflows int64");
      return std::nullopt;
    }
    return normalize({{*product, coefficient}}, 0);
  }

  /// `numerator` over `denominator` when it is exact: by a constant, by the
  /// same scaled symbol, or a declared product by one of its factors.
  std::optional<Expr> quotient(const Expr &numerator, const Expr &denominator) {
    if (isConstant(denominator)) {
      if (denominator.offset == 0)
        return std::nullopt;
      if (divisionOverflows(numerator.offset, denominator.offset)) {
        overflow("extent quotient overflows int64");
        return std::nullopt;
      }
      for (auto [name, coefficient] : numerator.terms) {
        if (divisionOverflows(coefficient, denominator.offset)) {
          overflow("extent quotient overflows int64");
          return std::nullopt;
        }
        if (coefficient % denominator.offset != 0)
          return std::nullopt;
      }
      if (numerator.offset % denominator.offset != 0)
        return std::nullopt;
      return divide(numerator, denominator.offset);
    }
    const auto n = numerator.getSingleTerm(), d = denominator.getSingleTerm();
    if (!n || !d || numerator.offset != 0 || denominator.offset != 0 ||
        d->second == 0)
      return std::nullopt;
    if (divisionOverflows(n->second, d->second)) {
      overflow("extent quotient overflows int64");
      return std::nullopt;
    }
    if (n->second % d->second != 0)
      return std::nullopt;
    const int64_t coefficient = n->second / d->second;
    if (n->first == d->first)
      return constant(coefficient);
    for (const Product &p : products) {
      if (p.product != n->first)
        continue;
      if (p.a == d->first)
        return normalize({{p.b, coefficient}}, 0);
      if (p.b == d->first)
        return normalize({{p.a, coefficient}}, 0);
    }
    // Two declared products of a factor alike (the cells times the step's
    // tokens over the cells times a sequence's): the others' quotient.
    for (const Product &top : products) {
      if (top.product != n->first)
        continue;
      for (const Product &bottom : products) {
        if (bottom.product != d->first)
          continue;
        std::optional<llvm::StringRef> over, under;
        if (top.a == bottom.a)
          over = top.b, under = bottom.b;
        else if (top.a == bottom.b)
          over = top.b, under = bottom.a;
        else if (top.b == bottom.a)
          over = top.a, under = bottom.b;
        else if (top.b == bottom.b)
          over = top.a, under = bottom.a;
        if (!over)
          continue;
        if (std::optional<Expr> rest =
                quotient(normalize({{*over, coefficient}}, 0),
                         normalize({{*under, 1}}, 0)))
          return rest;
      }
    }
    return std::nullopt;
  }

  /// `whole` (a declared product a x b) times `c`, grouped as a declared
  /// product of a factor and another declared product (a x (b x c): the KV
  /// cells times a sequence's tokens times the sequences are the cells times
  /// the step's tokens).
  std::optional<llvm::StringRef> regrouped(llvm::StringRef whole,
                                           llvm::StringRef c) const {
    for (const Product &p : products) {
      if (p.product != whole)
        continue;
      if (std::optional<llvm::StringRef> bc = productOf(p.b, c))
        if (std::optional<llvm::StringRef> all = productOf(p.a, *bc))
          return all;
      if (std::optional<llvm::StringRef> ac = productOf(p.a, c))
        if (std::optional<llvm::StringRef> all = productOf(p.b, *ac))
          return all;
    }
    return std::nullopt;
  }

  /// The declared product of `a` and `b`, in either order.
  std::optional<llvm::StringRef> productOf(llvm::StringRef a,
                                           llvm::StringRef b) const {
    for (const Product &p : products)
      if ((p.a == a && p.b == b) || (p.a == b && p.b == a))
        return p.product;
    return std::nullopt;
  }

  /// Divides an extent exactly. GGML's integer division would round, so a
  /// remainder means the result is not this expression scaled down and the
  /// derivation must stop rather than guess.
  Expr divide(const Expr &expression, int64_t divisor) {
    if (divisor == 0)
      return nonAffine("division by zero");
    if (divisionOverflows(expression.offset, divisor))
      return overflow("extent quotient overflows int64");
    if (expression.offset % divisor != 0)
      return nonAffine("constant term is not divisible by " +
                       llvm::Twine(divisor));
    llvm::SmallVector<std::pair<llvm::StringRef, int64_t>, 4> terms;
    for (auto [name, coefficient] : expression.terms) {
      if (divisionOverflows(coefficient, divisor))
        return overflow("extent quotient overflows int64");
      if (coefficient % divisor != 0) {
        return nonAffine("coefficient of '" + name + "' is not divisible by " +
                         llvm::Twine(divisor));
      }
      terms.emplace_back(name, coefficient / divisor);
    }
    return normalize(terms, expression.offset / divisor);
  }

  Expr product(llvm::ArrayRef<Expr> expressions) {
    Expr result = constant(1);
    for (const Expr &expression : expressions)
      result = multiply(result, expression);
    return result;
  }

  /// The value `expression` takes when each symbol takes the value the capture
  /// observed. An unknown symbol is a derivation bug, not a graph property.
  int64_t
  evaluate(const Expr &expression,
           llvm::function_ref<std::optional<int64_t>(llvm::StringRef)> lookup) {
    int64_t total = expression.offset;
    for (auto [name, coefficient] : expression.terms) {
      std::optional<int64_t> value = lookup(name);
      if (!value) {
        unresolved("no captured value for symbol '" + name + "'");
        return 0;
      }
      int64_t scaled = 0;
      if (llvm::MulOverflow(coefficient, *value, scaled) ||
          llvm::AddOverflow(total, scaled, total)) {
        overflow("evaluation overflows int64");
        return 0;
      }
    }
    return total;
  }

  /// Combine coefficients with checked addition, drop canceled terms and sort.
  Expr normalize(llvm::ArrayRef<std::pair<llvm::StringRef, int64_t>> terms,
                 int64_t offset) {
    llvm::MapVector<llvm::StringRef, int64_t> combined;
    for (auto [name, coefficient] : terms) {
      int64_t &sum = combined[name];
      if (llvm::AddOverflow(sum, coefficient, sum))
        return overflow("extent coefficient sum overflows int64");
    }
    Expr result;
    result.offset = offset;
    for (auto [name, coefficient] : combined) {
      if (coefficient != 0)
        result.terms.emplace_back(name, coefficient);
    }
    llvm::sort(result.terms, [](const auto &lhs, const auto &rhs) {
      return lhs.first < rhs.first;
    });
    return result;
  }

private:
  struct Product {
    llvm::StringRef product, a, b;
  };

  static bool divisionOverflows(int64_t numerator, int64_t denominator) {
    // Both division and remainder overflow for this pair in C++.
    return numerator == std::numeric_limits<int64_t>::min() &&
           denominator == -1;
  }

  Expr record(FailureKind kind, const llvm::Twine &reason) {
    if (!hasFailure()) {
      failureKind = kind;
      failure = reason.str();
    }
    return constant(0);
  }

  FailureKind failureKind = FailureKind::None;
  std::string failure;
  llvm::SmallVector<Product, 1> products;
};

/// The attribute spelling of `expression`: an integer when it does not vary,
/// and `#ggml.affine` when it does.
inline Attribute toAttribute(MLIRContext *context, const Expr &expression) {
  return makeShapeExpr(context, expression.terms, expression.offset);
}

} // namespace shape_expr
} // namespace ggml
} // namespace mlir

#endif // HIP_DIALECT_GGML_IR_GGMLSHAPEARITHMETIC_H
