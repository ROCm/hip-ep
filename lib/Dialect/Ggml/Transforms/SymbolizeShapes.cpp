/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- SymbolizeShapes.cpp - Recover request-dependent extents ----------===//
//
// GGML records the value a dimension had, not the expression that produced it,
// so a captured graph arrives with one request size already substituted in.
// This pass puts the expression back: it seeds the symbols where they entered,
// at the runtime inputs, and deduces every other extent from what each
// operation does to its operands.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Ggml/Transforms/Passes.h"

#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlOps.h"
#include "hip/Dialect/Ggml/IR/GgmlProvenance.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeArithmetic.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringMap.h"

#include <algorithm>
#include <array>
#include <numeric>
#include <optional>
#include <tuple>

namespace mlir {
namespace ggml {

#define GEN_PASS_DEF_GGMLSYMBOLIZESHAPESPASS
#include "hip/Dialect/Ggml/Transforms/Passes.h.inc"

namespace {

using namespace mlir::ggml::shape_expr;

//===----------------------------------------------------------------------===//
// Reading the captured graph
//===----------------------------------------------------------------------===//

/// Four constants the capture observed, or nullopt when any of them varies.
///
/// The pass runs on a module straight from the capture, where every extent,
/// stride, and offset is the one number that request produced. Finding an
/// expression here means the pass has already run, and nothing to observe.
std::optional<SmallVector<int64_t, kGgmlRank>>
getConstants(ArrayRef<Attribute> expressions) {
  if (expressions.size() != kGgmlRank)
    return std::nullopt;
  SmallVector<int64_t, kGgmlRank> observed;
  for (Attribute entry : expressions) {
    std::optional<int64_t> constant = getShapeExprConstant(entry);
    if (!constant)
      return std::nullopt;
    observed.push_back(*constant);
  }
  return observed;
}

/// The extents the capture observed for `value`, innermost first.
std::optional<SmallVector<int64_t, kGgmlRank>> getObservedExtents(Value value) {
  return getConstants(getGgmlExtents(value));
}

/// The byte strides the capture observed for `value`, in the same order.
std::optional<SmallVector<int64_t, kGgmlRank>> getObservedStrides(Value value) {
  return getConstants(getGgmlStrides(value));
}

//===----------------------------------------------------------------------===//
// Derivation
//===----------------------------------------------------------------------===//

using Shape = SmallVector<Expr, 4>;

/// What has been derived for one value.
struct Derived {
  Shape shape;
  Shape strides;
  Expr storageOffset;
  /// Dimensions whose stride no dense layout fixes (denseStrides).
  unsigned freeStrides = 0;
};

class Symbolizer {
public:
  Symbolizer(func::FuncOp function) : function(function) {}

  LogicalResult run();

private:
  /// Seeds the entry arguments and the file-backed parameters, and learns what
  /// each symbol was worth in the capture.
  LogicalResult seed();

  /// Places `name` at `dimension` of `shape`, an entry argument's, learning
  /// what it was worth there.
  LogicalResult place(BlockArgument argument, Shape &shape,
                      ArrayRef<int64_t> observed, unsigned dimension,
                      StringRef name, int64_t perToken = 1, int64_t bias = 0);

  /// Derives one operation's result from its operands.
  LogicalResult deriveNode(Operation *op);

  /// Walks the alias chain so that each value knows where in its root's storage
  /// it begins and how far that root reaches.
  LogicalResult deriveStorage();
  /// Whether `view`, begun at `start` of a source extent `extent` it steps
  /// along by `stride`, ends inside it at every size.
  bool endsInsideSource(Value view, const Expr &extent, const Expr &start,
                        int64_t stride);

  /// Recomputes every derived extent at the captured symbol values and requires
  /// it to reproduce what was observed.
  LogicalResult verify();

  /// Replaces the observed constants with the derived expressions.
  void writeBack();

  /// Records a value's derivation, computing its layout from its extents unless
  /// the operation reinterprets existing storage.
  LogicalResult record(Value value, Shape shape,
                       std::optional<Shape> strides = std::nullopt);

  /// The strides of an entry argument laid out densely in another order than
  /// its dimensions': dense strides of its extents taken in stride order.
  /// Empty when it is dense in its own order, or no permutation of a dense
  /// layout.
  std::optional<Shape> permutedDenseStrides(BlockArgument argument,
                                            const Shape &shape);

  /// The dense strides of `shape`; `free` gets the dimensions whose stride
  /// is no affine expression and does not matter.
  Shape denseStrides(const Shape &shape, BlockLayout layout,
                     unsigned *free = nullptr);

  /// Recovers the symbols in a reinterpreted shape from the element layout.
  std::optional<Shape> relayout(const Shape &source,
                                ArrayRef<int64_t> observed);

  /// The same for a graph of declared products (an image's), where a
  /// boundary may also fall inside a source extent - splitting an image's
  /// columns in two, or its blocks into columns and rows: each result
  /// boundary is a source boundary, or one times the one factor of the next
  /// source extent it reaches.
  std::optional<Shape> relayoutByBoundaries(const Shape &source,
                                            ArrayRef<int64_t> observed);

  /// The scaled symbols an observed number is a multiple of.
  SmallVector<Expr, 2> multiplesOf(int64_t observed);

  /// Recovers the symbols in an observed walk from the strides it steps by;
  /// `layout` is the walk's storage (a quantized cache's blocks).
  std::optional<Walk> recoverWalk(const Shape &sourceExtents,
                                  const Shape &sourceStrides,
                                  ArrayRef<int64_t> extents,
                                  ArrayRef<int64_t> strides,
                                  BlockLayout layout);

  /// Where a view begins, as an address that may carry symbols.
  std::optional<Expr> viewOrigin(Value view, int64_t byteOffset);

  /// Which part of a composed extent `observed` names, when it names one.
  std::optional<Expr> namedPart(const Expr &composed, int64_t observed);

  /// Collects what each operation requires of its operands, so a value whose
  /// extents nothing derives can take them from what consumes it.
  void collectConstraints();

  /// The extent a consumer's precondition requires of one of `value`'s
  /// dimensions. A consumer that keeps its operand's extents passes on whatever
  /// is required of its own result, so the precondition need not sit on the
  /// first use.
  std::optional<Expr> requiredExtent(Value value, unsigned dimension,
                                     unsigned depth = 0) const;

  std::optional<int64_t> lookupSymbol(StringRef name) const {
    auto entry = captured.find(name);
    if (entry == captured.end())
      return std::nullopt;
    return entry->second;
  }

  int64_t at(const Expr &expression) {
    return arithmetic.evaluate(
        expression, [&](StringRef name) { return lookupSymbol(name); });
  }

  InFlightDiagnostic reportOn(Value value);

  /// Where a declared symbol enters, by the value's GGML name: `scale` times
  /// the symbol, plus `bias`, at `dimension`, or, at kOffsetDimension, where
  /// the view begins.
  struct DeclaredSeed {
    StringRef input;
    unsigned dimension = 0;
    int64_t scale = 1;
    StringRef name;
    int64_t bias = 0;
  };
  static constexpr unsigned kOffsetDimension = 4;

  func::FuncOp function;
  Arithmetic arithmetic;
  SmallVector<DeclaredSeed> declaredSeeds;
  /// Declared products: name, factor, factor.
  SmallVector<std::array<StringRef, 3>> declaredProducts;
  llvm::StringMap<int64_t> captured;
  /// The request dimensions the module says it has, and whether it said.
  llvm::StringMap<bool> declared;
  bool declaresSymbols = false;
  DenseMap<Value, Derived> derived;
  /// For each value a consumer constrains, which value and dimension each of
  /// its four extents is required to equal.
  DenseMap<Value,
           std::array<std::optional<std::pair<Value, unsigned>>, kGgmlRank>>
      requiredExtents;

  // Inferring (`ggml.infer_symbols` on the module), every declared symbol is
  // a guess: a size of its own. Where an operation requires two extents
  // equal (its `constrainOperands`, a reshape's elements), two guesses are
  // made one, or a guess the number the other side leaves it; what cannot be
  // derived is fixed at what the capture saw. What is learnt is written to
  // `ggml.inferred` rather than the layouts.
  bool inferring = false;
  /// The guess each guess was made one with; the number a guess was fixed
  /// at, by the guess its class comes to.
  llvm::StringMap<StringRef> parentGuess;
  llvm::StringMap<int64_t> fixedValue;
  /// How many things have been learnt.
  unsigned inferredCount = 0;

  StringRef rootOf(StringRef name) const;
  /// `expression` in what has been learnt: each guess its class's, or its
  /// number.
  Expr resolve(const Expr &expression) const;
  void resolve(Derived &description) const;
  Shape resolve(ArrayRef<Expr> shape) const;
  /// Learns that `left` is `right`.
  void equate(const Expr &left, const Expr &right);
  /// The products of the two equal, where both are affine.
  void equateProducts(ArrayRef<Expr> left, ArrayRef<Expr> right);
  /// Fixes the guesses `expression` holds at what the capture saw.
  void fix(const Expr &expression);
  void writeInferred();
};

/// The GGML slot each SSA operand of `op` fills.
///
/// An operation with optional operands carries only the ones it was given, so
/// a slot is the declared operand group rather than the position: a soft max
/// without a mask has its sinks at SSA operand one and at GGML slot two. The
/// segment sizes say which group each operand belongs to. Where there are none
/// every operand is required, or the one optional is last, and the two orders
/// coincide.
std::optional<SmallVector<int64_t>> operandSlots(Operation *op) {
  SmallVector<int64_t> slots;
  if (op->hasTrait<OpTrait::AttrSizedOperandSegments>()) {
    // The trait's own verifier requires these, and requires them to account
    // for every operand, so they are read rather than checked again.
    std::optional<Attribute> segments =
        op->getInherentAttr("operandSegmentSizes");
    auto sizes =
        segments ? dyn_cast_or_null<DenseI32ArrayAttr>(*segments) : nullptr;
    if (!sizes)
      return std::nullopt;
    for (auto [group, count] : llvm::enumerate(sizes.asArrayRef()))
      slots.append(count, static_cast<int64_t>(group));
    return slots;
  }
  for (unsigned index = 0; index < op->getNumOperands(); ++index)
    slots.push_back(index);
  return slots;
}

InFlightDiagnostic Symbolizer::reportOn(Value value) {
  if (Operation *producer = value.getDefiningOp())
    return producer->emitOpError();
  return function.emitOpError()
         << "entry argument " << cast<BlockArgument>(value).getArgNumber()
         << ": ";
}

Shape Symbolizer::denseStrides(const Shape &shape, BlockLayout layout,
                               unsigned *freeDimensions) {
  Shape strides;
  strides.push_back(constant(layout.typeSize));
  strides.push_back(arithmetic.scale(
      arithmetic.divide(shape[0], layout.blockElements), layout.typeSize));
  bool free = false;
  for (unsigned dimension = 2; dimension < kGgmlRank; ++dimension) {
    // Dimensions of one element step nowhere, so their strides say nothing
    // (GGML's own contiguity ignores them). Past two request-dependent
    // extents - a mask as wide as the session's cache and as tall as the
    // step - the dense stride is no affine expression, and such dimensions
    // keep the last one.
    const bool steps = llvm::any_of(
        llvm::drop_begin(shape, dimension), [](const Expr &extent) {
          return !isConstant(extent) || extent.offset != 1;
        });
    // (A declared product of the two is one: an image's blocks.)
    if (!steps && !free && !isConstant(strides[dimension - 1]) &&
        !isConstant(shape[dimension - 1])) {
      if (std::optional<Expr> product = arithmetic.symbolProduct(
              strides[dimension - 1], shape[dimension - 1])) {
        strides.push_back(*product);
        continue;
      }
    }
    if (!steps && !isConstant(strides[dimension - 1]) &&
        (free || !isConstant(shape[dimension - 1]))) {
      free = true;
      if (freeDimensions)
        *freeDimensions |= 1u << dimension;
      strides.push_back(strides[dimension - 1]);
      continue;
    }
    strides.push_back(
        arithmetic.multiply(strides[dimension - 1], shape[dimension - 1]));
  }
  return strides;
}

std::optional<Shape> Symbolizer::permutedDenseStrides(BlockArgument argument,
                                                      const Shape &shape) {
  std::optional<BlockLayout> layout = getStorageLayout(argument);
  std::optional<SmallVector<int64_t, kGgmlRank>> observed =
      getObservedStrides(argument);
  if (!layout || layout->blockElements != 1 || !observed)
    return std::nullopt;
  std::array<unsigned, kGgmlRank> order;
  std::iota(order.begin(), order.end(), 0u);
  std::stable_sort(order.begin(), order.end(), [&](unsigned a, unsigned b) {
    return (*observed)[a] < (*observed)[b];
  });
  if (llvm::is_sorted(order))
    return std::nullopt;
  Shape permuted;
  for (unsigned dimension : order)
    permuted.push_back(shape[dimension]);
  arithmetic.clearFailure();
  const Shape dense = denseStrides(permuted, *layout);
  Shape strides(kGgmlRank, constant(0));
  for (auto [position, dimension] : llvm::enumerate(order))
    strides[dimension] = dense[position];
  // A dimension of one element steps nowhere: its stride is anything.
  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension)
    if (at(shape[dimension]) != 1 &&
        at(strides[dimension]) != (*observed)[dimension])
      arithmetic.unresolved("not a permuted dense layout");
  if (arithmetic.hasFailure()) {
    arithmetic.clearFailure();
    return std::nullopt;
  }
  return strides;
}

LogicalResult Symbolizer::record(Value value, Shape shape,
                                 std::optional<Shape> strides) {
  std::optional<BlockLayout> layout = getStorageLayout(value);
  if (!layout)
    return reportOn(value) << "has no known GGML block layout";

  arithmetic.clearFailure();
  unsigned free = 0;
  Shape resolved = strides ? *strides : denseStrides(shape, *layout, &free);
  if (arithmetic.hasFailure() && inferring) {
    // (Inferring, a layout no affine expression holds is the one the capture
    // saw: strides say nothing of which extents are one size.)
    arithmetic.clearFailure();
    resolved.clear();
    free = 0;
    if (std::optional<SmallVector<int64_t, kGgmlRank>> observed =
            getObservedStrides(value))
      for (int64_t stride : *observed)
        resolved.push_back(constant(stride));
  }
  if (arithmetic.hasFailure()) {
    return reportOn(value) << "cannot express its layout: "
                           << arithmetic.getFailure();
  }

  derived[value] =
      Derived{std::move(shape), std::move(resolved), constant(0), free};
  return success();
}

std::optional<Shape> Symbolizer::relayout(const Shape &source,
                                          ArrayRef<int64_t> observed) {
  if (arithmetic.hasProducts())
    return relayoutByBoundaries(source, observed);
  // RESHAPE and a zero-offset VIEW keep every element at the same linear index
  // and only redraw the dimension boundaries, so a source extent survives in
  // the result exactly when both start at the same cumulative element count and
  // span the same number of elements.
  SmallVector<int64_t, 4> sourceOffsets;
  int64_t running = 1;
  for (const Expr &extent : source) {
    sourceOffsets.push_back(running);
    running *= at(extent);
  }
  if (arithmetic.hasFailure())
    return std::nullopt;

  Shape result;
  /// The source extent each result extent keeps, when it keeps one.
  SmallVector<std::optional<unsigned>, kGgmlRank> kept;
  running = 1;
  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
    const int64_t extent = observed[dimension];
    SmallVector<const Expr *, 2> matches;
    kept.push_back(std::nullopt);
    for (auto [index, candidate] : llvm::enumerate(source)) {
      if (!isConstant(candidate) && sourceOffsets[index] == running &&
          at(candidate) == extent) {
        matches.push_back(&candidate);
        kept.back() = static_cast<unsigned>(index);
      }
    }
    bool distinct = llvm::any_of(matches, [&](const Expr *candidate) {
      return !(candidate->hasSameTerms(*matches.front()) &&
               candidate->offset == matches.front()->offset);
    });
    if (distinct) {
      arithmetic.unresolved("dimension " + Twine(dimension) +
                            " matches several source extents");
      return std::nullopt;
    }
    if (!matches.empty()) {
      // A capture where the symbol happens to equal one carries no evidence:
      // every constant extent of one matches it just as well.
      if (extent == 1) {
        arithmetic.unresolved(
            "dimension " + Twine(dimension) +
            " is degenerate in this capture; a symbol whose value is one is "
            "indistinguishable from a constant");
        return std::nullopt;
      }
      result.push_back(*matches.front());
    } else {
      result.push_back(constant(extent));
    }
    running *= extent;
  }

  if (inferring) {
    // The same elements: a source extent the result keeps begins after as
    // many elements in both, and the two hold as many in all. Learning so
    // may make the source another, which is then redrawn again.
    const unsigned before = inferredCount;
    for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension)
      if (kept[dimension])
        equateProducts(ArrayRef<Expr>(source).take_front(*kept[dimension]),
                       ArrayRef<Expr>(result).take_front(dimension));
    equateProducts(source, result);
    if (inferredCount != before)
      return relayout(resolve(source), observed);
  }

  const Expr sourceTotal = arithmetic.product(source);
  const Expr resultTotal = arithmetic.product(result);
  const int64_t recovered = at(resultTotal);
  const int64_t expected = at(sourceTotal);
  if (arithmetic.hasFailure())
    return std::nullopt;
  if (!resultTotal.hasSameTerms(sourceTotal)) {
    // No source extent survived whole, but the source may still hold a fixed
    // number of rows per symbol -- a gather of several table rows per token,
    // reshaped to one wide row per token. The symbol is then the one result
    // dimension that spans exactly what it was worth, with the others making
    // up the rest.
    std::optional<std::pair<StringRef, int64_t>> term =
        sourceTotal.getSingleTerm();
    const std::optional<int64_t> symbolValue = term && sourceTotal.offset == 0
                                                   ? lookupSymbol(term->first)
                                                   : std::nullopt;
    std::optional<unsigned> symbolDimension;
    if (symbolValue && *symbolValue > 1) {
      for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
        if (observed[dimension] != *symbolValue)
          continue;
        int64_t rest = 1;
        for (unsigned other = 0; other < kGgmlRank; ++other)
          rest *= other == dimension ? 1 : observed[other];
        if (rest != term->second)
          continue;
        if (symbolDimension) {
          symbolDimension.reset();
          break;
        }
        symbolDimension = dimension;
      }
    }
    if (symbolDimension) {
      for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
        result[dimension] = dimension == *symbolDimension
                                ? shape_expr::symbol(term->first)
                                : constant(observed[dimension]);
      }
      return result;
    }
    // Source extents merged into one (a step's heads and tokens as the rows a
    // rotation multiplies): the result's boundaries are its source's.
    if (std::optional<Shape> merged = relayoutByBoundaries(source, observed))
      return merged;
    arithmetic.clearFailure();
  }
  if (recovered != expected) {
    arithmetic.nonAffine("reinterpreted shape holds " + Twine(recovered) +
                         " elements, source holds " + Twine(expected));
    return std::nullopt;
  }
  return result;
}

SmallVector<Expr, 2> Symbolizer::multiplesOf(int64_t observed) {
  SmallVector<Expr, 2> found;
  for (const auto &entry : captured) {
    const int64_t value = entry.second;
    if (value > 1 && observed % value == 0)
      found.push_back(arithmetic.scale(shape_expr::symbol(entry.first()),
                                       observed / value));
  }
  return found;
}

std::optional<Shape>
Symbolizer::relayoutByBoundaries(const Shape &source,
                                 ArrayRef<int64_t> observed) {
  const auto same = [](const Expr &a, const Expr &b) {
    return a.hasSameTerms(b) && a.offset == b.offset;
  };
  // The source's cumulative element counts, as expressions and numbers.
  SmallVector<Expr, kGgmlRank + 1> bounds{constant(1)};
  for (const Expr &extent : source)
    bounds.push_back(arithmetic.multiply(bounds.back(), extent));
  SmallVector<int64_t, kGgmlRank + 1> boundValues;
  for (const Expr &bound : bounds)
    boundValues.push_back(at(bound));
  if (arithmetic.hasFailure())
    return std::nullopt;

  Shape result;
  Expr below = constant(1);
  int64_t belowValue = 1;
  // Where the dimension below ends inside a source extent (a sequence's
  // tokens in the step's): that extent and the factor of it below.
  std::optional<std::pair<unsigned, Expr>> inside;
  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
    int64_t reach = 0;
    if (llvm::MulOverflow(belowValue, observed[dimension], reach)) {
      arithmetic.overflow("reinterpreted shape overflows int64");
      return std::nullopt;
    }
    std::optional<Expr> boundary;
    std::optional<unsigned> boundaryAt;
    std::optional<std::pair<unsigned, Expr>> endsInside;
    for (unsigned k = 0; k < bounds.size() && !boundary; ++k)
      if (boundValues[k] == reach) {
        boundary = bounds[k];
        boundaryAt = k;
      }
    if (!boundary) {
      // Inside source extent k: the source boundary below it times the one
      // factor of that extent the number names.
      for (unsigned k = 0; k + 1 < bounds.size(); ++k) {
        if (!(boundValues[k] < reach && reach < boundValues[k + 1]) ||
            reach % boundValues[k] != 0)
          continue;
        const int64_t part = reach / boundValues[k];
        SmallVector<Expr, 2> factors;
        SmallVector<Expr, 3> candidates{constant(part)};
        candidates.append(multiplesOf(part));
        for (const Expr &candidate : candidates)
          if (arithmetic.quotient(source[k], candidate) &&
              !llvm::any_of(factors,
                            [&](const Expr &f) { return same(f, candidate); }))
            factors.push_back(candidate);
        if (factors.size() != 1) {
          arithmetic.unresolved("dimension " + Twine(dimension) +
                                " splits extent " + Twine(k) + " in " +
                                Twine(factors.size()) + " ways");
          return std::nullopt;
        }
        boundary = arithmetic.multiply(bounds[k], factors.front());
        endsInside = std::make_pair(k, factors.front());
        break;
      }
    }
    if (!boundary) {
      arithmetic.unresolved("dimension " + Twine(dimension) +
                            " ends at no boundary of its source");
      return std::nullopt;
    }
    std::optional<Expr> extent;
    if (same(*boundary, below)) {
      extent = constant(1);
    } else if (inside && boundaryAt && *boundaryAt > inside->first) {
      // From inside source extent k to a boundary of the source: the rest of
      // extent k, and the whole extents after it.
      extent = arithmetic.quotient(source[inside->first], inside->second);
      for (unsigned k = inside->first + 1; extent && k < *boundaryAt; ++k)
        extent = arithmetic.multiply(*extent, source[k]);
    } else {
      extent = arithmetic.quotient(*boundary, below);
    }
    if (!extent || at(*extent) != observed[dimension]) {
      std::string sizes;
      for (unsigned d = 0; d < kGgmlRank; ++d)
        sizes += (d ? "," : "") + std::to_string(at(source[d])) + "/" +
                 std::to_string(observed[d]);
      arithmetic.unresolved("dimension " + Twine(dimension) +
                            " is no exact part of its source (source/result " +
                            sizes + ")");
      return std::nullopt;
    }
    result.push_back(*extent);
    below = *boundary;
    belowValue = reach;
    inside = endsInside;
  }
  if (!same(below, bounds.back())) {
    arithmetic.nonAffine(
        "reinterpreted shape does not hold its source's elements");
    return std::nullopt;
  }
  return result;
}

/// Which part of a composed extent a number names.
///
/// A dimension whose extent is a sum holds several things end to end -- an
/// operation that hands back its output and its new state in one buffer makes
/// one, and so does a step that keeps the tail of the last one. A view into
/// such a dimension takes one of those things, and starts where one of them
/// starts. Either way the number the capture observed is the size of a part,
/// and which part it is, is the one whose size it is.
///
/// Nothing is named unless exactly one part is the observed size: two parts
/// that happen to be equal at this request are no evidence at all. A dimension
/// that is not a sum names nothing, which leaves the number the number it is.
std::optional<Expr> Symbolizer::namedPart(const Expr &composed,
                                          int64_t observed) {
  SmallVector<Expr, 3> parts;
  for (auto [name, coefficient] : composed.terms)
    parts.push_back(arithmetic.scale(shape_expr::symbol(name), coefficient));
  if (composed.offset != 0)
    parts.push_back(constant(composed.offset));
  if (parts.size() < 2)
    return std::nullopt;

  std::optional<Expr> named;
  unsigned matches = 0;
  for (const Expr &part : parts) {
    if (at(part) != observed)
      continue;
    ++matches;
    named = part;
  }
  if (arithmetic.hasFailure() || matches != 1)
    return std::nullopt;
  return named;
}

std::optional<Walk> Symbolizer::recoverWalk(const Shape &sourceExtents,
                                            const Shape &sourceStrides,
                                            ArrayRef<int64_t> extents,
                                            ArrayRef<int64_t> strides,
                                            BlockLayout layout) {
  // A view walks its source; the stride says which dimension of it. Reading the
  // walk this way rather than by cumulative element count is what lets a view
  // of part of its source be read at all, and it is also the only sound
  // reading when the source is not contiguous -- where a linear index says
  // nothing about which bytes a dimension reaches.
  Walk walk;
  // A view of nothing addresses no bytes, so its strides say nothing about
  // how it would walk them. Its extents are the numbers they are: an empty
  // dimension is empty at every request, not a request size that happened to
  // be zero.
  if (llvm::is_contained(extents, 0)) {
    for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
      walk.extents.push_back(constant(extents[dimension]));
      walk.strides.push_back(constant(strides[dimension]));
    }
    return walk;
  }

  // The source dimension a run of split dimensions is cutting, when it is a
  // sum of parts, and how many elements the dimensions of the run below the
  // current one hold.
  std::optional<unsigned> splitSource;
  int64_t splitBelow = 0;

  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
    const int64_t extent = extents[dimension];
    const int64_t stride = strides[dimension];

    // A dimension that steps by nothing repeats what is under it (Pixtral's
    // merger views one patch as its im2col kernel so): as many times as the
    // caller said, whatever the source.
    if (stride == 0) {
      walk.extents.push_back(constant(extent));
      walk.strides.push_back(constant(0));
      splitSource.reset();
      continue;
    }

    // Every source dimension this one could be stepping along. More than one
    // whenever the source has a degenerate dimension, whose stride repeats its
    // neighbour's.
    SmallVector<unsigned, 2> candidates;
    for (unsigned source = 0; source < kGgmlRank; ++source) {
      if (at(sourceStrides[source]) == stride)
        candidates.push_back(source);
    }
    if (arithmetic.hasFailure())
      return std::nullopt;

    const auto disagree = [&](llvm::function_ref<const Expr &(unsigned)> of) {
      return llvm::any_of(candidates, [&](unsigned source) {
        return !(of(source).hasSameTerms(of(candidates.front())) &&
                 of(source).offset == of(candidates.front()).offset);
      });
    };

    if (candidates.empty()) {
      // A dimension that steps by none of the source's strides was cut inside
      // one of them, which a view does when it splits a source dimension in
      // two. It steps by as far as everything under it reaches.
      if (dimension == 0) {
        arithmetic.unresolved("the innermost dimension steps by " +
                              Twine(stride) +
                              " bytes, which is none of the source's strides");
        return std::nullopt;
      }
      // Whole steps of what is under it: the dimensions below fit inside one
      // step, so this one walks chunks of a source dimension. A step of
      // exactly that span is a split; a larger one is a split that also
      // leaves part of each chunk behind, which a view is equally free to do.
      // Over the innermost dimension of a block-encoded walk that is its
      // blocks, not its elements, times the block's bytes.
      Expr below =
          dimension == 1 && layout.blockElements > 1
              ? arithmetic.scale(
                    arithmetic.divide(walk.extents[0], layout.blockElements),
                    layout.typeSize)
              : arithmetic.multiply(walk.strides[dimension - 1],
                                    walk.extents[dimension - 1]);
      // A dimension of one element steps nowhere, and where what is under it
      // spans no affine expression - a mask's rows, as wide as the cache and
      // as tall as the rows a step returns - its stride is left free, as the
      // mask's own past its two extents are: its neighbour's.
      if (extent == 1 && arithmetic.hasFailure()) {
        arithmetic.clearFailure();
        walk.extents.push_back(constant(1));
        walk.strides.push_back(walk.strides[dimension - 1]);
        splitSource.reset();
        continue;
      }
      const int64_t span = at(below);
      if (arithmetic.hasFailure() || span <= 0 || stride % span != 0) {
        arithmetic.unresolved(
            "dimension " + Twine(dimension) + " steps by " + Twine(stride) +
            " bytes, which is neither one of the source's strides nor whole "
            "steps of the dimensions under it");
        return std::nullopt;
      }
      walk.strides.push_back(
          stride == span ? below : arithmetic.scale(below, stride / span));
      // Splitting a source dimension that holds several parts end to end --
      // a scan's outputs ahead of its state, say -- reaches the end of one
      // part when the run so far holds exactly that part's elements. That
      // part is whatever it is a multiple of, so this dimension is the part
      // over what is under it: one part of exactly this size and no other.
      std::optional<Expr> part;
      if (splitSource && stride == span && splitBelow > 0 &&
          extent <= std::numeric_limits<int64_t>::max() / splitBelow) {
        part = namedPart(sourceExtents[*splitSource], splitBelow * extent);
        if (arithmetic.hasFailure())
          return std::nullopt;
        if (part && (isConstant(*part) || part->offset % splitBelow != 0 ||
                     llvm::any_of(part->terms, [&](const auto &term) {
                       return term.second % splitBelow != 0;
                     })))
          part.reset();
      }
      if (part) {
        walk.extents.push_back(arithmetic.divide(*part, splitBelow));
        splitSource.reset();
      } else {
        walk.extents.push_back(constant(extent));
        if (splitSource && extent > 0 &&
            splitBelow <= std::numeric_limits<int64_t>::max() / extent)
          splitBelow *= extent;
        else
          splitSource.reset();
      }
      continue;
    }
    splitSource.reset();
    if (disagree([&](unsigned source) -> const Expr & {
          return sourceStrides[source];
        })) {
      arithmetic.unresolved("dimension " + Twine(dimension) +
                            " steps by a stride several source dimensions "
                            "spell differently");
      return std::nullopt;
    }
    walk.strides.push_back(sourceStrides[candidates.front()]);

    // Spanning the whole of a source dimension is what makes this dimension
    // that dimension, symbol and all. Spanning less of it is a slice, and how
    // far a slice reaches is a number the caller chose.
    SmallVector<unsigned, 2> whole;
    for (unsigned source : candidates) {
      if (at(sourceExtents[source]) == extent)
        whole.push_back(source);
    }
    if (arithmetic.hasFailure())
      return std::nullopt;
    if (whole.empty()) {
      // A source dimension that is a sum holds several things end to end --
      // an operation that hands back its output and its new state in one
      // buffer makes one -- and a view of it takes one of them. Which one is
      // the one whose size this reaches, and only when exactly one does:
      // two parts that happen to be the same size at this request are no
      // evidence at all. Anything else is a slice, and how far a slice
      // reaches is a number the caller chose.
      std::optional<Expr> part;
      if (!disagree([&](unsigned source) -> const Expr & {
            return sourceExtents[source];
          })) {
        part = namedPart(sourceExtents[candidates.front()], extent);
        if (arithmetic.hasFailure())
          return std::nullopt;
      }
      // A walk on through the dimensions above, which lie end to end, that
      // stops short of their end by less than one step of this one: the
      // whole of them but that tail (Pixtral's rows of blocks, each with a
      // break after it, but the last). Only the outermost dimension of the
      // view may walk so.
      if (!part && llvm::all_of(llvm::drop_begin(extents, dimension + 1),
                                [](int64_t e) { return e == 1; })) {
        const unsigned first = candidates.front();
        Expr reach = sourceExtents[first];
        for (unsigned above = first + 1; above < kGgmlRank; ++above) {
          if (isConstant(sourceExtents[above]) &&
              sourceExtents[above].offset == 1)
            continue;
          Expr step = arithmetic.multiply(sourceStrides[above - 1],
                                          sourceExtents[above - 1]);
          if (arithmetic.hasFailure() ||
              !step.hasSameTerms(sourceStrides[above]) ||
              step.offset != sourceStrides[above].offset) {
            arithmetic.clearFailure();
            break;
          }
          reach = arithmetic.multiply(reach, sourceExtents[above]);
          if (arithmetic.hasFailure()) {
            arithmetic.clearFailure();
            break;
          }
          const int64_t whole = at(reach);
          const int64_t short_ = whole - extent;
          if (!isConstant(reach) && short_ > 0 &&
              short_ < at(sourceExtents[first])) {
            part = reach;
            part->offset -= short_;
            break;
          }
        }
        if (arithmetic.hasFailure())
          return std::nullopt;
      }
      // Inferring, a dimension that runs over several whole source
      // dimensions lying end to end ([128, 8] read as 1024: a head's width
      // times the heads) holds their product: the guesses in them multiply
      // to this number.
      if (!part && inferring) {
        const unsigned first = candidates.front();
        int64_t covered = at(sourceExtents[first]);
        for (unsigned last = first + 1; last < kGgmlRank && covered < extent;
             ++last) {
          if (at(sourceStrides[last]) !=
              at(sourceStrides[last - 1]) * at(sourceExtents[last - 1]))
            break;
          covered *= at(sourceExtents[last]);
          if (covered == extent) {
            const Expr whole = constant(extent);
            equateProducts(
                ArrayRef<Expr>(sourceExtents).slice(first, last - first + 1),
                ArrayRef<Expr>(whole));
            break;
          }
        }
        if (arithmetic.hasFailure())
          return std::nullopt;
      }
      walk.extents.push_back(part ? *part : constant(extent));
      // A slice of a summed dimension that is not itself one part may be the
      // innermost cut of a run that is.
      if (!part && !isConstant(sourceExtents[candidates.front()])) {
        splitSource = candidates.front();
        splitBelow = extent;
      }
      continue;
    }
    candidates = whole;
    if (disagree([&](unsigned source) -> const Expr & {
          return sourceExtents[source];
        })) {
      arithmetic.unresolved("dimension " + Twine(dimension) +
                            " spans several source dimensions that are not "
                            "the same extent");
      return std::nullopt;
    }
    // A capture where the symbol happens to equal one carries no evidence:
    // a constant one spans it just as well.
    if (extent == 1 && !isConstant(sourceExtents[candidates.front()])) {
      arithmetic.unresolved(
          "dimension " + Twine(dimension) +
          " is degenerate in this capture; a symbol whose value is one is "
          "indistinguishable from a constant");
      return std::nullopt;
    }
    walk.extents.push_back(sourceExtents[candidates.front()]);
  }
  return walk;
}

void Symbolizer::collectConstraints() {
  /// Turns one operation's slot-to-slot requirement into a value-to-value one.
  class Recorder final : public ShapeConstraints {
  public:
    Recorder(
        Operation *op, ArrayRef<int64_t> slots,
        DenseMap<Value, std::array<std::optional<std::pair<Value, unsigned>>,
                                   kGgmlRank>> &requirements)
        : op(op), slots(slots), requirements(requirements) {}

    void equate(int64_t slot, unsigned dimension, int64_t otherSlot,
                unsigned otherDimension) override {
      Value constrained = at(slot);
      Value reference = at(otherSlot);
      if (!constrained || !reference || dimension >= kGgmlRank ||
          otherDimension >= kGgmlRank) {
        return;
      }
      // The first requirement wins. Two consumers requiring different things
      // of the same value is a graph GGML would already have refused.
      auto &entry = requirements[constrained][dimension];
      if (!entry)
        entry = std::make_pair(reference, otherDimension);
    }

  private:
    Value at(int64_t slot) const {
      for (auto [index, filled] : llvm::enumerate(slots)) {
        if (filled == slot)
          return op->getOperand(index);
      }
      return {};
    }

    Operation *op;
    ArrayRef<int64_t> slots;
    DenseMap<Value,
             std::array<std::optional<std::pair<Value, unsigned>>, kGgmlRank>>
        &requirements;
  };

  function.walk([&](GgmlShapeRule rule) {
    Operation *op = rule.getOperation();
    std::optional<SmallVector<int64_t>> slots = operandSlots(op);
    if (!slots)
      return;
    Recorder recorder(op, *slots, requiredExtents);
    rule.constrainOperands(recorder);
  });
}

/// Whether `op` gives its result the extents of the operand in slot zero, so
/// that whatever a consumer requires of the result it requires of that
/// operand too.
static bool keepsFirstOperandExtents(Operation *op) {
  return isa<AddOp, MulOp, RmsNormOp>(op);
}

std::optional<Expr> Symbolizer::requiredExtent(Value value, unsigned dimension,
                                               unsigned depth) const {
  if (dimension >= kGgmlRank)
    return std::nullopt;
  auto entry = requiredExtents.find(value);
  if (entry != requiredExtents.end() && entry->second[dimension]) {
    const std::pair<Value, unsigned> &requirement = *entry->second[dimension];
    // A requirement not derived yet may be by a later round; another use may
    // say now.
    auto source = derived.find(requirement.first);
    if (source != derived.end())
      return source->second.shape[requirement.second];
  }
  // A chain of shape-keeping operations is bounded so that a long one is not
  // taken as evidence on its own.
  if (depth >= 4)
    return std::nullopt;
  for (OpOperand &use : value.getUses()) {
    Operation *user = use.getOwner();
    if (use.getOperandNumber() != 0 || user->getNumResults() != 1 ||
        !keepsFirstOperandExtents(user))
      continue;
    if (std::optional<Expr> extent =
            requiredExtent(user->getResult(0), dimension, depth + 1))
      return extent;
  }
  return std::nullopt;
}

LogicalResult Symbolizer::place(BlockArgument argument, Shape &shape,
                                ArrayRef<int64_t> observed, unsigned dimension,
                                StringRef name, int64_t perToken,
                                int64_t bias) {
  const int64_t value = (observed[dimension] - bias) / perToken;
  auto existing = captured.find(name);
  if (existing != captured.end() && existing->second != value) {
    return function.emitOpError()
           << "runtime inputs disagree on '" << name
           << "': " << existing->second << " and " << value << " (argument "
           << argument.getArgNumber() << ")";
  }
  captured[name] = value;
  shape[dimension] = arithmetic.add(
      perToken == 1 ? symbol(name) : arithmetic.scale(symbol(name), perToken),
      constant(bias));
  return success();
}

LogicalResult Symbolizer::seed() {
  Block &entry = function.getBody().front();

  // Runtime inputs are where a request size enters the graph, so they are both
  // the seeds and the only evidence of what each symbol was worth.
  for (BlockArgument argument : entry.getArguments()) {
    std::optional<SmallVector<int64_t, 4>> observed =
        getObservedExtents(argument);
    if (!observed) {
      // Arguments the GGML frontend did not describe, such as an inserted
      // context handle, carry no extents to symbolize.
      continue;
    }

    Shape shape;
    for (int64_t extent : *observed)
      shape.push_back(constant(extent));

    // The symbols the module says enter here.
    if (auto source = function.getArgAttrOfType<StringAttr>(
            argument.getArgNumber(), "ggml.source_name")) {
      for (const DeclaredSeed &declaredSeed : declaredSeeds) {
        if (declaredSeed.input != source.getValue() ||
            declaredSeed.dimension == kOffsetDimension)
          continue;
        if (((*observed)[declaredSeed.dimension] - declaredSeed.bias) %
                declaredSeed.scale !=
            0)
          return function.emitOpError()
                 << "input '" << source.getValue() << "' holds "
                 << (*observed)[declaredSeed.dimension] << " at dimension "
                 << declaredSeed.dimension << ", not a multiple of "
                 << declaredSeed.scale << " " << declaredSeed.name;
        if (failed(place(argument, shape, *observed, declaredSeed.dimension,
                         declaredSeed.name, declaredSeed.scale,
                         declaredSeed.bias)))
          return failure();
      }
    }
    // A copy the scheduler made of a permuted view keeps the view's layout
    // (`ggml_dup_tensor_layout`): dense, in another order.
    std::optional<Shape> strides = permutedDenseStrides(argument, shape);
    if (failed(record(argument, std::move(shape), std::move(strides))))
      return failure();
  }

  // A product is worth its factors' product, whether or not an input carries
  // it, and one seen on its own must agree. A factor no input carries is
  // what the product seen leaves of the other (a sequence's tokens: the
  // step's over its sequences).
  for (const auto &[product, a, b] : declaredProducts) {
    std::optional<int64_t> left = lookupSymbol(a), right = lookupSymbol(b);
    const std::optional<int64_t> whole = lookupSymbol(product);
    for (auto [known, unknown, name] :
         {std::tuple{left, &right, b}, std::tuple{right, &left, a}})
      if (known && !*unknown && whole && *known > 0 && *whole % *known == 0) {
        *unknown = *whole / *known;
        captured[name] = **unknown;
      }
    if (!left || !right)
      return function.emitOpError()
             << "no runtime input carries '" << (left ? b : a)
             << "', a factor of '" << product << "'";
    if (std::optional<int64_t> seen = lookupSymbol(product);
        seen && *seen != *left * *right)
      return function.emitOpError() << "'" << product << "' was " << *seen
                                    << ", its factors make " << *left * *right;
    captured[product] = *left * *right;
  }

  // A graph the request does not reach has nothing to symbolize, and says so
  // by declaring no symbols; every extent in it is already the number it is.
  // A graph that declares one and never places it is the contradiction.
  if (captured.empty() && (!declaresSymbols || !declared.empty())) {
    return function.emitOpError()
           << "no runtime input declares a request-dependent extent; nothing "
              "to symbolize";
  }

  return success();
}

LogicalResult Symbolizer::run() {
  // The module says which dimensions a request varies (`ggml.symbols`), and
  // at which runtime inputs each enters.
  if (auto module = function->getParentOfType<ModuleOp>()) {
    inferring = module->hasAttr("ggml.infer_symbols");
    if (auto symbols = module->getAttrOfType<ArrayAttr>("ggml.symbols")) {
      declaresSymbols = true;
      for (Attribute entry : symbols) {
        auto described = dyn_cast<DictionaryAttr>(entry);
        auto name =
            described ? described.getAs<StringAttr>("name") : StringAttr();
        if (!name || name.getValue().empty())
          return function.emitOpError("a declared symbol has no name");
        declared[name.getValue()] = true;
        if (auto factors = described.getAs<ArrayAttr>("product")) {
          auto a = factors.size() == 2 ? dyn_cast<StringAttr>(factors[0])
                                       : StringAttr();
          auto b = factors.size() == 2 ? dyn_cast<StringAttr>(factors[1])
                                       : StringAttr();
          if (!a || !b)
            return function.emitOpError("symbol '")
                   << name.getValue() << "' is a product of two named symbols";
          arithmetic.declareProduct(name.getValue(), a.getValue(),
                                    b.getValue());
          declaredProducts.push_back(
              {name.getValue(), a.getValue(), b.getValue()});
        }
        // Where it enters: `input` (at `dimension`, `scale` times it), and
        // any more such places in `inputs`.
        SmallVector<DictionaryAttr> places{described};
        if (auto more = described.getAs<ArrayAttr>("inputs"))
          for (Attribute place : more) {
            auto dictionary = dyn_cast<DictionaryAttr>(place);
            if (!dictionary || !dictionary.getAs<StringAttr>("input"))
              return function.emitOpError("symbol '")
                     << name.getValue()
                     << "' lists an input it enters at without a name";
            places.push_back(dictionary);
          }
        for (DictionaryAttr place : places) {
          auto input = place.getAs<StringAttr>("input");
          if (!input)
            continue;
          auto dimension = place.getAs<IntegerAttr>("dimension");
          auto scale = place.getAs<IntegerAttr>("scale");
          auto bias = place.getAs<IntegerAttr>("offset");
          if (!dimension || dimension.getInt() < 0 ||
              dimension.getInt() > kOffsetDimension ||
              (scale && scale.getInt() <= 0))
            return function.emitOpError("symbol '")
                   << name.getValue()
                   << "' enters at a dimension and a positive scale";
          declaredSeeds.push_back({input.getValue(),
                                   static_cast<unsigned>(dimension.getInt()),
                                   scale ? scale.getInt() : 1, name.getValue(),
                                   bias ? bias.getInt() : 0});
        }
      }
    }
  }

  if (failed(seed()))
    return failure();

  // What each operation requires of its operands, before anything is derived:
  // the requirement is structural, and the extent it points at is derived by
  // the time the operation that needs it is reached.
  collectConstraints();

  // Operations are emitted in execution order, so one pass in IR order sees
  // every operand already derived. The parameters `seed` already handled are
  // skipped by being derived; every other GGML operation has a rule or fails.
  //
  // An operation whose rule needs an extent derived later is left for the
  // next pass, quietly, for as long as each pass derives something; the last
  // pass is the one that says what failed.
  bool deferFailures = true;
  WalkResult walked = WalkResult::advance();
  while (true) {
    bool progressed = false;
    bool pending = false;
    std::optional<ScopedDiagnosticHandler> suppressDiagnostics;
    if (deferFailures)
      suppressDiagnostics.emplace(function.getContext(),
                                  [](Diagnostic &) { return success(); });
    walked = function.walk([&](Operation *op) {
      if (!isa_and_nonnull<GgmlDialect>(op->getDialect()))
        return WalkResult::advance();
      if (op->getNumResults() == 1 && derived.count(op->getResult(0)))
        return WalkResult::advance();
      if (deferFailures && llvm::any_of(op->getOperands(), [&](Value operand) {
            return !derived.count(operand);
          })) {
        pending = true;
        return WalkResult::advance();
      }
      if (failed(deriveNode(op))) {
        if (!deferFailures)
          return WalkResult::interrupt();
        pending = true;
        return WalkResult::advance();
      }
      progressed = true;
      return WalkResult::advance();
    });
    if (!pending || walked.wasInterrupted())
      break;
    if (!progressed)
      deferFailures = false;
  }
  if (walked.wasInterrupted())
    return failure();

  if (inferring) {
    writeInferred();
    return success();
  }
  if (failed(deriveStorage()) || failed(verify()))
    return failure();
  writeBack();
  return success();
}

StringRef Symbolizer::rootOf(StringRef name) const {
  for (auto joined = parentGuess.find(name); joined != parentGuess.end();
       joined = parentGuess.find(name))
    name = joined->second;
  return name;
}

Expr Symbolizer::resolve(const Expr &expression) const {
  Arithmetic local;
  SmallVector<std::pair<StringRef, int64_t>, 4> terms;
  int64_t offset = expression.offset;
  for (auto [name, coefficient] : expression.terms) {
    const StringRef root = rootOf(name);
    auto fixed = fixedValue.find(root);
    if (fixed == fixedValue.end()) {
      terms.emplace_back(root, coefficient);
      continue;
    }
    int64_t scaled = 0;
    if (llvm::MulOverflow(coefficient, fixed->second, scaled) ||
        llvm::AddOverflow(offset, scaled, offset))
      return expression;
  }
  Expr resolved = local.normalize(terms, offset);
  return local.hasFailure() ? expression : resolved;
}

Shape Symbolizer::resolve(ArrayRef<Expr> shape) const {
  Shape resolved;
  for (const Expr &extent : shape)
    resolved.push_back(resolve(extent));
  return resolved;
}

void Symbolizer::resolve(Derived &description) const {
  description.shape = resolve(description.shape);
  description.strides = resolve(description.strides);
  description.storageOffset = resolve(description.storageOffset);
}

void Symbolizer::equate(const Expr &left, const Expr &right) {
  // What is learnt holds at the capture's sizes; at other sizes it holds
  // when the guesses keep to it, which the guard checks. Anything but one
  // guess or two alike (a guess and a number, a guess and a guess) says
  // nothing that holds at other sizes, and nothing is learnt.
  Arithmetic local;
  const Expr difference =
      local.add(resolve(left), local.scale(resolve(right), -1));
  const int64_t value = local.evaluate(
      difference, [&](StringRef name) { return lookupSymbol(name); });
  if (local.hasFailure() || value != 0)
    return;
  if (difference.terms.size() == 1) {
    auto [name, coefficient] = difference.terms.front();
    if (difference.offset % coefficient != 0)
      return;
    fixedValue[name] = -difference.offset / coefficient;
    ++inferredCount;
  } else if (difference.terms.size() == 2 && difference.offset == 0 &&
             difference.terms[0].second == -difference.terms[1].second) {
    parentGuess[difference.terms[0].first] = difference.terms[1].first;
    ++inferredCount;
  }
}

void Symbolizer::equateProducts(ArrayRef<Expr> left, ArrayRef<Expr> right) {
  const Shape resolvedLeft = resolve(left), resolvedRight = resolve(right);
  Arithmetic local;
  const Expr l = local.product(resolvedLeft);
  const Expr r = local.product(resolvedRight);
  if (!local.hasFailure()) {
    equate(l, r);
    return;
  }
  // Guesses whose product is a number (a head's width times the heads, the
  // width of a projection) can vary only together and inversely, as no size
  // a request varies does: each is fixed at what the capture saw.
  const auto constant = [](ArrayRef<Expr> side) {
    return llvm::all_of(side, [](const Expr &e) { return isConstant(e); });
  };
  if (constant(resolvedLeft))
    for (const Expr &extent : resolvedRight)
      fix(extent);
  else if (constant(resolvedRight))
    for (const Expr &extent : resolvedLeft)
      fix(extent);
}

void Symbolizer::fix(const Expr &expression) {
  for (auto [name, coefficient] : resolve(expression).terms)
    if (std::optional<int64_t> value = lookupSymbol(name)) {
      fixedValue[name] = *value;
      ++inferredCount;
    }
}

void Symbolizer::writeInferred() {
  Builder builder(function.getContext());
  SmallVector<NamedAttribute> inferredClasses;
  for (const auto &entry : declared) {
    const StringRef root = rootOf(entry.getKey());
    auto fixed = fixedValue.find(root);
    inferredClasses.push_back(builder.getNamedAttr(
        entry.getKey(),
        fixed != fixedValue.end()
            ? Attribute(builder.getI64IntegerAttr(fixed->second))
            : Attribute(builder.getStringAttr(root))));
  }
  if (auto module = function->getParentOfType<ModuleOp>())
    module->setAttr("ggml.inferred",
                    builder.getDictionaryAttr(inferredClasses));
}

LogicalResult Symbolizer::deriveNode(Operation *op) {
  if (op->getNumResults() != 1)
    return op->emitOpError() << "expects exactly one result to derive";
  Value result = op->getResult(0);

  std::optional<SmallVector<int64_t>> slots = operandSlots(op);
  if (!slots)
    return op->emitOpError() << "has no operand segment sizes to read";

  DenseMap<int64_t, const Derived *> operands;
  for (auto [index, operand] : llvm::enumerate(op->getOperands())) {
    auto entry = derived.find(operand);
    if (entry == derived.end()) {
      return op->emitOpError() << "cannot derive its result: operand " << index
                               << " has no derived shape";
    }
    operands[(*slots)[index]] = &entry->second;
  }

  std::optional<SmallVector<int64_t, 4>> observed = getObservedExtents(result);
  std::optional<SmallVector<int64_t, 4>> observedStrides =
      getObservedStrides(result);
  if (!observed || !observedStrides)
    return op->emitOpError() << "requires a rank-4 `ggml.ne` and `ggml.nb`";

  if (inferring) {
    // What the operation requires of its operands, learnt before its result
    // is derived from them.
    class Requirements final : public ShapeConstraints {
    public:
      void equate(int64_t slot, unsigned dimension, int64_t otherSlot,
                  unsigned otherDimension) override {
        pairs.push_back({slot, dimension, otherSlot, otherDimension});
      }
      SmallVector<std::tuple<int64_t, unsigned, int64_t, unsigned>> pairs;
    } requirements;
    for (Value operand : op->getOperands())
      resolve(derived.find(operand)->second);
    if (auto rule = dyn_cast<GgmlShapeRule>(op))
      rule.constrainOperands(requirements);
    for (auto [slot, dimension, otherSlot, otherDimension] :
         requirements.pairs) {
      auto a = operands.find(slot), b = operands.find(otherSlot);
      if (a == operands.end() || b == operands.end() ||
          dimension >= kGgmlRank || otherDimension >= kGgmlRank)
        continue;
      equate(a->second->shape[dimension], b->second->shape[otherDimension]);
    }
    for (Value operand : op->getOperands())
      resolve(derived.find(operand)->second);
  }
  // What cannot be derived, inferring, is what the capture saw, and what it
  // was derived from is fixed there.
  const auto useObservedExtents = [&]() {
    arithmetic.clearFailure();
    for (Value operand : op->getOperands())
      for (const Expr &extent : derived.find(operand)->second.shape)
        fix(extent);
    Shape shape, strides;
    for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
      shape.push_back(constant((*observed)[dimension]));
      strides.push_back(constant((*observedStrides)[dimension]));
    }
    return record(result, std::move(shape), std::move(strides));
  };

  // What the rule may read and what it produces. The pass decides where a
  // request size enters the graph; the operation decides what it does with it.
  class Derivation final : public ShapeDerivation {
  public:
    Derivation(Symbolizer &owner, const DenseMap<int64_t, const Derived *> &by,
               ArrayRef<int64_t> observed, ArrayRef<int64_t> observedStrides,
               BlockLayout layout)
        : owner(owner), operands(by), observed(observed),
          observedStrides(observedStrides), layout(layout) {}

    const Shape &operandShape(int64_t slot) const override {
      static const Shape empty;
      auto entry = operands.find(slot);
      return entry == operands.end() ? empty : entry->second->shape;
    }
    const Shape &operandStrides(int64_t slot) const override {
      static const Shape empty;
      auto entry = operands.find(slot);
      return entry == operands.end() ? empty : entry->second->strides;
    }
    bool hasOperand(int64_t slot) const override {
      return operands.count(slot) != 0;
    }
    ArrayRef<int64_t> observedResultExtents() const override {
      return observed;
    }
    ArrayRef<int64_t> observedResultStrides() const override {
      return observedStrides;
    }
    std::optional<Shape> relayout(const Shape &source,
                                  ArrayRef<int64_t> extents) override {
      return owner.relayout(source, extents);
    }
    std::optional<Walk> recoverWalk(const Shape &sourceExtents,
                                    const Shape &sourceStrides,
                                    ArrayRef<int64_t> extents,
                                    ArrayRef<int64_t> strides) override {
      return owner.recoverWalk(sourceExtents, sourceStrides, extents, strides,
                               layout);
    }
    Arithmetic &arithmetic() override { return owner.arithmetic; }
    void setResultShape(Shape value) override { shape = std::move(value); }
    void setResultStrides(Shape value) override { strides = std::move(value); }

    Shape shape;
    std::optional<Shape> strides;

  private:
    Symbolizer &owner;
    const DenseMap<int64_t, const Derived *> &operands;
    ArrayRef<int64_t> observed;
    ArrayRef<int64_t> observedStrides;
    BlockLayout layout;
  };

  arithmetic.clearFailure();
  Derivation derivation(*this, operands, *observed, *observedStrides,
                        getStorageLayout(result).value_or(BlockLayout{4, 1}));

  // An operation with no rule is refused rather than left at whatever extents
  // the capture observed.
  auto rule = dyn_cast<GgmlShapeRule>(op);
  if (!rule)
    return op->emitOpError() << "has no shape rule";
  if (failed(rule.deriveResultShape(derivation)))
    return inferring ? useObservedExtents() : failure();
  Shape shape = std::move(derivation.shape);
  std::optional<Shape> strides = std::move(derivation.strides);

  // A symbol the module declares at this value (a view a ggml backend's
  // split takes, whose extents its caller chose) is its extent there.
  bool declared = false;
  if (auto source = op->getAttrOfType<StringAttr>("ggml.source_name")) {
    for (const DeclaredSeed &seed : declaredSeeds) {
      if (seed.input != source.getValue() || seed.dimension >= shape.size())
        continue;
      const int64_t value = (*observed)[seed.dimension];
      if (value % seed.scale != 0)
        return op->emitOpError()
               << "holds " << value << " at dimension " << seed.dimension
               << ", not a multiple of " << seed.scale << " " << seed.name;
      auto existing = captured.find(seed.name);
      if (existing == captured.end())
        captured[seed.name] = value / seed.scale;
      else if (existing->second != value / seed.scale)
        return op->emitOpError()
               << "holds " << value << " at dimension " << seed.dimension
               << ", which is not " << seed.scale << " " << seed.name;
      shape[seed.dimension] =
          seed.scale == 1 ? symbol(seed.name)
                          : arithmetic.scale(symbol(seed.name), seed.scale);
      declared = true;
    }
  }
  // Such a view of all its source holds, laid out as densely (a step's
  // queries split into its sequences, each its tokens), is a reshape of it:
  // each stride all that is below it, symbols and all.
  if (declared && strides && isa<ViewOp>(op) && derivation.hasOperand(0)) {
    const Shape &source = derivation.operandShape(0);
    Expr elements = arithmetic.product(shape),
         sourceElements = arithmetic.product(source);
    bool dense = !arithmetic.hasFailure() &&
                 elements.hasSameTerms(sourceElements) &&
                 elements.offset == sourceElements.offset;
    for (unsigned dimension = 1; dense && dimension < strides->size();
         ++dimension)
      dense = (*observedStrides)[dimension] ==
              (*observedStrides)[dimension - 1] * (*observed)[dimension - 1];
    arithmetic.clearFailure();
    for (unsigned dimension = 1; dense && dimension < strides->size();
         ++dimension) {
      (*strides)[dimension] =
          arithmetic.multiply((*strides)[dimension - 1], shape[dimension - 1]);
      dense = !arithmetic.hasFailure();
    }
  }

  if (arithmetic.hasFailure()) {
    if (inferring)
      return useObservedExtents();
    return op->emitOpError()
           << "cannot derive its extents: " << arithmetic.getFailure();
  }
  if (failed(record(result, std::move(shape), std::move(strides))))
    return failure();
  if (inferring)
    resolve(derived.find(result)->second);
  return success();
}

/// Where a view begins, as a byte address that carries whatever symbols its
/// source's strides and extents carry.
///
/// The offset GGML records is a number, but it is the address of a coordinate
/// in the source, and a coordinate into a dimension made of several parts may
/// be the size of one of them -- which is how a step that keeps the tail of
/// the last one begins at a place that moves with the request. So the address
/// is decoded outermost first, each coordinate is asked what it names, and the
/// address is rebuilt from what they said. Where nothing is named this
/// rebuilds the number it started from, which is the answer for every view
/// whose source is not composed.
bool Symbolizer::endsInsideSource(Value view, const Expr &extent,
                                  const Expr &start, int64_t stride) {
  auto own = derived.find(view);
  const std::optional<SmallVector<int64_t, kGgmlRank>> viewExtents =
      getObservedExtents(view);
  const std::optional<SmallVector<int64_t, kGgmlRank>> viewStrides =
      getObservedStrides(view);
  if (own == derived.end() || !viewExtents || !viewStrides)
    return false;
  std::optional<unsigned> along;
  for (unsigned k = 0; k < kGgmlRank; ++k)
    if ((*viewStrides)[k] == stride && ((*viewExtents)[k] != 1 || !along))
      along = k;
  if (!along)
    return false;
  // What is left past the view's end: no symbol taken away, nothing below
  // zero.
  Expr left = arithmetic.add(
      extent,
      arithmetic.scale(arithmetic.add(start, own->second.shape[*along]), -1));
  if (arithmetic.hasFailure()) {
    arithmetic.clearFailure();
    return false;
  }
  return left.offset >= 0 && llvm::all_of(left.terms, [](const auto &term) {
           return term.second >= 0;
         });
}

std::optional<Expr> Symbolizer::viewOrigin(Value view, int64_t byteOffset) {
  // Where its builder said the view begins.
  if (auto op = view.getDefiningOp())
    if (auto source = op->getAttrOfType<StringAttr>("ggml.source_name"))
      for (const DeclaredSeed &seed : declaredSeeds) {
        if (seed.input != source.getValue() ||
            seed.dimension != kOffsetDimension)
          continue;
        if ((byteOffset - seed.bias) % seed.scale != 0) {
          arithmetic.unresolved("a view begins at byte " + Twine(byteOffset) +
                                ", no multiple of " + Twine(seed.scale) + " " +
                                seed.name);
          return std::nullopt;
        }
        const int64_t value = (byteOffset - seed.bias) / seed.scale;
        auto existing = captured.find(seed.name);
        if (existing != captured.end() && existing->second != value) {
          arithmetic.unresolved("views begin at " + Twine(value) + " and " +
                                Twine(existing->second) + " " + seed.name);
          return std::nullopt;
        }
        captured[seed.name] = value;
        return arithmetic.add(arithmetic.scale(symbol(seed.name), seed.scale),
                              constant(seed.bias));
      }
  auto description = derived.find(getStorageSource(view));
  if (description == derived.end())
    return constant(byteOffset);

  const Shape &extents = description->second.shape;
  const Shape &strides = description->second.strides;
  // The view's own extents by the stride each steps by: where one walks a
  // dimension of one request size to its end, the view is its tail (a
  // decision head reads the hidden state of the step's last token). A sum
  // of parts is namedPart's to read: which part's end is reached may be no
  // evidence at all.
  const std::optional<SmallVector<int64_t, kGgmlRank>> viewExtents =
      getObservedExtents(view);
  const std::optional<SmallVector<int64_t, kGgmlRank>> viewStrides =
      getObservedStrides(view);
  const auto tail = [&](unsigned dimension, int64_t stride,
                        int64_t coordinate) -> std::optional<Expr> {
    if (extents[dimension].terms.size() != 1 || !viewExtents || !viewStrides ||
        coordinate <= 0)
      return std::nullopt;
    std::optional<int64_t> spans;
    for (unsigned k = 0; k < kGgmlRank; ++k)
      if ((*viewStrides)[k] == stride && ((*viewExtents)[k] != 1 || !spans))
        spans = (*viewExtents)[k];
    if (!spans || coordinate + *spans != at(extents[dimension]))
      return std::nullopt;
    Expr start = extents[dimension];
    start.offset -= *spans;
    return start;
  };
  Expr origin = constant(0);
  int64_t remaining = byteOffset;
  for (unsigned dimension = kGgmlRank; dimension-- > 0;) {
    if (remaining == 0)
      break;
    const int64_t stride = at(strides[dimension]);
    if (arithmetic.hasFailure())
      return std::nullopt;
    if (stride <= 0 || remaining < stride)
      continue;
    const int64_t coordinate = remaining / stride;
    remaining -= coordinate * stride;
    std::optional<Expr> named = namedPart(extents[dimension], coordinate);
    if (arithmetic.hasFailure())
      return std::nullopt;
    // A part's size is where the view begins only if the view then ends
    // inside the dimension at every size (a convolution tap's window of a
    // sequence's tokens, two columns in, begins at 2 when there are two
    // tokens too).
    if (named && !endsInsideSource(view, extents[dimension], *named, stride))
      named.reset();
    if (!named)
      named = tail(dimension, stride, coordinate);
    origin = arithmetic.add(
        origin, named ? arithmetic.multiply(strides[dimension], *named)
                      : arithmetic.scale(strides[dimension], coordinate));
  }
  if (remaining != 0 || arithmetic.hasFailure()) {
    arithmetic.unresolved("a view begins at byte " + Twine(byteOffset) +
                          ", which is not a coordinate in its source");
    return std::nullopt;
  }
  return origin;
}

LogicalResult Symbolizer::deriveStorage() {
  // Storage identity is a property of the view chain, not of any operation: an
  // offset accumulates along the chain from the root, and the span a value may
  // address is whatever its root occupies.
  DenseMap<Value, Expr> offsets;
  SmallVector<Value> stack;

  const auto offsetOf = [&](Value value) -> std::optional<Expr> {
    stack.clear();
    Value current = value;
    // Walk to the root, then unwind, so a chain of any length costs no stack.
    while (true) {
      if (offsets.count(current))
        break;
      Value source = getStorageSource(current);
      if (!source) {
        offsets[current] = constant(0);
        break;
      }
      if (llvm::is_contained(stack, current))
        return std::nullopt; // A cycle in the chain.
      stack.push_back(current);
      current = source;
    }
    for (Value pending : llvm::reverse(stack)) {
      Expr own = constant(0);
      if (pending.getDefiningOp<ViewOp>()) {
        // GGML normalizes a view's source to the root of the chain, so the
        // offset a view records is already measured from there.
        std::optional<int64_t> byteOffset = getStorageOffset(pending);
        if (!byteOffset)
          return std::nullopt;
        std::optional<Expr> where = viewOrigin(pending, *byteOffset);
        if (!where)
          return std::nullopt;
        own = *where;
      }
      offsets[pending] =
          arithmetic.add(offsets[getStorageSource(pending)], own);
    }
    return offsets[value];
  };

  for (auto &[value, description] : derived) {
    std::optional<Expr> offset = offsetOf(value);
    if (!offset)
      return reportOn(value) << "cannot resolve its position in its storage";
    description.storageOffset = *offset;
  }
  return success();
}

LogicalResult Symbolizer::verify() {
  // Evaluating a derived expression at the capture's symbol values must
  // reproduce the observed extent exactly, so a wrong rule fails on the graph
  // that produced it.
  for (auto &[value, description] : derived) {
    std::optional<SmallVector<int64_t, 4>> observedShape =
        getObservedExtents(value);
    std::optional<SmallVector<int64_t, 4>> observedStrides =
        getObservedStrides(value);
    if (!observedShape || !observedStrides)
      return reportOn(value) << "lost its observed `ggml.ne` or `ggml.nb`";

    for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
      arithmetic.clearFailure();
      const int64_t derivedExtent = at(description.shape[dimension]);
      const int64_t derivedStride = at(description.strides[dimension]);
      if (arithmetic.hasFailure())
        return reportOn(value)
               << "cannot evaluate: " << arithmetic.getFailure();
      if (derivedExtent != (*observedShape)[dimension]) {
        return reportOn(value)
               << "derived extent " << derivedExtent << " at dimension "
               << dimension << " but the capture observed "
               << (*observedShape)[dimension];
      }
      // A trailing dimension of one element past two request-dependent
      // extents steps nowhere, and its dense stride is none to compare; nor
      // is a view's that keeps its neighbour's for that reason (recoverWalk).
      const bool free = (description.freeStrides & (1u << dimension)) ||
                        (dimension > 0 && derivedExtent == 1 &&
                         description.strides[dimension].hasSameTerms(
                             description.strides[dimension - 1]) &&
                         description.strides[dimension].offset ==
                             description.strides[dimension - 1].offset &&
                         !isConstant(description.strides[dimension]));
      if (!free && derivedStride != (*observedStrides)[dimension]) {
        return reportOn(value)
               << "derived stride " << derivedStride << " at dimension "
               << dimension << " but the capture observed "
               << (*observedStrides)[dimension];
      }
    }

    if (std::optional<int64_t> observedOffset = getStorageOffset(value)) {
      arithmetic.clearFailure();
      const int64_t offset = at(description.storageOffset);
      if (arithmetic.hasFailure())
        return reportOn(value)
               << "cannot evaluate: " << arithmetic.getFailure();
      if (offset != *observedOffset) {
        return reportOn(value)
               << "derived storage offset " << offset
               << " but the capture observed " << *observedOffset;
      }
    }
  }
  return success();
}

void Symbolizer::writeBack() {
  MLIRContext *context = function.getContext();
  Builder builder(context);

  const auto toList = [&](const Shape &expressions) {
    SmallVector<Attribute, kGgmlRank> entries;
    for (const Expr &expression : expressions)
      entries.push_back(toAttribute(context, expression));
    return entries;
  };

  // What was derived is the value's layout, and a value's layout is its type.
  // Setting it also sets the shape, since `#ggml.layout` requires the two to
  // be the same four extents read in opposite directions.
  const auto symbolicType = [&](Value value, const Derived &description) {
    auto typed = cast<RankedTensorType>(value.getType());
    // What the bytes are encoded as is not something a shape says, so it is
    // carried across rather than derived.
    StringAttr storage;
    if (auto was = dyn_cast_or_null<LayoutAttr>(typed.getEncoding()))
      storage = was.getStorage();
    return withGgmlLayout(
        typed, LayoutAttr::get(context, toList(description.shape),
                               toList(description.strides),
                               toAttribute(context, description.storageOffset),
                               storage));
  };

  for (auto &[value, description] : derived) {
    Operation *producer = value.getDefiningOp();
    if (!producer)
      continue;
    value.setType(symbolicType(value, description));
  }

  Block &entry = function.getBody().front();
  for (BlockArgument argument : entry.getArguments()) {
    auto entryDerived = derived.find(argument);
    if (entryDerived != derived.end())
      argument.setType(symbolicType(argument, entryDerived->second));
  }

  // The entry signature follows the argument and result types.
  SmallVector<Type> results(function.getResultTypes());
  auto terminator = cast<func::ReturnOp>(entry.getTerminator());
  results.assign(terminator.getOperandTypes().begin(),
                 terminator.getOperandTypes().end());
  function.setType(builder.getFunctionType(entry.getArgumentTypes(), results));
}

struct GgmlSymbolizeShapesPass
    : impl::GgmlSymbolizeShapesPassBase<GgmlSymbolizeShapesPass> {
  using GgmlSymbolizeShapesPassBase::GgmlSymbolizeShapesPassBase;

  void runOnOperation() override {
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      if (function.getBody().empty())
        continue;
      Symbolizer symbolizer(function);
      if (failed(symbolizer.run()))
        return signalPassFailure();
    }
  }
};

} // namespace

} // namespace ggml
} // namespace mlir
