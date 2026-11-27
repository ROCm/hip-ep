/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Ggml/IR/GgmlDialect.h"
#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlOps.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeArithmetic.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/DialectImplementation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

using namespace mlir;
using namespace mlir::ggml;

#include "hip/Dialect/Ggml/IR/GgmlDialect.cpp.inc"

#define GET_ATTRDEF_CLASSES
#include "hip/Dialect/Ggml/IR/GgmlAttrs.cpp.inc"

#include "hip/Dialect/Ggml/IR/GgmlEnums.cpp.inc"

#include "hip/Dialect/Ggml/IR/GgmlShapeExprInterface.cpp.inc"

namespace {

/// Names the layouts so that a type reads as its shape.
///
/// A layout is spelled out once, at the top of the module, and every type that
/// carries it prints `#layout7`. Without this a module repeats each layout at
/// every operand and result that has it, which is most of them, and the shape
/// disappears into the repetition.
struct GgmlAsmInterface : public OpAsmDialectInterface {
  using OpAsmDialectInterface::OpAsmDialectInterface;

  AliasResult getAlias(Attribute attribute, raw_ostream &os) const override {
    if (!isa<LayoutAttr>(attribute))
      return AliasResult::NoAlias;
    os << "layout";
    return AliasResult::OverridableAlias;
  }
};

} // namespace

void GgmlDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "hip/Dialect/Ggml/IR/GgmlOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "hip/Dialect/Ggml/IR/GgmlAttrs.cpp.inc"
      >();
  addInterfaces<GgmlAsmInterface>();
}

namespace {

/// Sums repeated symbols, drops the ones that cancel, and sorts what remains,
/// which is what makes two attributes equal exactly when they denote the same
/// function.
FailureOr<SmallVector<AffineTermAttr>>
canonicalTerms(MLIRContext *context,
               llvm::ArrayRef<std::pair<llvm::StringRef, int64_t>> terms) {
  shape_expr::Arithmetic arithmetic;
  shape_expr::Expr normalized = arithmetic.normalize(terms, 0);
  if (arithmetic.hasFailure())
    return failure();

  SmallVector<AffineTermAttr> attributes;
  attributes.reserve(normalized.terms.size());
  for (auto [symbol, coefficient] : normalized.terms)
    attributes.push_back(AffineTermAttr::get(context, symbol, coefficient));
  return attributes;
}

/// Legacy metadata can carry arbitrary integer types. Reading their signed
/// value must neither assert nor truncate; the consumer decides its sign rule.
std::optional<int64_t> readShapeInteger(IntegerAttr integer) {
  if (!integer)
    return std::nullopt;
  auto type = dyn_cast<IntegerType>(integer.getType());
  if (!integer.getType().isIndex() &&
      (!type || !type.isSignless() || type.getWidth() == 1))
    return std::nullopt;
  return integer.getValue().trySExtValue();
}

} // namespace

//===----------------------------------------------------------------------===//
// AffineTermAttr
//===----------------------------------------------------------------------===//

LogicalResult
AffineTermAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                       llvm::StringRef symbol, int64_t coefficient) {
  if (symbol.empty())
    return emitError() << "requires a non-empty symbol name";
  if (coefficient == 0) {
    // A zero coefficient names a symbol the expression does not depend on, so
    // two spellings would denote one function and equality would stop meaning
    // sameness.
    return emitError() << "requires a nonzero coefficient; drop the term "
                          "instead of scaling it by zero";
  }
  return success();
}

//===----------------------------------------------------------------------===//
// AffineAttr
//===----------------------------------------------------------------------===//

LogicalResult AffineAttr::verify(function_ref<InFlightDiagnostic()> emitError,
                                 llvm::ArrayRef<AffineTermAttr> terms,
                                 int64_t offset) {
  if (terms.empty()) {
    return emitError() << "requires at least one term; a constant belongs in "
                          "an integer attribute";
  }
  for (auto [previous, current] :
       llvm::zip(terms.drop_back(), terms.drop_front())) {
    if (previous.getSymbol() == current.getSymbol())
      return emitError() << "names symbol '" << current.getSymbol()
                         << "' twice";
    if (previous.getSymbol() > current.getSymbol()) {
      return emitError() << "requires terms sorted by symbol name; '"
                         << previous.getSymbol() << "' precedes '"
                         << current.getSymbol() << "'";
    }
  }
  return success();
}

int64_t AffineAttr::getCoefficient(llvm::StringRef symbol) const {
  for (AffineTermAttr term : getTerms()) {
    if (term.getSymbol() == symbol)
      return term.getCoefficient();
  }
  return 0;
}

FailureOr<int64_t> AffineAttr::evaluate(SymbolLookup lookup) const {
  int64_t total = getOffset();
  for (AffineTermAttr term : getTerms()) {
    std::optional<int64_t> value = lookup(term.getSymbol());
    if (!value)
      return failure();
    int64_t scaled = 0;
    if (llvm::MulOverflow(term.getCoefficient(), *value, scaled) ||
        llvm::AddOverflow(total, scaled, total)) {
      return failure();
    }
  }
  return total;
}

void AffineAttr::getSymbols(SmallVectorImpl<llvm::StringRef> &symbols) const {
  for (AffineTermAttr term : getTerms()) {
    if (!llvm::is_contained(symbols, term.getSymbol()))
      symbols.push_back(term.getSymbol());
  }
}

std::optional<int64_t> AffineAttr::getConstantValue() const {
  // A verified attribute always carries a term, so it always varies.
  return std::nullopt;
}

Attribute AffineAttr::parse(AsmParser &parser, Type) {
  if (parser.parseLess())
    return {};

  // Symbol names are owned here until the attribute is built, so they are kept
  // by value: a StringRef into a vector that is still growing would dangle.
  SmallVector<std::pair<std::string, int64_t>> parsed;
  int64_t offset = 0;
  bool sawOffset = false;
  do {
    int64_t number = 0;
    llvm::SMLoc numberLocation = parser.getCurrentLocation();
    if (parser.parseInteger(number))
      return {};
    if (succeeded(parser.parseOptionalStar())) {
      std::string symbol;
      if (parser.parseKeywordOrString(&symbol))
        return {};
      parsed.emplace_back(std::move(symbol), number);
      continue;
    }
    if (sawOffset) {
      parser.emitError(numberLocation, "expected a single constant offset");
      return {};
    }
    offset = number;
    sawOffset = true;
  } while (succeeded(parser.parseOptionalPlus()));

  if (parser.parseGreater())
    return {};

  SmallVector<std::pair<llvm::StringRef, int64_t>> terms;
  terms.reserve(parsed.size());
  for (const auto &[symbol, coefficient] : parsed)
    terms.emplace_back(symbol, coefficient);

  FailureOr<SmallVector<AffineTermAttr>> canonical =
      canonicalTerms(parser.getContext(), terms);
  if (failed(canonical)) {
    parser.emitError(parser.getCurrentLocation(),
                     "shape expression coefficient sum overflows int64");
    return {};
  }
  if (canonical->empty()) {
    parser.emitError(parser.getCurrentLocation(),
                     "expected at least one scaled symbol");
    return {};
  }
  return parser.getChecked<AffineAttr>(parser.getContext(), *canonical, offset);
}

void AffineAttr::print(AsmPrinter &printer) const {
  printer << "<";
  llvm::interleave(
      getTerms(), printer.getStream(),
      [&](AffineTermAttr term) {
        printer << term.getCoefficient() << "*" << term.getSymbol();
      },
      " + ");
  if (getOffset() != 0)
    printer << " + " << getOffset();
  printer << ">";
}

//===----------------------------------------------------------------------===//
// Form-agnostic access
//===----------------------------------------------------------------------===//

bool mlir::ggml::isShapeExpr(Attribute attribute) {
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attribute))
    return readShapeInteger(integer).has_value();
  return isa_and_present<ShapeExprAttrInterface>(attribute);
}

FailureOr<int64_t> mlir::ggml::evaluateShapeExpr(Attribute attribute,
                                                 SymbolLookup lookup) {
  if (auto constant = dyn_cast_or_null<IntegerAttr>(attribute)) {
    if (std::optional<int64_t> value = readShapeInteger(constant))
      return *value;
    return failure();
  }
  if (auto expression = dyn_cast_or_null<ShapeExprAttrInterface>(attribute))
    return expression.evaluate(lookup);
  return failure();
}

void mlir::ggml::getShapeExprSymbols(
    Attribute attribute, SmallVectorImpl<llvm::StringRef> &symbols) {
  if (auto expression = dyn_cast_or_null<ShapeExprAttrInterface>(attribute))
    expression.getSymbols(symbols);
}

std::optional<int64_t> mlir::ggml::getShapeExprConstant(Attribute attribute) {
  if (auto constant = dyn_cast_or_null<IntegerAttr>(attribute))
    return readShapeInteger(constant);
  if (auto expression = dyn_cast_or_null<ShapeExprAttrInterface>(attribute))
    return expression.getConstantValue();
  return std::nullopt;
}

std::optional<ShapeExprView> mlir::ggml::getShapeExprView(Attribute attribute) {
  if (auto constant = dyn_cast_or_null<IntegerAttr>(attribute)) {
    std::optional<int64_t> value = readShapeInteger(constant);
    if (!value)
      return std::nullopt;
    return ShapeExprView{{}, *value};
  }

  if (auto affine = dyn_cast_or_null<AffineAttr>(attribute)) {
    ShapeExprView view;
    view.offset = affine.getOffset();
    for (AffineTermAttr term : affine.getTerms())
      view.terms.emplace_back(term.getSymbol(), term.getCoefficient());
    return view;
  }

  // Dictionary encodings predate the attribute and still appear in modules
  // emitted before it. Both are read here so that no consumer has to know they
  // exist.
  auto expression = dyn_cast_or_null<DictionaryAttr>(attribute);
  if (!expression)
    return std::nullopt;

  std::optional<int64_t> offset =
      readShapeInteger(expression.getAs<IntegerAttr>("offset"));
  if (expression.contains("offset") && !offset)
    return std::nullopt;
  const int64_t offsetValue = offset.value_or(0);

  SmallVector<std::pair<llvm::StringRef, int64_t>> terms;
  if (auto kind = expression.getAs<StringAttr>("kind")) {
    auto termList = expression.getAs<ArrayAttr>("terms");
    if (kind.getValue() != "affine" || !offset || !termList || termList.empty())
      return std::nullopt;
    for (Attribute entry : termList) {
      auto term = dyn_cast<DictionaryAttr>(entry);
      auto symbol = term ? term.getAs<StringAttr>("symbol") : StringAttr{};
      std::optional<int64_t> coefficient = readShapeInteger(
          term ? term.getAs<IntegerAttr>("coefficient") : IntegerAttr{});
      if (!symbol || symbol.getValue().empty() || !coefficient ||
          *coefficient == 0) {
        return std::nullopt;
      }
      terms.emplace_back(symbol.getValue(), *coefficient);
    }
  } else {
    // The one-symbol form: {symbol, scale, offset?}.
    auto symbol = expression.getAs<StringAttr>("symbol");
    std::optional<int64_t> scale =
        readShapeInteger(expression.getAs<IntegerAttr>("scale"));
    if (!symbol || symbol.getValue().empty() || !scale || *scale == 0)
      return std::nullopt;
    terms.emplace_back(symbol.getValue(), *scale);
  }

  ShapeExprView view;
  view.offset = offsetValue;
  for (auto [symbol, coefficient] : terms) {
    auto existing = llvm::find_if(
        view.terms, [&](const auto &kept) { return kept.first == symbol; });
    if (existing != view.terms.end())
      return std::nullopt; // A duplicated symbol has no single reading.
    view.terms.emplace_back(symbol, coefficient);
  }
  llvm::sort(view.terms, [](const auto &lhs, const auto &rhs) {
    return lhs.first < rhs.first;
  });
  return view;
}

bool mlir::ggml::haveEquivalentShapeExprs(Attribute lhs, Attribute rhs) {
  std::optional<ShapeExprView> left = getShapeExprView(lhs);
  std::optional<ShapeExprView> right = getShapeExprView(rhs);
  return left && right && left->offset == right->offset &&
         left->hasSameTerms(*right);
}

Attribute mlir::ggml::makeShapeExpr(
    MLIRContext *context,
    llvm::ArrayRef<std::pair<llvm::StringRef, int64_t>> terms, int64_t offset) {
  FailureOr<SmallVector<AffineTermAttr>> canonical =
      canonicalTerms(context, terms);
  if (failed(canonical))
    return {};
  if (canonical->empty())
    return IntegerAttr::get(IntegerType::get(context, 64), offset);
  return AffineAttr::get(context, *canonical, offset);
}

//===----------------------------------------------------------------------===//
// The tensor encoding
//===----------------------------------------------------------------------===//

namespace {

/// Prints one shape expression without the noise a bare attribute carries: a
/// constant as itself rather than `4096 : i64`, and an expression without its
/// dialect prefix, so a layout reads as the four numbers it is.
void printShapeExpr(AsmPrinter &printer, Attribute expression) {
  if (auto constant = dyn_cast<IntegerAttr>(expression)) {
    printer << constant.getInt();
    return;
  }
  if (auto affine = dyn_cast<AffineAttr>(expression)) {
    llvm::interleave(
        affine.getTerms(), printer.getStream(),
        [&](AffineTermAttr term) {
          printer << term.getCoefficient() << "*" << term.getSymbol();
        },
        " + ");
    if (affine.getOffset() != 0)
      printer << " + " << affine.getOffset();
    return;
  }
  printer << expression;
}

/// Reads back what `printShapeExpr` writes: a sum of scaled symbols with an
/// optional constant, or a constant alone.
Attribute parseShapeExpr(AsmParser &parser) {
  SmallVector<std::pair<std::string, int64_t>> parsed;
  int64_t offset = 0;
  bool sawOffset = false;
  do {
    int64_t number = 0;
    llvm::SMLoc numberLocation = parser.getCurrentLocation();
    if (parser.parseInteger(number))
      return {};
    if (succeeded(parser.parseOptionalStar())) {
      std::string symbol;
      if (parser.parseKeywordOrString(&symbol))
        return {};
      parsed.emplace_back(std::move(symbol), number);
      continue;
    }
    if (sawOffset) {
      parser.emitError(numberLocation, "expected a single constant offset");
      return {};
    }
    offset = number;
    sawOffset = true;
  } while (succeeded(parser.parseOptionalPlus()));

  SmallVector<std::pair<llvm::StringRef, int64_t>> terms;
  terms.reserve(parsed.size());
  for (auto &[symbol, coefficient] : parsed)
    terms.emplace_back(symbol, coefficient);
  Attribute result = makeShapeExpr(parser.getContext(), terms, offset);
  if (!result) {
    parser.emitError(parser.getCurrentLocation(),
                     "shape expression coefficient sum overflows int64");
  }
  return result;
}

/// Parses `[e0, e1, e2, e3]`.
LogicalResult parseShapeExprList(AsmParser &parser,
                                 SmallVectorImpl<Attribute> &into) {
  if (parser.parseLSquare())
    return failure();
  do {
    Attribute expression = parseShapeExpr(parser);
    if (!expression)
      return failure();
    into.push_back(expression);
  } while (succeeded(parser.parseOptionalComma()));
  return parser.parseRSquare();
}

void printShapeExprList(AsmPrinter &printer, ArrayRef<Attribute> list) {
  printer << "[";
  llvm::interleave(
      list, printer.getStream(),
      [&](Attribute expression) { printShapeExpr(printer, expression); }, ", ");
  printer << "]";
}

} // namespace

Attribute LayoutAttr::parse(AsmParser &parser, Type) {
  SmallVector<Attribute> extents;
  SmallVector<Attribute> strides;
  if (parser.parseLess() || parser.parseKeyword("ne") || parser.parseEqual() ||
      failed(parseShapeExprList(parser, extents)) || parser.parseComma() ||
      parser.parseKeyword("nb") || parser.parseEqual() ||
      failed(parseShapeExprList(parser, strides)) || parser.parseComma() ||
      parser.parseKeyword("offset") || parser.parseEqual()) {
    return {};
  }
  Attribute offset = parseShapeExpr(parser);
  if (!offset)
    return {};
  // Absent when the bytes are what the element type says they are, which is
  // every value in a graph that quantizes nothing.
  StringAttr storage;
  if (succeeded(parser.parseOptionalComma())) {
    std::string encoding;
    if (parser.parseKeyword("storage") || parser.parseEqual() ||
        parser.parseString(&encoding)) {
      return {};
    }
    storage = StringAttr::get(parser.getContext(), encoding);
  }
  if (parser.parseGreater())
    return {};
  return parser.getChecked<LayoutAttr>(parser.getContext(), extents, strides,
                                       offset, storage);
}

void LayoutAttr::print(AsmPrinter &printer) const {
  printer << "<ne = ";
  printShapeExprList(printer, getExtents());
  printer << ", nb = ";
  printShapeExprList(printer, getStrides());
  printer << ", offset = ";
  printShapeExpr(printer, getOffset());
  if (StringAttr storage = getStorage())
    printer << ", storage = \"" << storage.getValue() << "\"";
  printer << ">";
}

LogicalResult
LayoutAttr::verify(llvm::function_ref<InFlightDiagnostic()> emitError,
                   ArrayRef<Attribute> extents, ArrayRef<Attribute> strides,
                   Attribute offset, StringAttr storage) {
  if (extents.size() != kGgmlRank || strides.size() != kGgmlRank) {
    return emitError() << "expects " << kGgmlRank
                       << " extents and as many strides";
  }
  for (auto [name, list] :
       {std::make_pair("extent", extents), std::make_pair("stride", strides)}) {
    for (auto [dimension, expression] : llvm::enumerate(list)) {
      if (!isShapeExpr(expression)) {
        return emitError() << name << " " << dimension
                           << " is not a shape expression";
      }
    }
  }
  if (!isShapeExpr(offset))
    return emitError() << "offset is not a shape expression";
  if (storage && !getBlockLayout(storage.getValue())) {
    return emitError() << "storage '" << storage.getValue()
                       << "' is not a ggml type";
  }
  return success();
}

LogicalResult LayoutAttr::verifyEncoding(
    ArrayRef<int64_t> shape, Type elementType,
    llvm::function_ref<InFlightDiagnostic()> emitError) const {
  if (shape.size() != kGgmlRank)
    return emitError() << "expects a rank-" << kGgmlRank << " tensor";

  // A packed encoding says the bytes are blocks, and the element type says
  // what a block decodes to. GGML decodes every one of them to f32 and to
  // nothing else, so a packed value that claims to be anything else is
  // claiming a conversion GGML does not have.
  if (StringAttr storage = getStorage()) {
    std::optional<BlockLayout> block = getBlockLayout(storage.getValue());
    if (block && block->blockElements == 1) {
      // Equal byte widths do not make dense encodings interchangeable: f16
      // and bf16, for example, give the same bits different values.
      const bool matches = llvm::StringSwitch<bool>(storage.getValue())
                               .Case("f32", elementType.isF32())
                               .Case("f16", elementType.isF16())
                               .Case("bf16", elementType.isBF16())
                               .Case("f64", elementType.isF64())
                               .Case("i8", elementType.isSignlessInteger(8))
                               .Case("i16", elementType.isSignlessInteger(16))
                               .Case("i32", elementType.isSignlessInteger(32))
                               .Case("i64", elementType.isSignlessInteger(64))
                               .Default(false);
      if (!matches) {
        return emitError() << "is stored as " << storage.getValue()
                           << ", but its logical element type is "
                           << elementType;
      }
    } else if (block) {
      if (!elementType.isF32()) {
        return emitError() << "is stored as " << storage.getValue()
                           << ", which GGML only decodes to f32";
      }
      // The fastest dimension is where the blocks run, so it cannot hold part
      // of one.
      if (std::optional<int64_t> fastest = getShapeExprConstant(getExtent(0))) {
        if (*fastest % block->blockElements != 0) {
          return emitError() << "has " << *fastest << " elements along its "
                             << "fastest dimension, which is not a whole "
                             << "number of " << block->blockElements
                             << "-element " << storage.getValue() << " blocks";
        }
      }
    }
  }

  // `ne` runs innermost first and the shape runs outermost first, so the two
  // are the same four numbers read in opposite directions. Requiring that here
  // is what lets a reader take either one and know it has the other.
  for (unsigned dimension = 0; dimension < kGgmlRank; ++dimension) {
    const int64_t extent = shape[kGgmlRank - 1 - dimension];
    std::optional<int64_t> constant =
        getShapeExprConstant(getExtent(dimension));
    if (constant.has_value() != !ShapedType::isDynamic(extent)) {
      return emitError() << "extent " << dimension
                         << (constant ? " is constant but the shape is dynamic"
                                      : " varies but the shape is static");
    }
    if (constant && *constant != extent) {
      return emitError() << "extent " << dimension << " is " << *constant
                         << " but the shape says " << extent;
    }
  }
  return success();
}

#include "hip/Dialect/Ggml/IR/GgmlShapeRuleInterface.cpp.inc"
