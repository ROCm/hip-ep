/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- LayoutConversion.cpp - Lower GGML layout operations ---------------===//
//
// GGML numbers dimensions from the contiguous dimension (ne[0]) outwards. The
// frontend reverses them into conventional MLIR outermost-to-innermost order.
// This file proves and lowers the resulting layout operations.
//
// Reshape and the supported subsets of view are metadata changes. Dense,
// offset-zero reinterprets become tensor.reshape. Constant-offset rectangular
// views of dense storage become tensor.extract_slice plus legal rank
// reassociation. Permute changes logical indexing, so it becomes a
// materializing hip.transpose. Aliases and byte layouts remain explicit until
// the lowering proves that a standard tensor or HIP operation preserves them.
//
//===----------------------------------------------------------------------===//

#include "GgmlToHipUtils.h"
#include "hip/Dialect/Ggml/IR/GgmlSymbols.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <utility>

namespace mlir::hip {
namespace {

constexpr int64_t kGgmlRank = 4;

static bool checkedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  if (lhs < 0 || rhs < 0) {
    return false;
  }
  if (lhs != 0 && rhs > std::numeric_limits<int64_t>::max() / lhs) {
    return false;
  }
  result = lhs * rhs;
  return true;
}

/// A deliberately bounded affine form used by ggml shape and byte-offset
/// metadata.  Constants have a null symbol and coefficient zero.  Supporting
/// one symbol is sufficient for the current dynamic token axis while keeping
/// the proof below linear and inspectable.
struct OneSymbolAffine {
  StringAttr symbol;
  int64_t coefficient = 0;
  int64_t offset = 0;
};

struct SymbolBounds {
  int64_t minimum = 0;
  int64_t maximum = 0;
};

static bool checkedSignedAdd(int64_t lhs, int64_t rhs, int64_t &result) {
  return !llvm::AddOverflow(lhs, rhs, result);
}

static bool checkedSignedMultiply(int64_t lhs, int64_t rhs, int64_t &result) {
  return !llvm::MulOverflow(lhs, rhs, result);
}

static std::optional<OneSymbolAffine>
parseOneSymbolAffine(Attribute attribute) {
  std::optional<mlir::ggml::ShapeExprView> view =
      mlir::ggml::getShapeExprView(attribute);
  if (!view) {
    return std::nullopt;
  }
  if (view->isConstant()) {
    return OneSymbolAffine{/*symbol=*/{}, /*coefficient=*/0,
                           /*offset=*/view->offset};
  }
  std::optional<std::pair<llvm::StringRef, int64_t>> term =
      view->getSingleTerm();
  if (!term) {
    return std::nullopt;
  }
  return OneSymbolAffine{StringAttr::get(attribute.getContext(), term->first),
                         term->second, view->offset};
}

static bool sameExpression(const OneSymbolAffine &lhs,
                           const OneSymbolAffine &rhs) {
  return lhs.symbol == rhs.symbol && lhs.coefficient == rhs.coefficient &&
         lhs.offset == rhs.offset;
}

static std::optional<OneSymbolAffine> addExpressions(const OneSymbolAffine &lhs,
                                                     const OneSymbolAffine &rhs,
                                                     int64_t rhsScale = 1) {
  if (lhs.symbol && rhs.symbol && lhs.symbol != rhs.symbol) {
    return std::nullopt;
  }
  int64_t rhsCoefficient = 0;
  int64_t rhsOffset = 0;
  int64_t coefficient = 0;
  int64_t offset = 0;
  if (!checkedSignedMultiply(rhs.coefficient, rhsScale, rhsCoefficient) ||
      !checkedSignedMultiply(rhs.offset, rhsScale, rhsOffset) ||
      !checkedSignedAdd(lhs.coefficient, rhsCoefficient, coefficient) ||
      !checkedSignedAdd(lhs.offset, rhsOffset, offset)) {
    return std::nullopt;
  }
  StringAttr symbol = lhs.symbol ? lhs.symbol : rhs.symbol;
  if (coefficient == 0) {
    symbol = {};
  }
  return OneSymbolAffine{symbol, coefficient, offset};
}

static std::optional<OneSymbolAffine>
multiplyExpression(const OneSymbolAffine &expression, int64_t factor) {
  int64_t coefficient = 0;
  int64_t offset = 0;
  if (!checkedSignedMultiply(expression.coefficient, factor, coefficient) ||
      !checkedSignedMultiply(expression.offset, factor, offset)) {
    return std::nullopt;
  }
  return OneSymbolAffine{coefficient == 0 ? StringAttr{} : expression.symbol,
                         coefficient, offset};
}

static std::optional<OneSymbolAffine>
multiplyExpressions(const OneSymbolAffine &lhs, const OneSymbolAffine &rhs) {
  if (lhs.coefficient != 0 && rhs.coefficient != 0) {
    return std::nullopt;
  }
  if (lhs.coefficient == 0) {
    return multiplyExpression(rhs, lhs.offset);
  }
  return multiplyExpression(lhs, rhs.offset);
}

static std::optional<OneSymbolAffine>
divideExpressionExactly(const OneSymbolAffine &expression, int64_t divisor) {
  if (divisor <= 0 || expression.coefficient % divisor != 0 ||
      expression.offset % divisor != 0) {
    return std::nullopt;
  }
  const int64_t coefficient = expression.coefficient / divisor;
  return OneSymbolAffine{coefficient == 0 ? StringAttr{} : expression.symbol,
                         coefficient, expression.offset / divisor};
}

static std::optional<SymbolBounds> getSymbolBounds(Operation *op,
                                                   StringAttr symbol) {
  if (!symbol) {
    return SymbolBounds{};
  }
  auto module = op->getParentOfType<ModuleOp>();
  std::string reason;
  auto symbols = ggml::readSymbolDeclarations(
      module ? module->getAttr("ggml.symbols") : Attribute{}, reason);
  if (failed(symbols))
    return std::nullopt;
  const ggml::SymbolDeclaration *selected =
      ggml::findSymbolDeclaration(*symbols, symbol.getValue());
  if (!selected || !selected->minimum || !selected->maximum)
    return std::nullopt;
  return SymbolBounds{*selected->minimum, *selected->maximum};
}

static std::optional<std::pair<int64_t, int64_t>>
getExpressionBounds(const OneSymbolAffine &expression,
                    const SymbolBounds &bounds) {
  const auto evaluate = [&](int64_t symbolValue) -> std::optional<int64_t> {
    int64_t product = 0;
    int64_t result = 0;
    if (!checkedSignedMultiply(expression.coefficient, symbolValue, product) ||
        !checkedSignedAdd(product, expression.offset, result)) {
      return std::nullopt;
    }
    return result;
  };
  std::optional<int64_t> atMinimum = evaluate(bounds.minimum);
  std::optional<int64_t> atMaximum = evaluate(bounds.maximum);
  if (!atMinimum || !atMaximum) {
    return std::nullopt;
  }
  return std::minmax(*atMinimum, *atMaximum);
}

static bool isProvablyNonNegative(const OneSymbolAffine &expression,
                                  const SymbolBounds &bounds) {
  std::optional<std::pair<int64_t, int64_t>> range =
      getExpressionBounds(expression, bounds);
  return range && range->first >= 0;
}

static bool isProvablyAtMost(const OneSymbolAffine &lhs,
                             const OneSymbolAffine &rhs,
                             const SymbolBounds &bounds) {
  std::optional<OneSymbolAffine> difference = addExpressions(rhs, lhs, -1);
  return difference && isProvablyNonNegative(*difference, bounds);
}

static std::optional<OneSymbolAffine>
getDimensionExpression(Value value, RankedTensorType type, int64_t dimension) {
  Attribute expression =
      getGgmlDimensionExpression(value, dimension, type.getRank());
  if (expression) {
    return parseOneSymbolAffine(expression);
  }
  if (!type.isDynamicDim(dimension)) {
    return OneSymbolAffine{/*symbol=*/{}, /*coefficient=*/0,
                           /*offset=*/type.getDimSize(dimension)};
  }
  return std::nullopt;
}

static std::optional<SmallVector<OneSymbolAffine>>
getDimensionExpressions(Value value, RankedTensorType type) {
  SmallVector<OneSymbolAffine> result;
  result.reserve(type.getRank());
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
    std::optional<OneSymbolAffine> expression =
        getDimensionExpression(value, type, dimension);
    if (!expression) {
      return std::nullopt;
    }
    result.push_back(*expression);
  }
  return result;
}

static std::optional<OneSymbolAffine>
getElementCountExpression(ArrayRef<OneSymbolAffine> dimensions) {
  OneSymbolAffine result{/*symbol=*/{}, /*coefficient=*/0, /*offset=*/1};
  for (const OneSymbolAffine &dimension : dimensions) {
    std::optional<OneSymbolAffine> product =
        multiplyExpressions(result, dimension);
    if (!product) {
      return std::nullopt;
    }
    result = *product;
  }
  return result;
}

static bool mergeExpressionSymbol(StringAttr &symbol,
                                  const OneSymbolAffine &expression) {
  if (!expression.symbol) {
    return true;
  }
  if (symbol && symbol != expression.symbol) {
    return false;
  }
  symbol = expression.symbol;
  return true;
}

/// Prove dense logical iteration for a GGML rank-4 tensor. Strides attached to
/// dimensions whose maximum extent is one are intentionally ignored: GGML is
/// free to retain a parent stride for a singleton view axis, and that stride is
/// never used to address a second element.
static bool hasAffineDenseByteStrides(ArrayRef<Attribute> byteStrides,
                                      ArrayRef<OneSymbolAffine> ggmlDimensions,
                                      int64_t elementBytes, StringAttr symbol,
                                      const SymbolBounds &bounds) {
  if (byteStrides.size() != kGgmlRank || ggmlDimensions.size() != kGgmlRank ||
      elementBytes <= 0) {
    return false;
  }

  OneSymbolAffine expected{/*symbol=*/{}, /*coefficient=*/0,
                           /*offset=*/elementBytes};
  for (int64_t dimension = 0; dimension < kGgmlRank; ++dimension) {
    std::optional<OneSymbolAffine> actual =
        parseOneSymbolAffine(byteStrides[dimension]);
    std::optional<std::pair<int64_t, int64_t>> dimensionRange =
        getExpressionBounds(ggmlDimensions[dimension], bounds);
    std::optional<std::pair<int64_t, int64_t>> strideRange =
        actual ? getExpressionBounds(*actual, bounds) : std::nullopt;
    if (!actual || !dimensionRange || !strideRange || strideRange->first < 0) {
      return false;
    }
    if ((actual->symbol && actual->symbol != symbol) ||
        (ggmlDimensions[dimension].symbol &&
         ggmlDimensions[dimension].symbol != symbol)) {
      return false;
    }
    if (dimensionRange->second > 1 && !sameExpression(*actual, expected)) {
      return false;
    }
    std::optional<OneSymbolAffine> next =
        multiplyExpressions(expected, ggmlDimensions[dimension]);
    if (!next) {
      return false;
    }
    expected = *next;
  }
  return true;
}

static Value materializeExpression(ConversionPatternRewriter &rewriter,
                                   Location location,
                                   const OneSymbolAffine &expression,
                                   Value symbolValue) {
  if (expression.coefficient == 0) {
    return arith::ConstantIndexOp::create(rewriter, location,
                                          expression.offset);
  }
  Value coefficient = arith::ConstantIndexOp::create(rewriter, location,
                                                     expression.coefficient);
  Value scaled =
      arith::MulIOp::create(rewriter, location, symbolValue, coefficient);
  if (expression.offset == 0) {
    return scaled;
  }
  Value offset =
      arith::ConstantIndexOp::create(rewriter, location, expression.offset);
  return arith::AddIOp::create(rewriter, location, scaled, offset);
}

static bool expressionsMatchTensorType(ArrayRef<OneSymbolAffine> dimensions,
                                       RankedTensorType type) {
  if (dimensions.size() != static_cast<size_t>(type.getRank())) {
    return false;
  }
  for (int64_t dimension = 0; dimension < type.getRank(); ++dimension) {
    const OneSymbolAffine &expression = dimensions[dimension];
    if (type.isDynamicDim(dimension)) {
      if (expression.coefficient == 0) {
        return false;
      }
      continue;
    }
    if (expression.coefficient != 0 ||
        expression.offset != type.getDimSize(dimension)) {
      return false;
    }
  }
  return true;
}

static SmallVector<OneSymbolAffine>
reverseDimensions(ArrayRef<OneSymbolAffine> dimensions) {
  return SmallVector<OneSymbolAffine>(dimensions.rbegin(), dimensions.rend());
}

/// Lower the bounded affine subset of dense GGML views as a contiguous slice
/// of flattened storage followed by a shape restoration.  The proof accepts a
/// single graph symbol with a finite declared range and covers prefix/suffix,
/// last-row, and fixed-plane views.  It deliberately rejects nonlinear shape
/// products, unbounded symbols, and partial writable aliases.
static LogicalResult
lowerAffineContiguousView(mlir::ggml::ViewOp op, Value input,
                          ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!inputType || !resultType || inputType.getRank() != kGgmlRank ||
      resultType.getRank() != kGgmlRank) {
    return op.emitOpError(
        "affine rectangular view lowering requires rank-4 tensors");
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return op.emitOpError(
        "affine rectangular view lowering requires matching element types");
  }
  if (!hasDenseInputEvidence(op.getInput()) || !hasDenseOutputEvidence(op)) {
    return op.emitOpError(
        "affine rectangular view lowering requires proven dense input and "
        "result layouts");
  }

  Type elementType = inputType.getElementType();
  const unsigned bitWidth =
      elementType.isIntOrFloat() ? elementType.getIntOrFloatBitWidth() : 0;
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    return op.emitOpError(
        "affine rectangular view lowering requires byte-addressable integer "
        "or floating-point elements");
  }
  const int64_t elementBytes = bitWidth / 8;

  std::optional<SmallVector<OneSymbolAffine>> inputDimensions =
      getDimensionExpressions(op.getInput(), inputType);
  std::optional<SmallVector<OneSymbolAffine>> resultDimensions =
      getDimensionExpressions(op.getResult(), resultType);
  if (!inputDimensions || !resultDimensions ||
      !expressionsMatchTensorType(*inputDimensions, inputType) ||
      !expressionsMatchTensorType(*resultDimensions, resultType)) {
    return op.emitOpError(
        "affine rectangular view lowering requires complete shape-expression "
        "metadata consistent with the tensor types");
  }

  Attribute inputStorageOffsetAttribute = getConvertedGgmlOffset(op.getInput());
  Attribute resultStorageOffsetAttribute =
      getConvertedGgmlOffset(op.getResult());
  std::optional<OneSymbolAffine> inputStorageOffset =
      parseOneSymbolAffine(inputStorageOffsetAttribute);
  std::optional<OneSymbolAffine> resultStorageOffset =
      parseOneSymbolAffine(resultStorageOffsetAttribute);
  std::optional<OneSymbolAffine> byteOffset =
      parseOneSymbolAffine(getConvertedGgmlOffset(op.getResult()));
  Attribute storageLengthAttribute =
      ggml::getProvenanceAttr(op.getInput(), "ggml.storage_byte_length");
  std::optional<OneSymbolAffine> storageLength =
      storageLengthAttribute ? parseOneSymbolAffine(storageLengthAttribute)
                             : std::optional<OneSymbolAffine>{};
  // That the result shares the input's storage root is not asked: `ggml.view`
  // is view-like, so its result views its input by definition.
  if (!inputStorageOffset || !resultStorageOffset || !byteOffset ||
      !sameExpression(*byteOffset, *resultStorageOffset)) {
    return op.emitOpError(
        "affine rectangular view lowering requires matching one-symbol "
        "byte_offset and storage_offset metadata");
  }

  StringAttr symbol;
  for (const OneSymbolAffine &expression : *inputDimensions) {
    if (!mergeExpressionSymbol(symbol, expression)) {
      return op.emitOpError(
          "affine rectangular view lowering does not support multiple symbols");
    }
  }
  for (const OneSymbolAffine &expression : *resultDimensions) {
    if (!mergeExpressionSymbol(symbol, expression)) {
      return op.emitOpError(
          "affine rectangular view lowering does not support multiple symbols");
    }
  }
  if (!mergeExpressionSymbol(symbol, *inputStorageOffset) ||
      !mergeExpressionSymbol(symbol, *resultStorageOffset)) {
    return op.emitOpError(
        "affine rectangular view lowering does not support multiple symbols");
  }
  if (storageLengthAttribute &&
      (!storageLength || !mergeExpressionSymbol(symbol, *storageLength))) {
    return op.emitOpError(
        "affine rectangular view has invalid or multi-symbol storage bounds");
  }
  if (!symbol) {
    return op.emitOpError(
        "affine rectangular view lowering requires one dynamic symbol");
  }

  std::optional<SymbolBounds> bounds = getSymbolBounds(op, symbol);
  if (!bounds) {
    return op.emitOpError()
           << "affine rectangular view lowering requires finite minimum and "
              "maximum bounds for symbol '"
           << symbol.getValue() << "'";
  }
  if (!isProvablyNonNegative(*inputStorageOffset, *bounds) ||
      !isProvablyNonNegative(*resultStorageOffset, *bounds)) {
    return op.emitOpError(
        "affine rectangular view storage offset can become negative");
  }

  for (const OneSymbolAffine &expression : *inputDimensions) {
    if (!isProvablyNonNegative(expression, *bounds)) {
      return op.emitOpError(
          "affine rectangular view input extent can become negative");
    }
  }
  for (const OneSymbolAffine &expression : *resultDimensions) {
    if (!isProvablyNonNegative(expression, *bounds)) {
      return op.emitOpError(
          "affine rectangular view result extent can become negative");
    }
  }

  SmallVector<OneSymbolAffine> inputGgmlDimensions =
      reverseDimensions(*inputDimensions);
  SmallVector<OneSymbolAffine> resultGgmlDimensions =
      reverseDimensions(*resultDimensions);
  ArrayRef<Attribute> inputByteStrides = getConvertedGgmlStrides(op.getInput());
  if (!hasAffineDenseByteStrides(inputByteStrides, inputGgmlDimensions,
                                 elementBytes, symbol, *bounds) ||
      !hasAffineDenseByteStrides(getConvertedGgmlStrides(op.getResult()),
                                 resultGgmlDimensions, elementBytes, symbol,
                                 *bounds)) {
    return op.emitOpError(
        "affine rectangular view strides do not prove dense logical storage");
  }

  std::optional<OneSymbolAffine> relativeByteOffset =
      addExpressions(*resultStorageOffset, *inputStorageOffset, -1);
  std::optional<OneSymbolAffine> relativeElementOffset =
      relativeByteOffset
          ? divideExpressionExactly(*relativeByteOffset, elementBytes)
          : std::nullopt;
  std::optional<OneSymbolAffine> inputElementCount =
      getElementCountExpression(*inputDimensions);
  std::optional<OneSymbolAffine> resultElementCount =
      getElementCountExpression(*resultDimensions);
  if (!relativeElementOffset || !inputElementCount || !resultElementCount) {
    return op.emitOpError(
        "affine rectangular view requires linear, element-aligned offset and "
        "cardinality expressions");
  }
  if (storageLength) {
    std::optional<OneSymbolAffine> inputByteCount =
        multiplyExpression(*inputElementCount, elementBytes);
    std::optional<OneSymbolAffine> resultByteCount =
        multiplyExpression(*resultElementCount, elementBytes);
    std::optional<OneSymbolAffine> inputEnd =
        inputByteCount ? addExpressions(*inputStorageOffset, *inputByteCount)
                       : std::nullopt;
    std::optional<OneSymbolAffine> resultEnd =
        resultByteCount ? addExpressions(*resultStorageOffset, *resultByteCount)
                        : std::nullopt;
    if (!inputEnd || !resultEnd ||
        !isProvablyNonNegative(*storageLength, *bounds) ||
        !isProvablyAtMost(*inputEnd, *storageLength, *bounds) ||
        !isProvablyAtMost(*resultEnd, *storageLength, *bounds)) {
      return op.emitOpError(
          "affine rectangular view exceeds its storage-root bounds");
    }
  }
  std::optional<OneSymbolAffine> sliceEnd =
      addExpressions(*relativeElementOffset, *resultElementCount);
  if (!sliceEnd || !isProvablyNonNegative(*relativeElementOffset, *bounds) ||
      !isProvablyNonNegative(*resultElementCount, *bounds) ||
      !isProvablyAtMost(*sliceEnd, *inputElementCount, *bounds)) {
    return op.emitOpError(
        "affine rectangular view offset and extent are not provably within "
        "the input storage bounds");
  }

  // The symbol is read back off an input extent that is a multiple of it. A
  // unit multiple is preferred; a flat buffer packing several parts end to
  // end -- a scan's outputs ahead of its state -- has only a larger one.
  int64_t anchorDimension = -1;
  OneSymbolAffine anchorExpression;
  for (int64_t dimension = 0; dimension < inputType.getRank(); ++dimension) {
    const OneSymbolAffine &expression = (*inputDimensions)[dimension];
    if (!inputType.isDynamicDim(dimension) || expression.symbol != symbol ||
        expression.coefficient <= 0)
      continue;
    if (anchorDimension < 0 || expression.coefficient == 1) {
      anchorDimension = dimension;
      anchorExpression = expression;
    }
    if (expression.coefficient == 1)
      break;
  }
  // A slice of a fixed input as long as a symbol - a cache's first n_kv
  // rows - reads it off an argument of the function that carries it.
  Value anchorSource = input;
  if (anchorDimension < 0)
    if (auto function = op->getParentOfType<func::FuncOp>())
      for (BlockArgument argument : function.getArguments()) {
        auto argumentType = dyn_cast<RankedTensorType>(argument.getType());
        if (!argumentType || argumentType.getRank() != kGgmlRank)
          continue;
        std::optional<SmallVector<OneSymbolAffine>> dimensions =
            getDimensionExpressions(argument, argumentType);
        for (int64_t dimension = 0; dimensions && dimension < kGgmlRank;
             ++dimension)
          if (argumentType.isDynamicDim(dimension) &&
              (*dimensions)[dimension].symbol == symbol &&
              (*dimensions)[dimension].coefficient == 1) {
            anchorSource = argument;
            anchorDimension = dimension;
            anchorExpression = (*dimensions)[dimension];
            break;
          }
        if (anchorDimension >= 0)
          break;
      }
  if (anchorDimension < 0) {
    return op.emitOpError()
           << "affine rectangular view requires an input dimension anchoring "
              "symbol '"
           << symbol.getValue() << "'";
  }

  Value anchor = tensor::DimOp::create(rewriter, op.getLoc(), anchorSource,
                                       anchorDimension);
  Value anchorOffset = arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                                      anchorExpression.offset);
  Value symbolValue =
      arith::SubIOp::create(rewriter, op.getLoc(), anchor, anchorOffset);
  if (anchorExpression.coefficient != 1) {
    symbolValue = arith::DivUIOp::create(
        rewriter, op.getLoc(), symbolValue,
        arith::ConstantIndexOp::create(rewriter, op.getLoc(),
                                       anchorExpression.coefficient));
  }

  const int64_t flatInputExtent = inputElementCount->coefficient == 0
                                      ? inputElementCount->offset
                                      : ShapedType::kDynamic;
  auto flatInputType = RankedTensorType::get({flatInputExtent}, elementType);
  Value flatInput = tensor::CollapseShapeOp::create(
      rewriter, op.getLoc(), flatInputType, input,
      SmallVector<ReassociationIndices>{{0, 1, 2, 3}});

  const int64_t flatResultExtent = resultElementCount->coefficient == 0
                                       ? resultElementCount->offset
                                       : ShapedType::kDynamic;
  auto flatResultType = RankedTensorType::get({flatResultExtent}, elementType);
  SmallVector<OpFoldResult> flatOffsets = {
      relativeElementOffset->coefficient == 0
          ? OpFoldResult(rewriter.getIndexAttr(relativeElementOffset->offset))
          : OpFoldResult(materializeExpression(
                rewriter, op.getLoc(), *relativeElementOffset, symbolValue))};
  SmallVector<OpFoldResult> flatSizes = {
      resultElementCount->coefficient == 0
          ? OpFoldResult(rewriter.getIndexAttr(resultElementCount->offset))
          : OpFoldResult(materializeExpression(
                rewriter, op.getLoc(), *resultElementCount, symbolValue))};
  SmallVector<OpFoldResult> flatStrides = {rewriter.getIndexAttr(1)};
  Value slice = tensor::ExtractSliceOp::create(
      rewriter, op.getLoc(), flatResultType, flatInput, flatOffsets, flatSizes,
      flatStrides);

  SmallVector<Value> resultShapeValues;
  resultShapeValues.reserve(kGgmlRank);
  for (const OneSymbolAffine &expression : *resultDimensions) {
    Value extent =
        materializeExpression(rewriter, op.getLoc(), expression, symbolValue);
    resultShapeValues.push_back(arith::IndexCastOp::create(
        rewriter, op.getLoc(), rewriter.getI64Type(), extent));
  }
  auto shapeType = RankedTensorType::get({kGgmlRank}, rewriter.getI64Type());
  Value resultShape = tensor::FromElementsOp::create(
      rewriter, op.getLoc(), shapeType, resultShapeValues);
  auto reshaped = tensor::ReshapeOp::create(rewriter, op.getLoc(), resultType,
                                            slice, resultShape);
  copyGgmlResultProvenanceAttrs(op, reshaped);
  reshaped->setAttr("ggml.contiguous", rewriter.getBoolAttr(true));
  rewriter.replaceOp(op, reshaped.getResult());
  return success();
}

static std::optional<int64_t> getStaticDimensionProduct(RankedTensorType type) {
  if (llvm::is_contained(type.getShape(), 0)) {
    return 0;
  }
  int64_t product = 1;
  for (int64_t dim : type.getShape()) {
    if (ShapedType::isDynamic(dim)) {
      continue;
    }
    if (!checkedMultiply(product, dim, product)) {
      return std::nullopt;
    }
  }
  return product;
}

static SmallVector<int64_t> getDynamicDimensions(RankedTensorType type) {
  SmallVector<int64_t> dimensions;
  for (int64_t i = 0; i < type.getRank(); ++i) {
    if (type.isDynamicDim(i)) {
      dimensions.push_back(i);
    }
  }
  return dimensions;
}

/// Decode either the canonical affine-expression form or the legacy
/// one-symbol form when it denotes exactly `scale * symbol`.
static std::optional<std::pair<StringAttr, int64_t>>
getSingleSymbolicScale(Attribute attribute) {
  std::optional<mlir::ggml::ShapeExprView> view =
      mlir::ggml::getShapeExprView(attribute);
  if (!view || view->offset != 0) {
    return std::nullopt;
  }
  std::optional<std::pair<llvm::StringRef, int64_t>> term =
      view->getSingleTerm();
  if (!term) {
    return std::nullopt;
  }
  return std::make_pair(StringAttr::get(attribute.getContext(), term->first),
                        term->second);
}

/// Validate a GGML byte-stride array against a dense rank-4 result layout.
/// The array is in GGML ne/nb order, hence dimension `i` corresponds to MLIR
/// dimension `rank - 1 - i`.  Dynamic strides emitted by the frontend are
/// dictionaries of the form `{scale = N, symbol = "..."}`.
static bool hasDenseByteStrides(ArrayRef<Attribute> strides,
                                RankedTensorType resultType) {
  if (resultType.getRank() != kGgmlRank ||
      static_cast<int64_t>(strides.size()) != kGgmlRank) {
    return false;
  }

  Type elementType = resultType.getElementType();
  if (!elementType.isIntOrFloat()) {
    return false;
  }
  unsigned bitWidth = elementType.getIntOrFloatBitWidth();
  if (bitWidth == 0 || bitWidth % 8 != 0) {
    return false;
  }

  int64_t scale = bitWidth / 8;
  bool containsDynamicExtent = false;
  // Whether the stride after the dynamic extent is still to be read: that
  // extent may be a multiple of its symbol (four rows per image block), whose
  // factor the first stride over it carries.
  bool absorbFactor = false;
  StringAttr dynamicSymbol;

  for (int64_t ggmlDim = 0; ggmlDim < kGgmlRank; ++ggmlDim) {
    Attribute stride = strides[ggmlDim];
    if (!containsDynamicExtent) {
      auto integer = dyn_cast<IntegerAttr>(stride);
      if (!integer || integer.getInt() != scale) {
        return false;
      }
    } else {
      auto symbolic = getSingleSymbolicScale(stride);
      if (symbolic && absorbFactor && symbolic->second % scale == 0)
        scale = symbolic->second;
      absorbFactor = false;
      if (!symbolic || symbolic->second != scale) {
        return false;
      }
      if (dynamicSymbol && dynamicSymbol != symbolic->first) {
        return false;
      }
      dynamicSymbol = symbolic->first;
    }

    int64_t mlirDim = kGgmlRank - 1 - ggmlDim;
    int64_t extent = resultType.getDimSize(mlirDim);
    if (ShapedType::isDynamic(extent)) {
      // The compact symbolic format currently models one dynamic dimension.
      if (containsDynamicExtent) {
        return false;
      }
      containsDynamicExtent = true;
      absorbFactor = true;
      continue;
    }
    if (!checkedMultiply(scale, extent, scale)) {
      return false;
    }
  }
  return true;
}

/// Materialize a rank-4 shape tensor for a cardinality-preserving reshape.
///
/// Supported dynamic shapes contain one varying dimension on each side and
/// equal static products.  In that case the output dynamic extent is exactly
/// the input dynamic extent.  More general multi-dynamic factorizations are
/// left illegal until the frontend carries explicit dimension SSA values;
/// guessing them from anonymous `?` dimensions would be unsound.
static FailureOr<Value>
buildSafeReshapeShape(Operation *op, Value input, RankedTensorType inputType,
                      RankedTensorType resultType,
                      ConversionPatternRewriter &rewriter) {
  if (inputType.getRank() != kGgmlRank || resultType.getRank() != kGgmlRank) {
    return failure();
  }
  if (inputType.getElementType() != resultType.getElementType()) {
    return failure();
  }

  std::optional<int64_t> inputStaticProduct =
      getStaticDimensionProduct(inputType);
  std::optional<int64_t> resultStaticProduct =
      getStaticDimensionProduct(resultType);
  if (!inputStaticProduct || !resultStaticProduct) {
    return failure();
  }

  SmallVector<int64_t> inputDynamic = getDynamicDimensions(inputType);
  SmallVector<int64_t> resultDynamic = getDynamicDimensions(resultType);
  Value resultDynamicExtent;

  if (inputDynamic.empty() && resultDynamic.empty()) {
    if (*inputStaticProduct != *resultStaticProduct) {
      return failure();
    }
  } else if (inputDynamic.size() == 1 && resultDynamic.size() == 1 &&
             *inputStaticProduct == *resultStaticProduct &&
             *inputStaticProduct != 0 &&
             haveEquivalentGgmlDimensionExtent(
                 op->getOperand(0), inputDynamic.front(), op->getResult(0),
                 resultDynamic.front())) {
    resultDynamicExtent = tensor::DimOp::create(rewriter, op->getLoc(), input,
                                                inputDynamic.front());
  } else {
    return failure();
  }

  SmallVector<Value> dimensions;
  dimensions.reserve(kGgmlRank);
  for (int64_t i = 0; i < kGgmlRank; ++i) {
    Value extent;
    if (resultType.isDynamicDim(i)) {
      extent = resultDynamicExtent;
    } else {
      extent = arith::ConstantIndexOp::create(rewriter, op->getLoc(),
                                              resultType.getDimSize(i));
    }
    dimensions.push_back(arith::IndexCastOp::create(
        rewriter, op->getLoc(), rewriter.getI64Type(), extent));
  }

  auto shapeType = RankedTensorType::get({kGgmlRank}, rewriter.getI64Type());
  return tensor::FromElementsOp::create(rewriter, op->getLoc(), shapeType,
                                        dimensions)
      .getResult();
}

static LogicalResult lowerDenseReshape(Operation *op, Value input,
                                       ArrayRef<Attribute> byteStrides,
                                       ConversionPatternRewriter &rewriter) {
  auto inputType = dyn_cast<RankedTensorType>(input.getType());
  auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
  if (!inputType || !resultType) {
    return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
  }
  if (inputType.getRank() != kGgmlRank || resultType.getRank() != kGgmlRank) {
    return rewriter.notifyMatchFailure(op,
                                       "requires fixed rank-4 tensor types");
  }
  if (!hasDenseOutputEvidence(op)) {
    return rewriter.notifyMatchFailure(
        op, "requires unambiguous ggml.contiguous = true evidence");
  }
  if (!hasDenseInputEvidence(op->getOperand(0))) {
    return rewriter.notifyMatchFailure(
        op, "input is not proven to have a dense logical layout");
  }
  if (!hasDenseByteStrides(byteStrides, resultType)) {
    return rewriter.notifyMatchFailure(
        op, "GGML byte strides do not prove a dense result layout");
  }
  if (inputType == resultType) {
    rewriter.replaceOp(op, input);
    return success();
  }

  FailureOr<Value> shape =
      buildSafeReshapeShape(op, input, inputType, resultType, rewriter);
  if (failed(shape)) {
    return rewriter.notifyMatchFailure(
        op, "cannot prove element cardinality for a dense rank-4 reshape");
  }

  auto reshape = tensor::ReshapeOp::create(rewriter, op->getLoc(), resultType,
                                           input, *shape);
  // Retain the proof for a following GGML view if dialect conversion visits
  // the producer first and remaps the consumer operand to this standard op.
  reshape->setAttr("ggml.contiguous", rewriter.getBoolAttr(true));
  rewriter.replaceOp(op, reshape.getResult());
  return success();
}

struct ReshapeConversion final : OpConversionPattern<mlir::ggml::ReshapeOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::ggml::ReshapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    return lowerDenseReshape(op, adaptor.getInput(),
                             getConvertedGgmlStrides(op.getResult()), rewriter);
  }
};

struct ViewConversion final : OpConversionPattern<mlir::ggml::ViewOp> {
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(mlir::ggml::ViewOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Attribute resultOffset = getConvertedGgmlOffset(op.getResult());
    if (!resultOffset)
      resultOffset = op.getByteOffsetAttr();
    auto byteOffset = dyn_cast_or_null<IntegerAttr>(resultOffset);
    auto inputType = dyn_cast<RankedTensorType>(adaptor.getInput().getType());
    auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
    const bool hasEquivalentDynamicExtent =
        inputType && resultType &&
        haveEquivalentGgmlDynamicExtent(op, op.getInput(), inputType,
                                        resultType);
    if ((!byteOffset || !hasEquivalentDynamicExtent) &&
        hasDenseInputEvidence(op.getInput()) && hasDenseOutputEvidence(op)) {
      return lowerAffineContiguousView(op, adaptor.getInput(), rewriter);
    }
    if (!byteOffset) {
      return op.emitOpError(
          "HIP lowering does not support symbolic GGML view byte offsets");
    }
    // A dense window of fewer elements than its input (a cache's first rows)
    // is no reshape of it.
    const bool window =
        inputType && resultType && inputType.hasStaticShape() &&
        resultType.hasStaticShape() &&
        inputType.getNumElements() != resultType.getNumElements();
    if (byteOffset.getInt() != 0 || !hasDenseOutputEvidence(op) || window) {
      return rewriter.notifyMatchFailure(
          op, "supports only dense views that are reshapes of their input");
    }
    return lowerDenseReshape(op, adaptor.getInput(),
                             getConvertedGgmlStrides(op.getResult()), rewriter);
  }
};

} // namespace

void populateGgmlLayoutConversionPatterns(RewritePatternSet &patterns) {
  patterns.add<ReshapeConversion, ViewConversion>(patterns.getContext());
}

} // namespace mlir::hip
