/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- GgmlToHipUtils.h - Shared GGML-to-HIP helpers -----*- C++ -*-===//
//
// This header is deliberately independent of the ONNX frontend. It contains
// only utilities shared by direct conversions from registered ggml operations
// to HIP and standard MLIR operations.
//
//===----------------------------------------------------------------------===//

#ifndef HIP_CONVERSION_GGMLTOHIP_GGMLTOHIPUTILS_H
#define HIP_CONVERSION_GGMLTOHIP_GGMLTOHIPUTILS_H

#include "hip/Conversion/GgmlToHip/Passes.h"
#include "hip/Dialect/Ggml/IR/GgmlLayout.h"
#include "hip/Dialect/Ggml/IR/GgmlOps.h"
#include "hip/Dialect/Ggml/IR/GgmlProvenance.h"
#include "hip/Dialect/Ggml/IR/GgmlShapeExpr.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipKQuant.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <limits>
#include <optional>
#include <string>
#include <tuple>

namespace mlir {
namespace hip {

struct GgmlKQuantFormatInfo {
  kquant::Format format;
  int64_t blockElements;
  int64_t blockBytes;
};

/// Return the ONNX TensorProto numeric code used by hip.cast's `to`
/// attribute.
inline std::optional<int64_t> getHipCastElementTypeCode(Type type) {
  if (type.isF32())
    return 1;
  if (type.isInteger(8))
    return 3;
  if (type.isInteger(16))
    return 5;
  if (type.isInteger(32))
    return 6;
  if (type.isInteger(64))
    return 7;
  if (type.isF16())
    return 10;
  if (type.isF64())
    return 11;
  if (type.isBF16())
    return 16;
  return std::nullopt;
}

/// The weight formats the native kernels read where llama.cpp loaded them.
inline std::optional<GgmlKQuantFormatInfo>
getGgmlKQuantFormatInfo(StringRef encoding) {
  auto kQuant = [](kquant::Format format) {
    return GgmlKQuantFormatInfo{format, kquant::kBlockElements,
                                kquant::getBlockBytes(format)};
  };
  if (encoding == "q4_K")
    return kQuant(kquant::Format::Q4K);
  if (encoding == "q5_K")
    return kQuant(kquant::Format::Q5K);
  if (encoding == "q6_K")
    return kQuant(kquant::Format::Q6K);
  return std::nullopt;
}

/// Return the runtime context threaded through argument zero by
/// `hip-add-context-arg`.
inline FailureOr<Value> getHipContext(Operation *op, PatternRewriter &) {
  auto function = op->getParentOfType<func::FuncOp>();
  if (!function) {
    return op->emitOpError("is not inside func.func");
  }
  if (function.isDeclaration() ||
      function.getBody().front().getNumArguments() == 0) {
    return op->emitOpError(
        "requires a parent function with an argument zero; run "
        "hip-add-context-arg first");
  }

  Value context = function.getBody().front().getArgument(0);
  if (!isa<hip::ContextType>(context.getType())) {
    return op->emitOpError(
        "requires function argument zero to be !hip.context; run "
        "hip-add-context-arg first");
  }
  return context;
}

/// Create a DPS tensor.empty whose dynamic extents are taken positionally
/// from a same-rank shaped value.
inline Value createGgmlEmptyTensor(OpBuilder &builder, Location loc,
                                   RankedTensorType resultType,
                                   Value shapeSource) {
  auto sourceType = dyn_cast<RankedTensorType>(shapeSource.getType());
  if (!sourceType) {
    emitError(loc) << "cannot materialize a dynamic tensor shape from "
                   << shapeSource.getType() << "; expected a ranked tensor";
    return {};
  }
  if (sourceType.getRank() != resultType.getRank()) {
    emitError(loc) << "cannot materialize rank-" << resultType.getRank()
                   << " tensor shape from rank-" << sourceType.getRank()
                   << " shape source";
    return {};
  }

  SmallVector<Value> dynamicSizes;
  for (int64_t dimension : llvm::seq<int64_t>(resultType.getRank())) {
    if (resultType.isDynamicDim(dimension)) {
      dynamicSizes.push_back(
          tensor::DimOp::create(builder, loc, shapeSource, dimension));
    }
  }
  auto empty =
      tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                              resultType.getElementType(), dynamicSizes);
  return empty.getResult();
}

/// Create a DPS tensor.empty for a NumPy-style broadcast result. Each dynamic
/// extent is read from the first operand that contributes a non-unit dimension
/// at the corresponding right-aligned axis.
inline FailureOr<Value>
createGgmlBroadcastEmptyTensor(OpBuilder &builder, Location loc,
                               RankedTensorType resultType,
                               ValueRange operands) {
  const int64_t resultRank = resultType.getRank();
  SmallVector<Value> dynamicSizes;
  for (int64_t resultDim : llvm::seq<int64_t>(resultRank)) {
    if (!resultType.isDynamicDim(resultDim)) {
      continue;
    }

    Value chosen;
    int64_t chosenDim = -1;
    Value fallback;
    int64_t fallbackDim = -1;
    for (Value operand : operands) {
      auto operandType = dyn_cast<RankedTensorType>(operand.getType());
      if (!operandType) {
        continue;
      }
      const int64_t rankOffset = resultRank - operandType.getRank();
      if (rankOffset < 0 || resultDim < rankOffset) {
        continue;
      }
      const int64_t operandDim = resultDim - rankOffset;
      if (!fallback) {
        fallback = operand;
        fallbackDim = operandDim;
      }
      if (!operandType.isDynamicDim(operandDim) &&
          operandType.getDimSize(operandDim) == 1) {
        continue;
      }
      chosen = operand;
      chosenDim = operandDim;
      break;
    }
    if (!chosen) {
      chosen = fallback;
      chosenDim = fallbackDim;
    }
    if (!chosen) {
      return failure();
    }
    dynamicSizes.push_back(
        tensor::DimOp::create(builder, loc, chosen, chosenDim));
  }

  return Value(tensor::EmptyOp::create(builder, loc, resultType.getShape(),
                                       resultType.getElementType(),
                                       dynamicSizes));
}

/// The extents a value's layout says, innermost first, whether or not it still
/// has one.
///
/// A conversion replaces producers as it goes, so by the time a pattern asks,
/// an operand may already be the `hip` operation that replaced it -- whose type
/// belongs to a dialect that carries no layout, and which was handed the
/// extents as attributes instead. Both are read here so that no pattern has to
/// know which of the two it is looking at.
inline ArrayRef<Attribute> getConvertedGgmlExtents(Value value) {
  if (mlir::ggml::LayoutAttr layout = mlir::ggml::getGgmlLayout(value))
    return layout.getExtents();
  if (auto extents = dyn_cast_or_null<ArrayAttr>(
          ggml::getProvenanceAttr(value, "ggml.ne"))) {
    return extents.getValue();
  }
  return {};
}

/// The byte strides a value's layout says, read the same way.
inline ArrayRef<Attribute> getConvertedGgmlStrides(Value value) {
  if (mlir::ggml::LayoutAttr layout = mlir::ggml::getGgmlLayout(value))
    return layout.getStrides();
  if (auto strides = dyn_cast_or_null<ArrayAttr>(
          ggml::getProvenanceAttr(value, "ggml.nb"))) {
    return strides.getValue();
  }
  return {};
}

/// Where a value begins in its buffer, read the same way, or null when it says
/// nothing about that.
inline Attribute getConvertedGgmlOffset(Value value) {
  if (mlir::ggml::LayoutAttr layout = mlir::ggml::getGgmlLayout(value))
    return layout.getOffset();
  return ggml::getProvenanceAttr(value, "ggml.storage_offset");
}

/// Whether reading `value` linearly walks its elements in order, answered from
/// wherever its layout currently is.
inline std::optional<mlir::ggml::Contiguity>
deriveConvertedContiguity(Value value) {
  ArrayRef<Attribute> extents = getConvertedGgmlExtents(value);
  ArrayRef<Attribute> strides = getConvertedGgmlStrides(value);
  if (extents.empty() || strides.empty())
    return std::nullopt;
  std::optional<mlir::ggml::BlockLayout> layout =
      mlir::ggml::getStorageLayout(value);
  if (!layout)
    return mlir::ggml::Contiguity::Varies;
  return mlir::ggml::deriveContiguity(extents, strides, *layout,
                                      mlir::ggml::getScope(value));
}

/// Return true when every captured execution mode proves that an operation
/// result has canonical dense storage.
inline bool hasDenseOutputEvidence(Operation *op) {
  if (op->getNumResults() == 1) {
    if (std::optional<mlir::ggml::Contiguity> derived =
            deriveConvertedContiguity(op->getResult(0))) {
      return *derived == mlir::ggml::Contiguity::Dense;
    }
  }
  auto contiguous = op->getAttrOfType<BoolAttr>("ggml.contiguous");
  return contiguous && contiguous.getValue();
}

/// Return true when reading `value` linearly preserves its logical tensor
/// semantics. Standard tensor values are dense by construction; live GGML
/// producers require explicit positive capture evidence.
inline bool hasDenseInputEvidence(Value value) {
  // A value that still states its GGML layout answers for itself; one that
  // carries only the conclusion was produced by a conversion that derived it
  // while the layout was still there.
  if (std::optional<mlir::ggml::Contiguity> derived =
          deriveConvertedContiguity(value)) {
    return *derived == mlir::ggml::Contiguity::Dense;
  }
  Attribute contiguous = ggml::getProvenanceAttr(value, "ggml.contiguous");
  if (auto flag = dyn_cast_or_null<BoolAttr>(contiguous))
    return flag.getValue();
  if (contiguous)
    return false;

  Operation *producer = value.getDefiningOp();
  return !producer || producer->getName().getDialectNamespace() != "ggml";
}

/// Retain result/storage identity on a replacement producer while a full
/// conversion still contains GGML users. Control metadata describes the source
/// operation itself and must not be copied, and neither must the operation's
/// own name: the target is not that operation.
inline void copyGgmlResultProvenanceAttrs(Operation *source,
                                          Operation *target) {
  for (StringRef name :
       {"ggml.storage_byte_length", "ggml.value_id", "ggml.source_name"}) {
    if (Attribute attribute = source->getAttr(name))
      target->setAttr(name, attribute);
  }
  if (source->getNumResults() != 1)
    return;
  OpBuilder builder(source);

  // The layout is the source's type, and the target's type belongs to a dialect
  // that does not carry one, so what is still needed of it travels as
  // attributes: the extents and strides for the readers that walk them, and
  // whether the bytes are dense for the readers that only want the conclusion.
  for (StringRef name : {"ggml.ne", "ggml.nb", "ggml.storage_offset"}) {
    if (Attribute attribute = source->getAttr(name))
      target->setAttr(name, attribute);
  }
  if (std::optional<mlir::ggml::Contiguity> derived =
          deriveConvertedContiguity(source->getResult(0))) {
    target->setAttr(
        "ggml.contiguous",
        builder.getBoolAttr(*derived == mlir::ggml::Contiguity::Dense));
  }
}

/// The graph-declared expression for one MLIR tensor dimension.
inline Attribute getGgmlDimensionExpression(Value value, int64_t dimension,
                                            int64_t rank) {
  ArrayRef<Attribute> extents = getConvertedGgmlExtents(value);
  if (extents.size() == static_cast<size_t>(rank))
    return extents[rank - 1 - dimension];

  // Modules written before #ggml.layout was introduced carry shape
  // expressions in MLIR dimension order; keep that representation readable.
  // Static dimensions need no attribute fallback because the caller can read
  // them from the ranked tensor type.
  if (auto shape = dyn_cast_or_null<ArrayAttr>(
          ggml::getProvenanceAttr(value, "ggml.shape_expr"))) {
    if (shape.size() == static_cast<size_t>(rank))
      return shape[dimension];
  }
  return {};
}

/// Prove that two tensor dimensions have the same runtime extent. Static
/// dimensions compare directly. Dynamic dimensions must carry identical GGML
/// shape expressions because RankedTensorType represents every symbolic extent
/// as the same anonymous `?`.
inline bool haveEquivalentGgmlDimensionExtent(Value lhs, int64_t lhsDimension,
                                              Value rhs, int64_t rhsDimension) {
  auto lhsType = dyn_cast<RankedTensorType>(lhs.getType());
  auto rhsType = dyn_cast<RankedTensorType>(rhs.getType());
  if (!lhsType || !rhsType || lhsDimension < 0 || rhsDimension < 0 ||
      lhsDimension >= lhsType.getRank() || rhsDimension >= rhsType.getRank()) {
    return false;
  }

  const int64_t lhsExtent = lhsType.getDimSize(lhsDimension);
  const int64_t rhsExtent = rhsType.getDimSize(rhsDimension);
  const bool lhsDynamic = ShapedType::isDynamic(lhsExtent);
  const bool rhsDynamic = ShapedType::isDynamic(rhsExtent);
  if (!lhsDynamic && !rhsDynamic) {
    return lhsExtent == rhsExtent;
  }
  if (lhsDynamic != rhsDynamic) {
    return false;
  }

  Attribute lhsExpression =
      getGgmlDimensionExpression(lhs, lhsDimension, lhsType.getRank());
  Attribute rhsExpression =
      getGgmlDimensionExpression(rhs, rhsDimension, rhsType.getRank());
  return lhsExpression && rhsExpression && lhsExpression == rhsExpression;
}

/// Prove that the sole dynamic input/result dimensions carry the same
/// graph-declared expression. Ranked tensor types erase symbolic identity to
/// anonymous `?`, so type equality alone cannot distinguish T from T-1.
inline bool haveEquivalentGgmlDynamicExtent(Operation *op, Value input,
                                            RankedTensorType inputType,
                                            RankedTensorType resultType) {
  SmallVector<int64_t> inputDynamic;
  SmallVector<int64_t> resultDynamic;
  for (int64_t dim = 0; dim < inputType.getRank(); ++dim) {
    if (inputType.isDynamicDim(dim)) {
      inputDynamic.push_back(dim);
    }
  }
  for (int64_t dim = 0; dim < resultType.getRank(); ++dim) {
    if (resultType.isDynamicDim(dim)) {
      resultDynamic.push_back(dim);
    }
  }
  if (inputDynamic.empty() && resultDynamic.empty()) {
    return true;
  }
  if (inputDynamic.size() != 1 || resultDynamic.size() != 1 ||
      inputType.getRank() != resultType.getRank()) {
    return false;
  }

  return haveEquivalentGgmlDimensionExtent(
      input, inputDynamic.front(), op->getResult(0), resultDynamic.front());
}

void populateGgmlBasicConversionPatterns(RewritePatternSet &patterns);
void populateGgmlFusionPatterns(RewritePatternSet &patterns);
void populateGgmlRmsNormScaleFusionPatterns(RewritePatternSet &patterns);
void populateHipAddRmsNormQ8FusionPatterns(RewritePatternSet &patterns);
void populateHipKQuantSwiGluFusionPatterns(RewritePatternSet &patterns);
void populateGgmlLayoutConversionPatterns(RewritePatternSet &patterns);
void populateGgmlComputeConversionPatterns(RewritePatternSet &patterns);
void populateGgmlMatmulConversionPatterns(RewritePatternSet &patterns);

/// A rank-1 i64 shape tensor of `dimensions`, static or not.
inline Value shapeTensor(PatternRewriter &rewriter, Location loc,
                         ArrayRef<OpFoldResult> dimensions) {
  SmallVector<Value> elements;
  elements.reserve(dimensions.size());
  for (OpFoldResult dimension : dimensions) {
    if (auto attribute = dyn_cast<Attribute>(dimension)) {
      const int64_t value = cast<IntegerAttr>(attribute).getInt();
      elements.push_back(
          arith::ConstantIntOp::create(rewriter, loc, value, 64));
      continue;
    }
    Value value = cast<Value>(dimension);
    if (value.getType().isIndex()) {
      value = arith::IndexCastOp::create(rewriter, loc, rewriter.getI64Type(),
                                         value);
    }
    elements.push_back(value);
  }
  auto type = RankedTensorType::get({static_cast<int64_t>(elements.size())},
                                    rewriter.getI64Type());
  return tensor::FromElementsOp::create(rewriter, loc, type, elements);
}

/// `input` reshaped to `resultType`, whose extents are `dimensions`.
inline Value reshapeTensor(PatternRewriter &rewriter, Location loc, Value input,
                           RankedTensorType resultType,
                           ArrayRef<OpFoldResult> dimensions) {
  return tensor::ReshapeOp::create(rewriter, loc, resultType, input,
                                   shapeTensor(rewriter, loc, dimensions))
      .getResult();
}

/// Materialize `input` with `targetElementType` using hip.cast, returning the
/// original value when no conversion is required.
Value castTensor(PatternRewriter &rewriter, Location loc, Value context,
                 Value input, Type targetElementType);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_GGMLTOHIP_GGMLTOHIPUTILS_H
