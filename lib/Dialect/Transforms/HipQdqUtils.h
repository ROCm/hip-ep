/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

// Readers for the parameters of hip.quantize_linear / hip.dequantize_linear and
// the init builder, shared by hip-fusion-transform's native helpers and
// hip-fold-qdq-to-quant-types.
#pragma once

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Quant/IR/QuantTypes.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

#include <cmath>
#include <cstdint>
#include <optional>

namespace hip {
namespace fusion_transform {

//===----------------------------------------------------------------------===//
// IR readers
//===----------------------------------------------------------------------===//

/// value is const op and return value as DenseElementsAttr, if not return {}
inline mlir::DenseElementsAttr tryHipConstantPayload(mlir::Value value) {
  if (!value)
    return {};
  auto constOp =
      mlir::dyn_cast_or_null<mlir::hip::ConstantOp>(value.getDefiningOp());
  if (!constOp)
    return {};
  return mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
      constOp->getAttr("value"));
}

/// value is a splat float constant and return it as float, if not return
/// nullopt
inline std::optional<float> tryHipSplatScale(mlir::Value value) {
  mlir::DenseElementsAttr payload = tryHipConstantPayload(value);
  // A non-splat scale is per-axis and has no single scalar equivalent.
  if (!payload || !payload.isSplat())
    return std::nullopt;
  // getSplatValue<FloatAttr> asserts on a non-float payload, so the element
  // type is what guards the read rather than a dyn_cast on its result.
  if (!mlir::isa<mlir::FloatType>(payload.getElementType()))
    return std::nullopt;
  return static_cast<float>(
      payload.getSplatValue<mlir::FloatAttr>().getValueAsDouble());
}

inline mlir::Value getQdqZeroPoint(mlir::Operation *op) {
  // Read through the ODS accessor, not an operand index: the zero point is
  // optional, and an absent one shifts the DPS init operand into its slot.
  if (auto quantOp = mlir::dyn_cast_or_null<mlir::hip::QuantizeLinearOp>(op))
    return quantOp.getZeroPoint();
  if (auto dequantOp =
          mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(op))
    return dequantOp.getZeroPoint();
  return {};
}

inline mlir::IntegerType getQdqQuantizedElementType(mlir::Operation *op) {
  mlir::Type type;
  if (auto quantOp = mlir::dyn_cast_or_null<mlir::hip::QuantizeLinearOp>(op))
    type = quantOp.getOutput().getType();
  else if (auto dequantOp =
               mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(op))
    type = dequantOp.getInput().getType();
  auto shaped = mlir::dyn_cast_or_null<mlir::ShapedType>(type);
  if (!shaped)
    return {};
  return mlir::dyn_cast<mlir::IntegerType>(shaped.getElementType());
}

/// op is a Q/DQ whose zero point is a splat constant and return it as int64,
/// an absent operand returns absentValue, anything else nullopt
inline std::optional<int64_t> tryHipQdqZeropoint(mlir::Operation *op,
                                                 int64_t absentValue) {
  if (!op)
    return std::nullopt;
  mlir::Value zeroPoint = getQdqZeroPoint(op);
  // ONNX makes the operand optional and defines its absence as zero.
  if (!zeroPoint)
    return absentValue;
  mlir::DenseElementsAttr payload = tryHipConstantPayload(zeroPoint);
  if (!payload || !payload.isSplat())
    return std::nullopt;
  if (!mlir::isa<mlir::IntegerType>(payload.getElementType()))
    return std::nullopt;
  // The payload's own element type carries no signedness for the signless
  // integers the importer produces, so the stored type of the quantized side
  // is what decides how the raw bits are read.
  auto intType = getQdqQuantizedElementType(op);
  if (!intType)
    return std::nullopt;
  llvm::APInt raw = payload.getSplatValue<llvm::APInt>();
  return intType.isUnsigned() ? static_cast<int64_t>(raw.getZExtValue())
                              : raw.getSExtValue();
}

/// per-tensor quant type of a Q/DQ: splat scale, readable zero point and hip
/// storage integer; null if not representable
inline mlir::quant::UniformQuantizedType
getHipQdqQuantType(mlir::Operation *op) {
  mlir::Value scale, floatSide;
  if (auto q = mlir::dyn_cast_or_null<mlir::hip::QuantizeLinearOp>(op)) {
    scale = q.getScale();
    floatSide = q.getInput();
  } else if (auto dq =
                 mlir::dyn_cast_or_null<mlir::hip::DequantizeLinearOp>(op)) {
    scale = dq.getScale();
    floatSide = dq.getResult(0);
  } else {
    return {};
  }
  // packed_int4 storage is not one value per element.
  if (op->hasAttr("packed_int4"))
    return {};

  auto intType = getQdqQuantizedElementType(op);
  auto expressed = mlir::dyn_cast<mlir::FloatType>(
      mlir::getElementTypeOrSelf(floatSide.getType()));
  std::optional<float> scaleValue = tryHipSplatScale(scale);
  std::optional<int64_t> zeroPoint = tryHipQdqZeropoint(op, 0);
  if (!intType || intType.isSigned() || intType.getWidth() > 32 || !expressed ||
      !scaleValue || !zeroPoint || !std::isfinite(*scaleValue) ||
      *scaleValue <= 0.0f)
    return {};

  bool isSigned = !intType.isUnsigned();
  unsigned width = intType.getWidth();
  int64_t min =
      mlir::quant::QuantizedType::getDefaultMinimumForInteger(isSigned, width);
  int64_t max =
      mlir::quant::QuantizedType::getDefaultMaximumForInteger(isSigned, width);
  if (*zeroPoint < min || *zeroPoint > max)
    return {};

  return mlir::quant::UniformQuantizedType::get(
      isSigned ? mlir::quant::QuantizationFlags::Signed : 0,
      mlir::IntegerType::get(op->getContext(), width), expressed, *scaleValue,
      *zeroPoint, min, max);
}

//===----------------------------------------------------------------------===//
// Rewrite helpers
//===----------------------------------------------------------------------===//

/// a tensor.empty of initType, sizing each dynamic dim from shapeSource
inline mlir::Value buildInitValue(mlir::PatternRewriter &rewriter,
                                  mlir::RankedTensorType initType,
                                  mlir::Value shapeSource) {
  mlir::Location loc = shapeSource.getLoc();

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t dim : llvm::seq<int64_t>(initType.getRank()))
    if (initType.isDynamicDim(dim))
      dynSizes.push_back(
          mlir::tensor::DimOp::create(rewriter, loc, shapeSource, dim));

  return mlir::tensor::EmptyOp::create(rewriter, loc, initType.getShape(),
                                       initType.getElementType(), dynSizes)
      .getResult();
}

} // namespace fusion_transform
} // namespace hip
