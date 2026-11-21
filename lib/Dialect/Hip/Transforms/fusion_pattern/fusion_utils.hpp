/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- fusion_utils.hpp - helpers shared by hip-fusion-transform patterns -===//
#pragma once

#include "hip/Dialect/Hip/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APFloat.h"

#include <optional>

namespace hip {
namespace fusion_transform {

inline void retargetShapeQueries(mlir::PatternRewriter &rewriter,
                                 mlir::Value from, mlir::Value to) {
  rewriter.replaceUsesWithIf(from, to, [](mlir::OpOperand &operand) {
    return mlir::isa<mlir::tensor::DimOp>(operand.getOwner());
  });
}

/// True when every use of `value` is either `consumer` or a tensor.dim.
/// tensor.dim only reads the type, so a fusion that preserves the type may
/// retarget it. A consumer that reads `value` twice is not a single use.
inline bool hasSingleValueUse(mlir::Value value, mlir::Operation *consumer) {
  mlir::Operation *seen = nullptr;
  for (mlir::OpOperand &use : value.getUses()) {
    if (mlir::isa<mlir::tensor::DimOp>(use.getOwner()))
      continue;
    if (seen)
      return false;
    seen = use.getOwner();
  }
  return seen == consumer;
}

inline std::optional<llvm::APFloat>
singleFloatValue(mlir::DenseElementsAttr attr) {
  if (!attr || attr.getNumElements() != 1 ||
      !mlir::isa<mlir::FloatType>(attr.getElementType()))
    return std::nullopt;
  return *attr.getValues<llvm::APFloat>().begin();
}

inline std::optional<llvm::APFloat>
getScalarFloatConstantValue(mlir::Value value) {
  mlir::Type type = value.getType();
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(type)) {
    if (!mlir::isa<mlir::RankedTensorType>(shaped) ||
        !shaped.hasStaticShape() || shaped.getNumElements() != 1)
      return std::nullopt;
    type = shaped.getElementType();
  }
  auto floatType = mlir::dyn_cast<mlir::FloatType>(type);
  if (!floatType)
    return std::nullopt;

  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto constant = mlir::dyn_cast<mlir::hip::ConstantOp>(def))
    return singleFloatValue(mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        constant.getValueAttr()));
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    if (auto scalar = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue()))
      return scalar.getValue();
    return singleFloatValue(
        mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue()));
  }
  if (auto splat = mlir::dyn_cast<mlir::tensor::SplatOp>(def))
    return getScalarFloatConstantValue(splat.getInput());

  mlir::Value input;
  llvm::APFloat::roundingMode rounding = llvm::APFloat::rmNearestTiesToEven;
  if (auto cast = mlir::dyn_cast<mlir::hip::CastOp>(def)) {
    input = cast.getInput();
    auto inputType = mlir::dyn_cast<mlir::ShapedType>(input.getType());
    // The bf16 runtime conversion passes through f32.
    if (inputType && inputType.getElementType().isF64() && floatType.isBF16())
      return std::nullopt;
  } else if (auto trunc = mlir::dyn_cast<mlir::arith::TruncFOp>(def)) {
    input = trunc.getIn();
    if (auto mode = trunc.getRoundingmode()) {
      switch (*mode) {
      case mlir::arith::RoundingMode::to_nearest_even:
        break;
      case mlir::arith::RoundingMode::downward:
        rounding = llvm::APFloat::rmTowardNegative;
        break;
      case mlir::arith::RoundingMode::upward:
        rounding = llvm::APFloat::rmTowardPositive;
        break;
      case mlir::arith::RoundingMode::toward_zero:
        rounding = llvm::APFloat::rmTowardZero;
        break;
      case mlir::arith::RoundingMode::to_nearest_away:
        rounding = llvm::APFloat::rmNearestTiesToAway;
        break;
      }
    }
  } else if (auto ext = mlir::dyn_cast<mlir::arith::ExtFOp>(def)) {
    input = ext.getIn();
  } else {
    return std::nullopt;
  }
  std::optional<llvm::APFloat> result = getScalarFloatConstantValue(input);
  if (!result)
    return std::nullopt;
  bool lostInfo = false;
  result->convert(floatType.getFloatSemantics(), rounding, &lostInfo);
  return result;
}

/// Read a scalar or statically one-element floating-point constant, applying
/// supported casts before returning its value in double precision.
inline std::optional<double> getScalarFloatConstant(mlir::Value value) {
  std::optional<llvm::APFloat> result = getScalarFloatConstantValue(value);
  if (!result)
    return std::nullopt;
  bool lostInfo = false;
  result->convert(llvm::APFloat::IEEEdouble(),
                  llvm::APFloat::rmNearestTiesToEven, &lostInfo);
  return result->convertToDouble();
}

} // namespace fusion_transform
} // namespace hip
