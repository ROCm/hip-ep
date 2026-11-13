/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- fusion_utils.hpp - helpers shared by hip-fusion-transform patterns -===//
#pragma once

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/APFloat.h"

#include <optional>

namespace hip {
namespace fusion_transform {

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

/// The value of a one-element float constant, in double precision.
/// std::nullopt when `attr` is null, not one element, or not a float.
inline std::optional<double> singleFloatValue(mlir::DenseElementsAttr attr) {
  if (!attr || attr.getNumElements() != 1 ||
      !mlir::isa<mlir::FloatType>(attr.getElementType()))
    return std::nullopt;
  llvm::APFloat value = *attr.getValues<llvm::APFloat>().begin();
  bool lostInfo = false;
  value.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lostInfo);
  return value.convertToDouble();
}

/// The value of a scalar float constant operand (rank-0 or one-element),
/// looking through the wrappers frontends add: tensor.splat of a scalar, and
/// hip.cast / arith.truncf / arith.extf. std::nullopt when `value` is not a
/// single float constant.
inline std::optional<double> getScalarFloatConstant(mlir::Value value) {
  mlir::Operation *def = value.getDefiningOp();
  if (!def)
    return std::nullopt;
  if (auto constant = mlir::dyn_cast<mlir::hip::ConstantOp>(def))
    return singleFloatValue(mlir::dyn_cast_or_null<mlir::DenseElementsAttr>(
        constant.getValueAttr()));
  if (auto constant = mlir::dyn_cast<mlir::arith::ConstantOp>(def)) {
    if (auto scalar = mlir::dyn_cast<mlir::FloatAttr>(constant.getValue())) {
      llvm::APFloat v = scalar.getValue();
      bool lostInfo = false;
      v.convert(llvm::APFloat::IEEEdouble(), llvm::APFloat::rmNearestTiesToEven,
                &lostInfo);
      return v.convertToDouble();
    }
    return singleFloatValue(
        mlir::dyn_cast<mlir::DenseElementsAttr>(constant.getValue()));
  }
  if (auto splat = mlir::dyn_cast<mlir::tensor::SplatOp>(def))
    return getScalarFloatConstant(splat.getInput());
  if (auto cast = mlir::dyn_cast<mlir::hip::CastOp>(def))
    return getScalarFloatConstant(cast.getInput());
  if (auto trunc = mlir::dyn_cast<mlir::arith::TruncFOp>(def))
    return getScalarFloatConstant(trunc.getIn());
  if (auto ext = mlir::dyn_cast<mlir::arith::ExtFOp>(def))
    return getScalarFloatConstant(ext.getIn());
  return std::nullopt;
}

} // namespace fusion_transform
} // namespace hip
