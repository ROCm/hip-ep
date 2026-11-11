/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- column_slice.hpp - recognize last-axis column slices ---------------===//
#pragma once

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/Sequence.h"

namespace hip {
namespace fusion_transform {

// A rank-preserving, unit-stride slice of `source` that keeps every leading
// axis whole and takes `width` columns of the last axis from `offset`.
inline bool isColumnSlice(mlir::tensor::ExtractSliceOp slice,
                          mlir::Value source, int64_t offset, int64_t width) {
  auto sourceType = mlir::dyn_cast<mlir::RankedTensorType>(source.getType());
  auto resultType = slice.getResultType();
  if (slice.getSource() != source || !sourceType ||
      resultType.getRank() != sourceType.getRank())
    return false;
  const int64_t rank = sourceType.getRank();
  auto offsets = slice.getMixedOffsets();
  auto sizes = slice.getMixedSizes();
  auto strides = slice.getMixedStrides();
  for (int64_t i : llvm::seq<int64_t>(0, rank)) {
    if (mlir::getConstantIntValue(strides[i]) != std::optional<int64_t>(1))
      return false;
    const bool last = i + 1 == rank;
    if (mlir::getConstantIntValue(offsets[i]) !=
        std::optional<int64_t>(last ? offset : 0))
      return false;
    if (last) {
      if (mlir::getConstantIntValue(sizes[i]) != std::optional<int64_t>(width))
        return false;
      continue;
    }
    if (std::optional<int64_t> size = mlir::getConstantIntValue(sizes[i])) {
      if (sourceType.isDynamicDim(i) || *size != sourceType.getDimSize(i))
        return false;
      continue;
    }
    auto dim = mlir::dyn_cast<mlir::Value>(sizes[i])
                   .getDefiningOp<mlir::tensor::DimOp>();
    if (!dim || dim.getSource() != source ||
        dim.getConstantIndex() != std::optional<int64_t>(i))
      return false;
  }
  return true;
}

} // namespace fusion_transform
} // namespace hip
