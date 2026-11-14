/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- extent_trace.hpp - Prove two dynamic extents equal -----------------===//
//
// Equal type spelling says nothing about a dynamic axis: tensor<?x16xf16>
// can be [4, 16] on one side and [1, 16] on the other. These helpers follow
// an axis back through casts, projections, DPS destinations, reshapes,
// slices and tensor.empty sizes and collect every value/axis pair it is
// provably equal to. Two axes have one runtime extent when their sets meet.
// No arithmetic equivalence is assumed; unknown provenance stops the walk,
// so a missing fact makes the comparison fail rather than succeed.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Value.h"

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <optional>

namespace hip {
namespace fusion_transform {

// One axis: a static size, or axis `dim` of `value` (dim == -1: `value` is
// an index scalar of unknown provenance).
struct Extent {
  bool isStatic = false;
  int64_t staticSize = 0;
  mlir::Value value;
  int64_t dim = 0;

  bool operator==(const Extent &other) const {
    if (isStatic || other.isStatic)
      return isStatic && other.isStatic && staticSize == other.staticSize;
    return value == other.value && dim == other.dim;
  }
};

// A declared size (constant, tensor.dim, or an opaque scalar).
std::optional<Extent> shapeSizeExtent(mlir::OpFoldResult size);

// Every extent axis `dim` of `value` is proven equal to, nearest first.
llvm::SmallVector<Extent> traceExtents(mlir::Value value, int64_t dim);

// True when the two extent sets share an element.
bool extentsMeet(llvm::ArrayRef<Extent> lhs, llvm::ArrayRef<Extent> rhs);

// True when axis `lhsDim` of `lhs` and axis `rhsDim` of `rhs` have one
// runtime extent.
bool sameRuntimeExtent(mlir::Value lhs, int64_t lhsDim, mlir::Value rhs,
                       int64_t rhsDim);

// True when the declared `size` equals the extent of axis `dim` of `value`.
bool sizeIsExtent(mlir::OpFoldResult size, mlir::Value value, int64_t dim);

} // namespace fusion_transform
} // namespace hip
