/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_INTERFACE_H
#define HIPSR_SHAPE_REGION_INTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
// Supplies the builtin IsIsolatedFromAbove trait the interface verifier checks.
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

/// Verifies the shape region (region 0): either empty, or a single block whose
/// arguments mirror the op's DPS inputs and that ends in `hipsr.shape_yield`.
::mlir::LogicalResult verifyShapeRegionStructure(::mlir::Operation *op);

namespace detail {
// Cores backing the interface's result accessors. They take a bare region so
// one body serves both the shape region (0) and the capacity region (1), and
// both require the region already populated (block ends with
// hipsr.shape_yield). Prefer the op-bound accessors
// (op.getShapeRegionResultShapes() etc.); call these only when you already hold
// a Region &.
::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
getShapeRegionResultShapes(::mlir::Region &shapeRegion);

::llvm::SmallVector<::mlir::RankedTensorType>
getShapeRegionResultTypes(::mlir::Region &shapeRegion);
} // namespace detail

} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h.inc"

#endif // HIPSR_SHAPE_REGION_INTERFACE_H
