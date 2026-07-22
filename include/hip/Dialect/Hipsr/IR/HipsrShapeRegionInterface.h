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

// Accessors for the fixed region layout (region 0 = shape, 1 = capacity). Free
// functions so their definitions live once, not once per op as an interface
// method's body would. Forward-declared here; defined by the .h.inc below.
class ShapeRegionInterface;

::mlir::Region &getShapeRegion(ShapeRegionInterface op);

/// Region 1, present only on EndBarrier ops; a fatal error on any other op.
::mlir::Region &getCapacityShapeRegion(ShapeRegionInterface op);

/// Each result's dim values, grouped per result. Region must be populated.
::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
getShapeRegionResultShapes(ShapeRegionInterface op);

/// Each result's tensor type, all extents dynamic. Region must be populated.
::llvm::SmallVector<::mlir::RankedTensorType>
getShapeRegionResultTypes(ShapeRegionInterface op);

/// The two above over the capacity region (region 1).
::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
getCapacityShapeRegionResultShapes(ShapeRegionInterface op);

::llvm::SmallVector<::mlir::RankedTensorType>
getCapacityShapeRegionResultTypes(ShapeRegionInterface op);

} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h.inc"

#endif // HIPSR_SHAPE_REGION_INTERFACE_H
