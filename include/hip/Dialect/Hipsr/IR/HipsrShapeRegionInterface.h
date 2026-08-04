/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_INTERFACE_H
#define HIPSR_SHAPE_REGION_INTERFACE_H

#include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.h"
#include "hip/Dialect/Hipsr/IR/HipsrStartBarrierInterface.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinTypes.h"
// Supplies the builtin IsIsolatedFromAbove trait the interface verifier checks.
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

/// Verifies region 0: empty, or a single block whose args match
/// `getShapeRegionArgOperands` and that ends in `hipsr.shape_yield`.
::mlir::LogicalResult verifyShapeRegionStructure(::mlir::Operation *op);

class ShapeRegionInterface;

/// The op operands that become the shape region's entry-block arguments, in
/// order. The isolated region reads these as block args. Keyed on the op's
/// barrier category:
///   Regular        -> data ins            (ctx dropped: shape is input-driven)
///   StartBarrier   -> ctx + data ins      (reads input data at runtime)
///   EndBarrier     -> data ins + outs     (shape comes from output data)
::llvm::SmallVector<::mlir::Value>
getShapeRegionArgOperands(ShapeRegionInterface op);

/// Region 0, the shape region; present on every ShapeRegionInterface op.
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
