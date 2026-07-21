/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_INTERFACE_H
#define HIPSR_SHAPE_REGION_INTERFACE_H

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"
#include "llvm/ADT/SmallVector.h"

// The generated interface verifier (in the .h.inc below) references the
// IsolatedFromAboveButAllowOperands trait type, so it must be declared first.
#include "hip/Dialect/Hipsr/IR/HipsrTraits.h"

namespace mlir {
namespace hipsr {

/// Verifies that the shape region (region 0) has exactly one block ending with
/// a `hipsr.shape_yield` terminator. Gives a clear error if an op implements
/// ShapeRegionInterface but forgets the
/// `SingleBlockImplicitTerminator<"ShapeYieldOp">` trait.
::mlir::LogicalResult verifyShapeRegionStructure(::mlir::Operation *op);

/// Read the `hipsr.shape_yield` at the end of `shapeRegion` and return each
/// result's dim values, grouped per result. The region must be populated.
::llvm::SmallVector<::llvm::SmallVector<::mlir::Value>>
getShapeRegionResultShapes(::mlir::Region &shapeRegion);

/// Read the `hipsr.shape_yield` at the end of `shapeRegion` and return each
/// result's tensor type. Every extent is dynamic; constant dims fold back to
/// static extents later via the tensor.empty canonicalizer. The region must be
/// populated.
::llvm::SmallVector<::mlir::RankedTensorType>
getShapeRegionResultTypes(::mlir::Region &shapeRegion);

} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h.inc"

#endif // HIPSR_SHAPE_REGION_INTERFACE_H
