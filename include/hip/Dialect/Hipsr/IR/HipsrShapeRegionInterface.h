/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_INTERFACE_H
#define HIPSR_SHAPE_REGION_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

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

} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h.inc"

#endif // HIPSR_SHAPE_REGION_INTERFACE_H
