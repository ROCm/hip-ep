/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_INTERFACES_H
#define HIPSR_INTERFACES_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace hipsr {

/// Verifies that every value used inside an op's shape region is either
/// defined within the region, a block argument of the region, or one of the
/// op's own operands. Referenced by ShapeRegionInterface's verifier.
::mlir::LogicalResult verifyShapeRegionScoping(::mlir::Operation *op);

} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrInterfaces.h.inc"

#endif // HIPSR_INTERFACES_H
