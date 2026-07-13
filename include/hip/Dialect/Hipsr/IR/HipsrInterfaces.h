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
///
/// Design note: we use custom scoping verification instead of the
/// IsolatedFromAbove trait because shape regions need *selective* isolation:
/// - IsolatedFromAbove: region cannot access ANY outer value (full isolation).
/// - verifyShapeRegionScoping: region CAN access the op's operands, nothing
///   else.
///
/// Shape regions must read the op's inputs to compute output shapes, e.g.
/// `tensor.dim %input, %c0` where %input is the op's operand. Under
/// IsolatedFromAbove that would require threading every such value through
/// verbose region arguments. This design allows direct access to the op's
/// operands for ergonomics while still preventing accidental capture of
/// arbitrary outer-scope values.
::mlir::LogicalResult verifyShapeRegionScoping(::mlir::Operation *op);

} // namespace hipsr
} // namespace mlir

#include "hip/Dialect/Hipsr/IR/HipsrInterfaces.h.inc"

#endif // HIPSR_INTERFACES_H
