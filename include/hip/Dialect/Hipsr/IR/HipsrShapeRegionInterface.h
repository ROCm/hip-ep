/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIPSR_SHAPE_REGION_INTERFACE_H
#define HIPSR_SHAPE_REGION_INTERFACE_H

#include "mlir/IR/OpDefinition.h"
#include "mlir/Support/LogicalResult.h"

namespace mlir {
namespace hipsr {

/// Verifies that the shape region (region 0) has exactly one block that ends
/// with a `hipsr.shape_yield` terminator. This is a defensive check for the
/// case where an op implements ShapeRegionInterface but forgets to add the
/// `SingleBlockImplicitTerminator<"ShapeYieldOp">` trait: it produces a clear
/// error instead of letting a wrong terminator crash downstream code that
/// assumes shape_yield. Runs before verifyShapeRegionScoping.
::mlir::LogicalResult verifyShapeRegionStructure(::mlir::Operation *op);

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

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h.inc"

#endif // HIPSR_SHAPE_REGION_INTERFACE_H
