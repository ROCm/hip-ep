/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

// A shape built outside a recipe may still be dynamic, so the check skips a
// dynamic extent count.
LogicalResult PreserveShapeOp::verify() {
  int64_t extents = cast<RankedTensorType>(getShape().getType()).getDimSize(0);
  int64_t rank = cast<ShapedType>(getData().getType()).getRank();
  if (!ShapedType::isDynamic(extents) && extents != rank) {
    return emitOpError("shape holds ")
           << extents << " extents but data has rank " << rank;
  }
  return success();
}

// PreserveShapeOp does not modify memory, but it must report a side effect
// to avoid being eliminated as a trivially dead operation.
void PreserveShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}
