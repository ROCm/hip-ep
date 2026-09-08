/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

// PreserveShapeOp does not modify memory, but it must report a side effect
// to avoid being eliminated as a trivially dead operation.
void PreserveShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SideEffects::DefaultResource::get());
}

// An extent tensor or memref holds one entry per dimension of $data, so its
// length must equal the data rank. An opaque !shape.shape carries no length.
LogicalResult PreserveShapeOp::verify() {
  auto shapeType = dyn_cast<ShapedType>(getShape().getType());
  auto dataType = dyn_cast<ShapedType>(getData().getType());
  if (!shapeType || !dataType || !dataType.hasRank() ||
      shapeType.isDynamicDim(0)) {
    return success();
  }
  if (shapeType.getDimSize(0) != dataType.getRank()) {
    return emitOpError() << "extent count " << shapeType.getDimSize(0)
                         << " does not match data rank " << dataType.getRank();
  }
  return success();
}
