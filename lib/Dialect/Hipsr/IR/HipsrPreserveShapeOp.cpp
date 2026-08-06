/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/BuiltinTypes.h"

using namespace mlir;
using namespace mlir::hipsr;

namespace {

struct ShapeMetadataResource
    : public SideEffects::Resource::Base<ShapeMetadataResource> {
  StringRef getName() final { return "hipsr::shape_metadata"; }
};

} // namespace

// PreserveShapeOp does not modify memory, but it must report a side effect
// to avoid being eliminated as a trivially dead operation.
void PreserveShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       ShapeMetadataResource::get());
}

LogicalResult PreserveShapeOp::verify() {
  auto fromExtents = getShape().getDefiningOp<shape::FromExtentsOp>();
  if (!fromExtents) {
    return success();
  }

  int64_t extents = static_cast<int64_t>(fromExtents.getExtents().size());
  int64_t dataRank = cast<ShapedType>(getData().getType()).getRank();
  if (extents != dataRank) {
    return emitOpError("shape has ")
           << extents << " extents but data has rank " << dataRank;
  }
  return success();
}
