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

// The resource hipsr.preserve_shape writes to. Its own, rather than the default
// one, because the write is not about memory: the op needs exactly enough of an
// effect that `wouldOpBeTriviallyDead` says no, and nothing more. A write on
// the default resource would order against every other memory op in the block,
// and a write naming the data operand would claim the op touches that buffer,
// which it does not.
struct ShapeMetadataResource
    : public SideEffects::Resource::Base<ShapeMetadataResource> {
  StringRef getName() final { return "hipsr::shape_metadata"; }
};

} // namespace

void PreserveShapeOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       ShapeMetadataResource::get());
}

LogicalResult PreserveShapeOp::verify() {
  // !shape.shape is rank erased, so the rank agreement this op documents is
  // only checkable when the extent list is right there in the IR. A shape that
  // comes out of an scf.execute_region -- the form the intended producer builds
  // -- lands in the unchecked case.
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
