/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegionPass.cpp - Fill hipsr shape regions -------------===//
//
// Walks every op implementing ShapeRegionInterface and, for each one whose
// shape region is empty (unpopulated), calls populateShapeRegion() so the op
// emits its own output-shape computation (the single source of truth). Ops
// with an already-populated region are skipped, so the pass is idempotent.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_POPULATESHAPEREGIONPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct PopulateShapeRegionPass
    : impl::PopulateShapeRegionPassBase<PopulateShapeRegionPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    OpBuilder builder(&getContext());
    // populateShapeRegion() only adds ops inside the op's own (previously
    // empty) shape region; none of those implement ShapeRegionInterface, so
    // the walk neither re-visits nor re-populates them.
    funcOp.walk([&](ShapeRegionInterface shapeRegionOp) {
      Region &shapeRegion = shapeRegionOp.getShapeRegion();
      if (shapeRegion.empty())
        shapeRegionOp.populateShapeRegion(builder, shapeRegion);
      // TODO: EndBarrier ops also carry a capacity shape region (region 1).
      // When such ops land, fill it too via getCapacityShapeRegion() /
      // populateCapacityShapeRegion(). No EndBarrier op exists today, so only
      // the shape region (region 0) is handled here.
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
