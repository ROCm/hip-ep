/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegionPass.cpp - Fill hipsr shape regions -------------===//
//
// Walks every op implementing ShapeRegionInterface and, for each one whose
// shape region is empty (unpopulated), creates the entry block with one arg
// per DPS input (arg i mirrors input i -- the region is IsolatedFromAbove and
// cannot reach the op's operands), then calls populateShapeRegion() so the op
// fills its own output-shape computation (the single source of truth). Ops
// with an already-populated region are skipped, so the pass is idempotent.
//
// Before (region declared but empty):
//   %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
//                         outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
// After (entry block + args added here; body emitted by the op). Args mirror
// the DPS `ins` in order, so arg 0 is ctx and arg 1 the input:
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init)
//       : tensor<?x8xf16> shape_region {
//   ^bb0(%ctxarg: !hipsr.context, %in: tensor<?x8xf32>):
//     ... op-emitted shape math ...
//     hipsr.shape_yield (%d0, %d1) : [f16]
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionInterface.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

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
      if (shapeRegion.empty()) {
        // Mirror the DPS `ins` as entry-block args in order (arg i == input i);
        // the isolated body reads these instead of the op's operands.
        Operation *op = shapeRegionOp.getOperation();
        Block *block = builder.createBlock(&shapeRegion);
        if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
          for (OpOperand *in : dps.getDpsInputOperands()) {
            block->addArgument(in->get().getType(), op->getLoc());
          }
        }
        shapeRegionOp.populateShapeRegion(builder);
      }
      // TODO: EndBarrier ops also carry a capacity shape region (region 1).
      // When such ops land, create its entry block + args the same way and
      // fill it via getCapacityShapeRegion() / populateCapacityShapeRegion().
      // No EndBarrier op exists today, so only the shape region (region 0) is
      // handled here.
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
