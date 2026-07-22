/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegionPass.cpp - Fill hipsr shape regions -------------===//
//
// For each op implementing ShapeRegionInterface whose shape region is empty,
// creates the entry block with one arg per DPS input, then calls
// populateShapeRegion() so the op emits its own output-shape computation.
// Populated regions are skipped, so the pass is idempotent.
//
// Before (region declared but empty):
//   %0 = hipsr.cast(%ctx) ins(%input : tensor<?x8xf32>)
//                         outs(%init : tensor<?x8xf16>) : tensor<?x8xf16>
// After (entry block + args added here; body emitted by the op):
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
    funcOp.walk([&](ShapeRegionInterface shapeRegionOp) {
      Region &shapeRegion = getShapeRegion(shapeRegionOp);
      if (!shapeRegion.empty()) {
        return;
      }

      // One block arg per DPS input, in order; the isolated region reads these
      // instead of the op's operands.
      Operation *op = shapeRegionOp.getOperation();
      Block *block = builder.createBlock(&shapeRegion);
      if (auto dps = dyn_cast<DestinationStyleOpInterface>(op)) {
        for (OpOperand *in : dps.getDpsInputOperands())
          block->addArgument(in->get().getType(), op->getLoc());
      }
      shapeRegionOp.populateShapeRegion(builder, *block);

      // TODO: when EndBarrier ops land, set up their capacity region (region 1)
      // the same way and fill it via populateCapacityShapeRegion().
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
