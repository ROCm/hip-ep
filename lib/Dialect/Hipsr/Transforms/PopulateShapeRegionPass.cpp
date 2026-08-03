/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegionPass.cpp - Fill hipsr shape regions -------------===//
//
// For each ShapeRegionInterface op with an empty shape region, adds the entry
// block with the op's args (getShapeRegionArgOperands) and calls
// populateShapeRegion() to emit the output-shape computation. Idempotent:
// already-populated regions are skipped.
//
// Before:
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init) : tensor<?x8xf16>
// After:
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init)
//       : tensor<?x8xf16> shape_region {
//   ^bb0(%in: tensor<?x8xf32>):
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
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
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
    funcOp.walk([&](ShapeRegionInterface shapeRegionOp) {
      Region &shapeRegion = getShapeRegion(shapeRegionOp);
      if (!shapeRegion.empty()) {
        return;
      }

      Operation *op = shapeRegionOp.getOperation();
      Block *block = builder.createBlock(&shapeRegion);
      for (Value operand : getShapeRegionArgOperands(shapeRegionOp)) {
        block->addArgument(operand.getType(), op->getLoc());
      }
      shapeRegionOp.populateShapeRegion(builder, *block);

      // TODO: EndBarrier ops also need their capacity region (region 1) filled.
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
