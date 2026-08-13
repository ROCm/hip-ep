/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegionPass.cpp - Fill hipsr shape regions -------------===//
//
// Each empty placeholder shape region is populated from the consumer whose
// `outs` it fills. Already-populated placeholders are skipped.
//
// Before:
//   %init = hipsr.placeholder(%ctx) ins(%input) : tensor<?x8xf16>
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init) : tensor<?x8xf16>
// After:
//   %init = hipsr.placeholder(%ctx) ins(%input) : tensor<?x8xf16>
//       shape_region {
//   ^bb0(%input_shape: !shape.shape):
//     hipsr.shape_yield %input_shape : !shape.shape
//   }
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init) : tensor<?x8xf16>
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulationUtils.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace hipsr {

LogicalResult populateAddShapeRegion(OpBuilder &builder, Block &block,
                                     AddOp op);
LogicalResult populateCastShapeRegion(OpBuilder &builder, Block &block,
                                      CastOp op);
LogicalResult populateExpandShapeRegion(OpBuilder &builder, Block &block,
                                        ExpandOp op);
LogicalResult populateMatMulShapeRegion(OpBuilder &builder, Block &block,
                                        MatMulOp op);

#define GEN_PASS_DEF_POPULATESHAPEREGIONPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

LogicalResult populatePlaceholderShapeRegion(OpBuilder &builder,
                                             PlaceholderOp placeholder) {
  if (!placeholder.getShapeRegion().empty()) {
    return success();
  }

  Operation *consumer = placeholder.getConsumer();
  Block &block = createPlaceholderShapeBlock(builder, placeholder);

  if (auto addOp = dyn_cast<AddOp>(consumer)) {
    return populateAddShapeRegion(builder, block, addOp);
  } else if (auto castOp = dyn_cast<CastOp>(consumer)) {
    return populateCastShapeRegion(builder, block, castOp);
  } else if (auto expandOp = dyn_cast<ExpandOp>(consumer)) {
    return populateExpandShapeRegion(builder, block, expandOp);
  } else if (auto matMulOp = dyn_cast<MatMulOp>(consumer)) {
    return populateMatMulShapeRegion(builder, block, matMulOp);
  }

  return placeholder.emitOpError(
             "shape-region population does not support consumer ")
         << consumer->getName();
}

struct PopulateShapeRegionPass
    : impl::PopulateShapeRegionPassBase<PopulateShapeRegionPass> {
  void runOnOperation() override {
    OpBuilder builder(&getContext());
    WalkResult walkResult = getOperation().walk([&](PlaceholderOp placeholder) {
      if (failed(populatePlaceholderShapeRegion(builder, placeholder))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
