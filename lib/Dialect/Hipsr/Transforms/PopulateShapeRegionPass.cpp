/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegionPass.cpp - Fill hipsr shape regions -------------===//
//
// Each empty placeholder shape region is populated from its DPS consumer.
// Already-populated placeholders are skipped.
//
// Before:
//   %init = hipsr.placeholder(%ctx) ins(%input) : tensor<?x8xf16>
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init) : tensor<?x8xf16>
// After:
//   %init = hipsr.placeholder(%ctx) ins(%input) : tensor<?x8xf16>
//       shape_region {
//   ^bb0(%input_shape: !shape.shape):
//     hipsr.shape_yield2 %input_shape : !shape.shape
//   }
//   %0 = hipsr.cast(%ctx) ins(%input) outs(%init) : tensor<?x8xf16>
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace hipsr {

LogicalResult populateCastShapeRegion(OpBuilder &builder, Block &block,
                                      CastOp op);

#define GEN_PASS_DEF_POPULATESHAPEREGIONPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

Block &createPlaceholderShapeBlock(OpBuilder &builder,
                                   PlaceholderOp placeholder) {
  Region &shapeRegion = placeholder.getShapeRegion();
  Block &block = *builder.createBlock(&shapeRegion);
  for (Type type : placeholder.getShapeRegionArgumentTypes()) {
    block.addArgument(type, placeholder.getLoc());
  }
  return block;
}

LogicalResult populatePlaceholderShapeRegion(OpBuilder &builder,
                                             PlaceholderOp placeholder) {
  if (!placeholder.getShapeRegion().empty()) {
    return success();
  }

  Operation *consumer = placeholder.getDpsConsumer();
  if (!consumer) {
    return placeholder.emitOpError(
        "shape-region population requires a DPS consumer");
  }

  if (auto castOp = dyn_cast<CastOp>(consumer)) {
    if (placeholder.getPlaceholderType() != PlaceholderType::Normal) {
      return placeholder.emitOpError(
          "type does not match the consumer's shape-region category");
    }
    Block &block = createPlaceholderShapeBlock(builder, placeholder);
    return populateCastShapeRegion(builder, block, castOp);
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
