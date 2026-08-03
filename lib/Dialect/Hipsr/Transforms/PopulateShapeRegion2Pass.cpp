/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegion2Pass.cpp - Fill placeholder shape regions ------===//
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace hipsr {

LogicalResult populateCastShapeRegion(OpBuilder &builder, Block &block,
                                      CastOp op);

#define GEN_PASS_DEF_POPULATESHAPEREGION2PASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

Block &createBlockWithArguments(OpBuilder &builder, PlaceholderOp placeholder) {
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
        "shape-region population 2 requires a DPS consumer");
  }
  Block &block = createBlockWithArguments(builder, placeholder);

  if (auto castOp = dyn_cast<CastOp>(consumer)) {
    if (placeholder.getPlaceholderType() != PlaceholderType::Normal) {
      return placeholder.emitOpError(
          "type does not match the consumer's shape-region category");
    }
    return populateCastShapeRegion(builder, block, castOp);
  }

  return placeholder.emitOpError(
             "shape-region population 2 does not support consumer ")
         << consumer->getName();
}

struct PopulateShapeRegion2Pass
    : impl::PopulateShapeRegion2PassBase<PopulateShapeRegion2Pass> {
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
