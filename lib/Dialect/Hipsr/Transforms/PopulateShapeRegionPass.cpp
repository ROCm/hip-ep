/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PopulateShapeRegionPass.cpp - Fill placeholder shape regions -------===//
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrShapeRegionPopulation.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
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
    func::FuncOp funcOp = getOperation();
    ShapeRegionPopulationPatternSet patterns;
    populateHipsrShapeRegionPatterns(patterns);
    SmallVector<ShapeRegionPopulationPlan> plans;
    WalkResult result = funcOp.walk([&](PlaceholderOp placeholder) {
      if (!placeholder.getShapeRegion().empty()) {
        return WalkResult::advance();
      }

      FailureOr<ShapeRegionPopulationPlan> plan =
          planShapeRegionPopulation(placeholder, patterns);
      if (failed(plan)) {
        return WalkResult::interrupt();
      }
      plans.push_back(*plan);
      return WalkResult::advance();
    });
    if (result.wasInterrupted()) {
      signalPassFailure();
      return;
    }

    OpBuilder builder(&getContext());
    for (const ShapeRegionPopulationPlan &plan : plans) {
      if (failed(populateShapeRegion(plan, builder))) {
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
