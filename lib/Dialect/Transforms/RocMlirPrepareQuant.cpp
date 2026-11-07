/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "QLinearPlan.h"
#include "RebiasUnsignedQuantize.h"
#include "SinkQuantize.h"

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>

namespace mlir::hip {

#define GEN_PASS_DEF_ROCMLIRPREPAREQUANTPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct RocMlirPrepareQuantPass
    : public impl::RocMlirPrepareQuantPassBase<RocMlirPrepareQuantPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    impl::RocMlirPrepareQuantPassBase<
        RocMlirPrepareQuantPass>::getDependentDialects(registry);
    registerElementMoveExternalModels(registry);
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    if (funcOp.getSymName() != "main_graph")
      return;
    sinkQuantizeThroughElementMoves(funcOp);
    rebiasUnsignedQuantizedTrees(funcOp, isPlannedQLinear);
  }
};

} // namespace
} // namespace mlir::hip
