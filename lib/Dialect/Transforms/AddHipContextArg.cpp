/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HIPADDCONTEXTARGPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct HipAddContextArgPass
    : public impl::HipAddContextArgPassBase<HipAddContextArgPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Type hipCtxType = ContextType::get(ctx);

    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getArguments().empty() &&
          funcOp.getArgument(0).getType() == hipCtxType)
        return;

      funcOp.insertArgument(0, hipCtxType, {}, funcOp.getLoc());
    });
  }
};

} // namespace

} // namespace hip
} // namespace mlir
