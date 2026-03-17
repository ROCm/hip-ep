/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/OnnxToHip/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HIPADDCONTEXTARGPASS
#include "hip/Conversion/OnnxToHip/Passes.h.inc"

namespace {

struct HipAddContextArgPass
    : public impl::HipAddContextArgPassBase<HipAddContextArgPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = &getContext();
    Type hipCtxType = ContextType::get(ctx);

    module.walk([&](func::FuncOp funcOp) {
      // Skip functions that already have a context argument.
      if (!funcOp.getArguments().empty() &&
          funcOp.getArgument(0).getType() == hipCtxType)
        return;

      // Insert the new argument at position 0.
      funcOp.insertArgument(0, hipCtxType, {}, funcOp.getLoc());

      // Update the function type to prepend !hip.context.
      FunctionType oldType = funcOp.getFunctionType();
      SmallVector<Type> newInputs;
      newInputs.push_back(hipCtxType);
      newInputs.append(oldType.getInputs().begin() + 1,
                       oldType.getInputs().end());
      funcOp.setType(FunctionType::get(ctx, newInputs, oldType.getResults()));
    });
  }
};

} // namespace

} // namespace hip
} // namespace mlir
