/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "udna-compiler/Conversion/OnnxToHip/Passes.h"
#include "udna-compiler/Dialect/Hip/IR/HipDialect.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace hip {

namespace {

/// Pass that inserts a `!hip.context` argument as argument index 0 into every
/// `func.func` in the module.  The OnnxToHip conversion patterns assume that
/// the context value is always available as the first function argument.
struct HipAddContextArgPass
    : public PassWrapper<HipAddContextArgPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(HipAddContextArgPass)

  StringRef getArgument() const override { return "hip-add-context-arg"; }
  StringRef getDescription() const override {
    return "Insert !hip.context as arg 0 into every func.func in the module";
  }

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

std::unique_ptr<Pass> createHipAddContextArgPass() {
  return std::make_unique<HipAddContextArgPass>();
}

} // namespace hip
} // namespace mlir
