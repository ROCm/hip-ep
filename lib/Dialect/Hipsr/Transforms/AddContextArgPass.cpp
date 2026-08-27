/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AddContextArgPass.cpp - Insert !hipsr.context as function arg 0 ----===//
//
// Inserts !hipsr.context as the first argument of every func.func in the
// module and updates all func.call sites to forward the context.
//
// Two-phase approach:
//   Phase 1: Walk all func.func ops, insert !hipsr.context as arg 0.
//   Phase 2: Collect all func.call ops targeting updated functions,
//            then prepend the caller's !hipsr.context argument.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_ADDCONTEXTARGPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct AddContextArgPass : impl::AddContextArgPassBase<AddContextArgPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Type ctxType = ContextType::get(&getContext());

    DenseSet<StringRef> updatedNames;
    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getArguments().empty() &&
          funcOp.getArgument(0).getType() == ctxType) {
        return;
      }

      (void)funcOp.insertArgument(0, ctxType, {}, funcOp.getLoc());
      updatedNames.insert(funcOp.getName());
    });

    if (updatedNames.empty()) {
      return;
    }

    SmallVector<func::CallOp> callsToUpdate;
    module.walk([&](func::CallOp callOp) {
      if (updatedNames.contains(callOp.getCallee())) {
        callsToUpdate.push_back(callOp);
      }
    });

    for (func::CallOp callOp : callsToUpdate) {
      auto callerFunc = callOp->getParentOfType<func::FuncOp>();
      if (!callerFunc) {
        callOp.emitError("call op is not inside a func.func");
        return signalPassFailure();
      }
      if (callerFunc.getArguments().empty() ||
          !isa<ContextType>(callerFunc.getArgument(0).getType())) {
        callOp.emitError("caller @")
            << callerFunc.getName() << " does not have !hipsr.context as arg 0";
        return signalPassFailure();
      }

      Value callerCtx = callerFunc.getArgument(0);
      SmallVector<Value> newOperands = {callerCtx};
      newOperands.append(callOp.getOperands().begin(),
                         callOp.getOperands().end());
      OpBuilder builder(callOp);
      auto newCall =
          func::CallOp::create(builder, callOp.getLoc(), callOp.getCallee(),
                               callOp.getResultTypes(), newOperands);
      callOp.replaceAllUsesWith(newCall.getResults());
      callOp.erase();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
