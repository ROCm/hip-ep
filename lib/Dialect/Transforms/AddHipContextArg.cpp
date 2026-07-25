/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- AddHipContextArg.cpp - Insert !hip.context as function arg 0 ------===//
//
// Inserts !hip.context as the first argument of every func.func in the
// module and updates all func.call sites to forward the context.
//
// Two-phase approach (following BufferResultsToOutParams pattern):
//   Phase 1: Walk all func.func ops, insert !hip.context as arg 0.
//   Phase 2: Collect all func.call ops targeting updated functions,
//            then prepend the caller's !hip.context argument.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-add-context-arg"

STATISTIC(NumFuncsUpdated, "Number of functions given !hip.context argument");
STATISTIC(NumCallsUpdated, "Number of call sites updated with !hip.context");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HIPADDCONTEXTARGPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct HipAddContextArgPass
    : public impl::HipAddContextArgPassBase<HipAddContextArgPass> {

  void runOnOperation() override {
    ModuleOp module = getOperation();
    Type hipCtxType = ContextType::get(&getContext());

    // Phase 1: update function signatures.
    DenseSet<StringRef> updatedNames;
    module.walk([&](func::FuncOp funcOp) {
      if (!funcOp.getArguments().empty() &&
          funcOp.getArgument(0).getType() == hipCtxType)
        return;

      (void)funcOp.insertArgument(0, hipCtxType, {}, funcOp.getLoc());
      updatedNames.insert(funcOp.getName());
      ++NumFuncsUpdated;
      LLVM_DEBUG(llvm::dbgs()
                 << "  Added !hip.context to @" << funcOp.getName() << "\n");
    });

    if (updatedNames.empty())
      return;

    // Phase 2: collect call sites, then update.  Collect-then-modify avoids
    // iterator invalidation: erasing a CallOp during a walk over the module
    // would invalidate the walk's internal iterator.
    SmallVector<func::CallOp> callsToUpdate;
    module.walk([&](func::CallOp callOp) {
      if (updatedNames.contains(callOp.getCallee()))
        callsToUpdate.push_back(callOp);
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
            << callerFunc.getName() << " does not have !hip.context as arg 0";
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
      ++NumCallsUpdated;
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
