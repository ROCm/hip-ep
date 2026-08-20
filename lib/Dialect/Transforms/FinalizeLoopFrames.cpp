/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/BufferUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FINALIZELOOPFRAMESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct FinalizeLoopFramesPass
    : impl::FinalizeLoopFramesPassBase<FinalizeLoopFramesPass> {
  void runOnOperation() override {
    func::FuncOp function = getOperation();
    if (function.empty())
      return;
    if (!function.getBody().hasOneBlock()) {
      function.emitOpError(
          "explicit loop-frame lifetime currently requires one function block");
      return signalPassFailure();
    }

    BufferViewFlowAnalysis aliases(function);
    SmallVector<LoopOp> loops;
    function.walk([&](LoopOp loop) {
      unsigned n = loop.getNumLoopCarried();
      if (loop.getNumResults() == n + 1 &&
          isa<LoopFrameType>(loop.getResult(n).getType()))
        loops.push_back(loop);
    });

    for (LoopOp loop : loops) {
      unsigned n = loop.getNumLoopCarried();
      bool escapes = false;
      Operation *lastUse = loop;
      for (unsigned i = 0; i < n; ++i) {
        for (Value alias :
             resolveAliasesIncludingLoopMayAlias(loop.getResult(i), aliases)) {
          for (OpOperand &use : alias.getUses()) {
            Operation *owner = use.getOwner();
            if (isa<func::ReturnOp>(owner)) {
              escapes = true;
              continue;
            }
            if (owner->getBlock() != loop->getBlock()) {
              loop.emitOpError(
                  "carrier descriptor use crosses a block boundary; cannot "
                  "place frame destruction safely");
              return signalPassFailure();
            }
            if (lastUse == loop || lastUse->isBeforeInBlock(owner))
              lastUse = owner;
          }
        }
      }

      if (escapes) {
        // Outlined loop bodies transfer escaping child-frame ownership to the
        // parent frame. Public graph outputs must have been copied to
        // hip.alloc_output by this point and therefore may not still escape.
        if (function.isPublic() || function.getNumArguments() == 0 ||
            !isa<LoopFrameType>(function.getArgumentTypes().back())) {
          loop.emitOpError(
              "loop carrier escapes without parent-frame ownership "
              "or an exact graph-output copy");
          return signalPassFailure();
        }
        continue;
      }

      OpBuilder builder(lastUse);
      builder.setInsertionPointAfter(lastUse);
      LoopFrameDestroyOp::create(builder, loop.getLoc(), loop.getCtx(),
                                 loop.getResult(n));
    }
  }
};

} // namespace
} // namespace hip
} // namespace mlir
