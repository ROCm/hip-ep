/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_PREPARELOOPBODYFAILURESPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct PrepareLoopBodyFailuresPass
    : impl::PrepareLoopBodyFailuresPassBase<PrepareLoopBodyFailuresPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    WalkResult result = module.walk([&](LoopOp loop) {
      auto body = module.lookupSymbol<func::FuncOp>(loop.getBodyFuncAttr());
      if (!body || body.empty())
        return WalkResult::advance();

      SmallVector<Operation *> checks;
      body.walk([&](Operation *op) {
        if (isa<LoopAllocOp, CopyOutputOp>(op))
          checks.push_back(op);
      });
      if (checks.empty())
        return WalkResult::advance();

      unsigned numCarriers = loop.getNumLoopCarried();
      unsigned carrierResultStart = loop.getCondIsPassthrough() ? 1u : 2u;
      if (body.getNumResults() != carrierResultStart + numCarriers) {
        body.emitOpError("cannot build allocation failure return for malformed "
                         "loop body result ABI");
        return WalkResult::interrupt();
      }

      Block *failure = body.addBlock();
      failure->addArgument(IntegerType::get(body.getContext(), 32),
                           body.getLoc());
      OpBuilder failureBuilder = OpBuilder::atBlockBegin(failure);
      SmallVector<Value> failureResults;
      failureResults.push_back(failure->getArgument(0));
      if (!loop.getCondIsPassthrough())
        failureResults.push_back(body.getArgument(2));
      for (unsigned i = 0; i < numCarriers; ++i)
        failureResults.push_back(body.getArgument(3 + i));
      func::ReturnOp::create(failureBuilder, body.getLoc(), failureResults);

      // Process in reverse textual order so splitting a block never moves an
      // allocation that has not yet been handled behind a newly inserted
      // terminator.
      for (Operation *checked : llvm::reverse(checks)) {
        Block *block = checked->getBlock();
        if (block->getParentOp() != body) {
          checked->emitOpError(
              "checked carrier operation must be in the body function region");
          return WalkResult::interrupt();
        }
        Block *continuation =
            block->splitBlock(std::next(Block::iterator(checked)));
        OpBuilder builder = OpBuilder::atBlockEnd(block);
        Value status;
        if (auto alloc = dyn_cast<LoopAllocOp>(checked))
          status = LoopFrameStatusOp::create(builder, alloc.getLoc(),
                                             alloc.getFrame());
        else
          status = cast<CopyOutputOp>(checked).getStatus();
        Value zero = arith::ConstantIntOp::create(builder, checked->getLoc(),
                                                  builder.getI32Type(), 0);
        Value ok = arith::CmpIOp::create(
            builder, checked->getLoc(), arith::CmpIPredicate::eq, status, zero);
        cf::CondBranchOp::create(builder, checked->getLoc(), ok, continuation,
                                 ValueRange{}, failure, ValueRange{status});
      }
      return WalkResult::advance();
    });
    if (result.wasInterrupted())
      signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
