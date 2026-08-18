/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "hipsr-use-output-allocator"
namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRUSEOUTPUTALLOCATORPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

static Value findVisibleContext(Operation *op) {
  Operation *anchor = op;
  while (Block *block = anchor->getBlock()) {
    for (BlockArgument arg : block->getArguments())
      if (isa<ContextType>(arg.getType()))
        return arg;

    Region *region = block->getParent();
    if (!region)
      return {};

    // Entry block arguments dominate all other blocks in the same region.
    if (!region->empty() && &region->front() != block)
      for (BlockArgument arg : region->front().getArguments())
        if (isa<ContextType>(arg.getType()))
          return arg;

    Operation *parentOp = region->getParentOp();
    if (!parentOp || parentOp->hasTrait<OpTrait::IsIsolatedFromAbove>())
      return {};

    anchor = parentOp;
  }

  return {};
}

struct HipsrUseOutputAllocatorPass
    : impl::HipsrUseOutputAllocatorPassBase<HipsrUseOutputAllocatorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hipsr::HipsrDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (!funcOp.isPublic() || funcOp.empty())
      return;

    if (funcOp.getNumArguments() == 0 ||
        !isa<ContextType>(funcOp.getArgument(0).getType()))
      return;

    BufferViewFlowAnalysis aliasAnalysis(funcOp);

    // Phase 1 -- classify (analysis only, no IR mutation). For each alloc that
    // is a graph output, record (alloc, out_idx). resolve() gives back the
    // alloc plus every value derived from it through view ops (cast / collapse
    // / expand / subview / ...), so a returned alloc is found even when it was
    // reshaped on the way to the return. getOperandNumber() on a func.return
    // use of any alias IS the graph output index; if the buffer is returned in
    // more than one slot (aliased multi-output), take the first (lowest).
    // Keeping every analysis query here -- before any rewrite -- means the
    // analysis (which caches Value handles) is never read after the IR it
    // describes has been mutated. Walk order is program order, so out_idx
    // values print in order in phase 2.
    SmallVector<std::pair<memref::AllocOp, int64_t>> outputs;
    funcOp.walk([&](memref::AllocOp allocOp) {
      int64_t outIdx = -1;
      for (Value aliased : aliasAnalysis.resolve(allocOp.getResult()))
        for (OpOperand &use : aliased.getUses())
          if (isa<func::ReturnOp>(use.getOwner())) {
            int64_t idx = static_cast<int64_t>(use.getOperandNumber());
            if (outIdx < 0 || idx < outIdx)
              outIdx = idx;
          }
      if (outIdx >= 0)
        outputs.emplace_back(allocOp, outIdx);
    });

    // hipsr.compute cannot pass through the ctx argument.
    // So we need to find the visible context for each output.
    SmallVector<Value> outputContexts;
    outputContexts.reserve(outputs.size());
    for (const auto &output : outputs) {
      memref::AllocOp allocOp = output.first;
      Value ctx = findVisibleContext(allocOp);
      if (!ctx) {
        allocOp.emitError()
            << "cannot find a visible !hipsr.context for graph output "
               "allocation";
        signalPassFailure();
        return;
      }
      outputContexts.push_back(ctx);
    }

    // Phase 2 -- rewrite (IR mutation only; the analysis is no longer queried).
    OpBuilder builder(funcOp.getContext());
    for (auto [index, output] : llvm::enumerate(outputs)) {
      auto [allocOp, outIdx] = output;
      // The EP owns this buffer now, so drop any dealloc of it. A returned
      // buffer normally has none, but remove one if present -- the EP-owned
      // output must never be freed by the graph.
      for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          dealloc.erase();

      builder.setInsertionPoint(allocOp);
      auto allocOutput = hipsr::AllocOutputOp::create(
          builder, allocOp.getLoc(), allocOp.getType(), outputContexts[index],
          allocOp.getDynamicSizes(), builder.getI64IntegerAttr(outIdx));
      allocOp.getResult().replaceAllUsesWith(allocOutput.getResult());
      allocOp.erase();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir