/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- UseOutputAllocator.cpp - returned allocs -> hip.alloc_output -------===//
//
// Rewrites each graph-output `memref.alloc` (a buffer returned by func.return)
// into a `hip.alloc_output`, so output buffers are obtained from the EP output
// allocator (EP/runtime-owned) instead of being deallocated / pooled like
// intermediates. `out_idx` is the operand position in func.return (= graph
// output index); the alloc's dynamic-size operands are reused verbatim.
//
// Requires `!hip.context` as function argument 0 (used to build the new op);
// functions without it are skipped. Intermediates (allocs NOT returned) and
// passthrough outputs (returns whose defining op is not a memref.alloc, e.g.
// block args / views) are left untouched. The function signature and the
// `func.return` terminator are intentionally NOT modified -- `convert-hip-to-
// llvm` synthesizes the `-> i32` entry wrapper in a later phase.
//
// Standalone: registered for hip-mlir-opt / LIT but NOT inserted into any
// pipeline (allocator-pipeline placement is a later phase; see
// docs/design/output-allocator-design.md, Phase 1).
//
// Before:
//   func.func @main_graph(%ctx: !hip.context, ...) -> memref<?x?xf16> {
//     %out = memref.alloc(%M, %N) : memref<?x?xf16>      // returned output
//     hip.sigmoid(%ctx) ins(%t) outs(%out)
//     return %out : memref<?x?xf16>
//   }
//
// After:
//   func.func @main_graph(%ctx: !hip.context, ...) -> memref<?x?xf16> {
//     %out = hip.alloc_output(%ctx, %M, %N) {out_idx = 0 : i64}
//          : memref<?x?xf16>                              // EP-owned output
//     hip.sigmoid(%ctx) ins(%t) outs(%out)
//     return %out : memref<?x?xf16>
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "hip-use-output-allocator"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_USEOUTPUTALLOCATORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct UseOutputAllocatorPass
    : impl::UseOutputAllocatorPassBase<UseOutputAllocatorPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    if (func.empty())
      return;

    // Need !hip.context arg 0 to build hip.alloc_output; skip gracefully if
    // absent (utility / pre-context-arg functions have no runtime handle).
    if (func.getNumArguments() == 0 ||
        !isa<ContextType>(func.getArgument(0).getType()))
      return;
    Value ctx = func.getArgument(0);

    // Collect returned memref.alloc outputs with their graph-output index (the
    // operand position in func.return). Dedupe on the op: an alloc returned at
    // more than one position (aliased multi-output) is rewritten exactly once
    // with its FIRST index, so we never erase the same op twice.
    SmallVector<std::pair<memref::AllocOp, int64_t>> work;
    llvm::SmallPtrSet<Operation *, 8> seen;
    func.walk([&](func::ReturnOp ret) {
      for (auto [idx, val] : llvm::enumerate(ret.getOperands()))
        if (auto alloc = val.getDefiningOp<memref::AllocOp>())
          if (seen.insert(alloc.getOperation()).second)
            work.emplace_back(alloc, static_cast<int64_t>(idx));
    });

    OpBuilder builder(func.getContext());
    for (auto [alloc, outIdx] : work) {
      builder.setInsertionPoint(alloc);
      auto allocOutput = AllocOutputOp::create(
          builder, alloc.getLoc(), alloc.getType(), ctx,
          alloc.getDynamicSizes(), builder.getI64IntegerAttr(outIdx));

      // EP owns the buffer now; drop any dealloc first (it references the alloc
      // result). Collect-then-erase to avoid iterator invalidation. Returned
      // values normally have no dealloc, but a buffer that is both returned and
      // used as a scratch intermediate could -- erase it so the EP-owned buffer
      // is never freed.
      SmallVector<memref::DeallocOp> deallocs;
      for (Operation *user : alloc->getUsers())
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          deallocs.push_back(dealloc);
      for (memref::DeallocOp dealloc : deallocs)
        dealloc.erase();

      alloc.replaceAllUsesWith(allocOutput.getResult());
      alloc.erase();
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
