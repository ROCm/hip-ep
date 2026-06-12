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
// Scope: ONLY public (graph-entry) functions are rewritten. Private helpers --
// e.g. outlined `onnx.Loop` bodies -- also carry a `!hip.context` arg 0 and
// return `memref.alloc`s, but their results are DLL-internal, never EP outputs;
// rewriting them would emit an `out_idx` colliding with the real graph outputs.
// Functions lacking `!hip.context` as argument 0 are likewise skipped (no
// runtime handle to build the new op). Intermediates (allocs NOT returned) and
// passthrough outputs (returns whose defining op is not a memref.alloc, e.g.
// block args / views) are left untouched. The function signature and the
// `func.return` terminator are intentionally NOT modified -- `convert-hip-to-
// llvm` synthesizes the `-> i32` entry wrapper in a later phase.
//
// The pass also stamps `hipdnn.use_output_allocator = true` on the parent
// module -- the marker that later phases (`convert-hip-to-llvm`,
// `generate-interface`) read to select the allocator ABI. It is a typed bool,
// not a presence-only UnitAttr, so readers test the VALUE (absence and
// `= false` both mean classic). The stamp is UNCONDITIONAL -- mode is decided
// by this pass running (it is scheduled only in allocator mode), not by whether
// a returned alloc was found, so a zero-output graph is still marked. A FuncOp
// pass writing its parent module relaxes MLIR's pass-isolation contract; safe
// here because it is an idempotent set of the same value (same precedent as
// PoolAllocs stamping hipdnn.pool_size).
//
// Before:
//   func.func @main_graph(%ctx: !hip.context, ...) -> memref<?x?xf16> {
//     %out = memref.alloc(%M, %N) : memref<?x?xf16>      // returned output
//     hip.sigmoid(%ctx) ins(%t) outs(%out)
//     return %out : memref<?x?xf16>
//   }
//
// After:
//   module attributes {hipdnn.use_output_allocator = true} {
//     func.func @main_graph(%ctx: !hip.context, ...) -> memref<?x?xf16> {
//       %out = hip.alloc_output(%ctx, %M, %N) {out_idx = 0 : i64}
//            : memref<?x?xf16>                            // EP-owned output
//       hip.sigmoid(%ctx) ins(%t) outs(%out)
//       return %out : memref<?x?xf16>
//     }
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "hip-use-output-allocator"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_USEOUTPUTALLOCATORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Rewrites a returned `memref.alloc` (in a public function with a
// `!hip.context` arg 0) into a `hip.alloc_output`, dropping any dealloc of the
// buffer (the EP owns it). An alloc returned at more than one operand position
// (aliased multi-output) matches once and is rewritten with its FIRST return
// index -- the one-match-per-alloc structure of the pattern is what dedupes it.
struct ReplaceOutputAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Find if this alloc is returned (graph output) and at which index.
    func::ReturnOp returnOp = nullptr;
    int64_t outIdx = -1;
    for (OpOperand &use : allocOp->getUses()) {
      if (auto ret = dyn_cast<func::ReturnOp>(use.getOwner())) {
        returnOp = ret;
        outIdx = use.getOperandNumber();
        break; // first occurrence (aliased outputs share one rewrite)
      }
    }
    if (!returnOp)
      return failure(); // not a graph output

    auto func = allocOp->getParentOfType<func::FuncOp>();
    // Only public (graph-entry) functions own EP outputs. Private helpers (e.g.
    // outlined onnx.Loop bodies) also carry a !hip.context arg 0 and return
    // memref.allocs, but their outputs are DLL-internal -> skip non-public.
    if (!func || !func.isPublic())
      return failure();

    // Need !hip.context arg 0 to build hip.alloc_output.
    if (func.getNumArguments() == 0 ||
        !isa<ContextType>(func.getArgument(0).getType()))
      return failure();
    Value ctx = func.getArgument(0);

    // EP owns the buffer now; drop any dealloc (it references the alloc
    // result).
    for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
        rewriter.eraseOp(dealloc);

    auto allocOutput = AllocOutputOp::create(
        rewriter, allocOp.getLoc(), allocOp.getType(), ctx,
        allocOp.getDynamicSizes(), rewriter.getI64IntegerAttr(outIdx));

    rewriter.replaceOp(allocOp, allocOutput.getResult());
    return success();
  }
};

struct UseOutputAllocatorPass
    : impl::UseOutputAllocatorPassBase<UseOutputAllocatorPass> {
  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceOutputAllocPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    // Stamp the allocator-mode marker (typed bool, value = true) on the parent
    // module. UNCONDITIONAL -- see the file header for why this must run even
    // when nothing was rewritten. Readers test the value, so absence and
    // `= false` both read as classic mode.
    if (auto module = getOperation()->getParentOfType<ModuleOp>())
      module->setAttr("hipdnn.use_output_allocator",
                      BoolAttr::get(&getContext(), true));
  }
};

} // namespace

} // namespace hip
} // namespace mlir
