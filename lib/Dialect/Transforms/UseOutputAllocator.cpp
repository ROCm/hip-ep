/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- UseOutputAllocator.cpp - returned allocs -> hip.alloc_output -------===//
//
// Turns each graph-output `memref.alloc` into a `hip.alloc_output`. A graph
// output is a buffer that func.return hands back -- either the alloc directly,
// or the alloc after it goes through view ops like `memref.cast`,
// `memref.collapse_shape`, `memref.expand_shape`, or `memref.subview` (any
// number of them, in any order). `hip.alloc_output` gets its buffer from the
// EP's output allocator, so the EP owns and frees it. Allocs that are NOT
// returned are left alone and get freed / pooled later like normal temporaries.
//
// For each matching alloc the rewrite:
//   - sets `out_idx` to its position in func.return (the graph output index),
//   - reuses the alloc's dynamic-size operands unchanged,
//   - deletes any `memref.dealloc` of it (the EP frees it, not us),
//   - leaves the view ops in place -- only the alloc op itself changes.
//
// How outputs are found. Rather than listing every view op by hand, the pass
// asks `BufferViewFlowAnalysis` one question per alloc: does any value derived
// from the alloc (through any chain of view ops) show up in func.return? That
// covers every case with no per-op special-casing -- a direct return, a single
// `memref.cast`, or a reshaped output such as a rank-4 conv result collapsed to
// rank-2 before the return.
//
// Only public (graph-entry) functions are rewritten. Private helpers (e.g.
// outlined `onnx.Loop` bodies) also take a `!hip.context` arg and return
// allocs, but those buffers stay inside the DLL and are not EP outputs --
// rewriting them would hand out an `out_idx` that clashes with the real
// outputs. Functions whose arg 0 is not `!hip.context` are skipped too (no
// runtime handle to pass to the new op). Pass-through outputs (a returned value
// that comes from a block argument or a view of an input, not from an alloc)
// are left alone. The function signature and the func.return are not touched
// here -- `convert-hip-to-llvm` builds the `-> i32` entry wrapper later.
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
//          : memref<?x?xf16>                            // EP-owned output
//     hip.sigmoid(%ctx) ins(%t) outs(%out)
//     return %out : memref<?x?xf16>
//   }
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "hip-use-output-allocator"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_USEOUTPUTALLOCATORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct UseOutputAllocatorPass
    : impl::UseOutputAllocatorPassBase<UseOutputAllocatorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    // The pass creates hip.alloc_output (HipDialect). BufferViewFlowAnalysis
    // follows the memref view ops (cast, collapse_shape, expand_shape, subview)
    // through their ViewLikeOpInterface, which MemRefDialect provides.
    registry.insert<hip::HipDialect, memref::MemRefDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Only public (graph-entry) functions own EP outputs. Private helpers (e.g.
    // outlined onnx.Loop bodies) also carry a !hip.context arg 0 and return
    // memref.allocs, but their outputs are DLL-internal -> skip non-public.
    if (!funcOp.isPublic() || funcOp.empty())
      return;

    // Need !hip.context arg 0 to build hip.alloc_output.
    if (funcOp.getNumArguments() == 0 ||
        !isa<ContextType>(funcOp.getArgument(0).getType()))
      return;
    Value ctx = funcOp.getArgument(0);

    // Build the alias analysis once for the whole function.
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

    // Phase 2 -- rewrite (IR mutation only; the analysis is no longer queried).
    OpBuilder builder(funcOp.getContext());
    for (auto [allocOp, outIdx] : outputs) {
      // The EP owns this buffer now, so drop any dealloc of it. A returned
      // buffer normally has none, but remove one if present -- the EP-owned
      // output must never be freed by the graph.
      for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          dealloc.erase();

      builder.setInsertionPoint(allocOp);
      auto allocOutput = AllocOutputOp::create(
          builder, allocOp.getLoc(), allocOp.getType(), ctx,
          allocOp.getDynamicSizes(), builder.getI64IntegerAttr(outIdx));
      allocOp.getResult().replaceAllUsesWith(allocOutput.getResult());
      allocOp.erase();
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
