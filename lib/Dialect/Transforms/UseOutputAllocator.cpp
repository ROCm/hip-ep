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
//   - allocates the EP-owned buffer at the GRAPH-OUTPUT type (the func.return
//     operand type), not the producer alloc's type. These differ when the
//     returned value is a reshaping view of the alloc (e.g. a rank-2 Gemm
//     result expanded to rank-3 with no trailing op); allocating at the alloc's
//     rank would make the EP report the wrong rank to ORT ("Invalid rank for
//     output"). When they differ, the reshaping view chain (cast / expand_shape
//     / collapse_shape, static output) is inverted so the producer keeps
//     writing into the same EP-owned memory viewed as its original type. Other
//     view ops (e.g. subview) or dynamic outputs fall back to allocating at the
//     alloc's type (legacy behavior).
//   - reuses the alloc's dynamic-size operands unchanged (fast/legacy path),
//   - deletes any `memref.dealloc` of it (the EP frees it, not us).
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
#include "mlir/Interfaces/ViewLikeInterface.h"

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

    // The single graph-entry return (graph functions have exactly one).
    func::ReturnOp retOp;
    funcOp.walk([&](func::ReturnOp r) { retOp = r; });

    // Phase 2 -- rewrite (IR mutation only; the analysis is no longer queried).
    OpBuilder builder(funcOp.getContext());
    for (auto [allocOp, outIdx] : outputs) {
      // The EP owns this buffer now, so drop any dealloc of it. A returned
      // buffer normally has none, but remove one if present -- the EP-owned
      // output must never be freed by the graph.
      for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          dealloc.erase();

      // The buffer the EP allocates and reports to ORT must have the GRAPH
      // OUTPUT shape (the func.return operand type), not the producer alloc's
      // shape. They differ when the returned value is a reshaping view of the
      // alloc (e.g. a rank-2 Gemm result expanded to rank-3 before the return,
      // with no trailing op). Allocating at the alloc's rank would make the EP
      // register the wrong rank with ORT -> "Invalid rank for output".
      Value retVal = retOp ? retOp.getOperand(outIdx) : Value();
      Type retTy = retVal ? retVal.getType() : allocOp.getType();

      builder.setInsertionPoint(allocOp);

      // Fast path: producer alloc already has the graph-output type (direct
      // return, or a shape-preserving view like memref.cast to same type).
      if (retTy == allocOp.getType()) {
        auto allocOutput = AllocOutputOp::create(
            builder, allocOp.getLoc(), allocOp.getType(), ctx,
            allocOp.getDynamicSizes(), builder.getI64IntegerAttr(outIdx));
        allocOp.getResult().replaceAllUsesWith(allocOutput.getResult());
        allocOp.erase();
        continue;
      }

      // Shape-changing view chain. Collect the reshaping ops from the returned
      // value down to the alloc. Only invertible metadata views (cast /
      // expand_shape / collapse_shape) over a static output are handled; any
      // other op (e.g. subview) or a dynamic output falls back to the legacy
      // behavior so we never regress an existing case.
      SmallVector<Operation *> chain;
      Value cur = retVal;
      bool invertible = mlir::cast<MemRefType>(retTy).hasStaticShape();
      while (invertible && cur != allocOp.getResult()) {
        Operation *def = cur.getDefiningOp();
        if (isa_and_nonnull<memref::CastOp, memref::ExpandShapeOp,
                            memref::CollapseShapeOp>(def)) {
          chain.push_back(def);
          cur = mlir::cast<ViewLikeOpInterface>(def).getViewSource();
        } else {
          invertible = false;
        }
      }

      if (!invertible) {
        // Legacy fallback: allocate at the producer alloc's type.
        auto allocOutput = AllocOutputOp::create(
            builder, allocOp.getLoc(), allocOp.getType(), ctx,
            allocOp.getDynamicSizes(), builder.getI64IntegerAttr(outIdx));
        allocOp.getResult().replaceAllUsesWith(allocOutput.getResult());
        allocOp.erase();
        continue;
      }

      // Allocate the EP-owned output at the graph-output (returned) type, then
      // rebuild the inverse of the view chain so the producer keeps writing
      // into the same memory viewed as its original (alloc) type.
      auto allocOutput = AllocOutputOp::create(
          builder, allocOp.getLoc(), retTy, ctx, /*dynamicSizes=*/ValueRange{},
          builder.getI64IntegerAttr(outIdx));

      Value inv = allocOutput.getResult();
      for (Operation *v : chain) { // outermost (produces retVal) -> innermost
        Type srcTy = mlir::cast<ViewLikeOpInterface>(v).getViewSource().getType();
        if (auto e = dyn_cast<memref::ExpandShapeOp>(v)) {
          // forward src -> larger rank; inverse collapses back to src.
          inv = memref::CollapseShapeOp::create(builder, v->getLoc(), srcTy, inv,
                                                e.getReassociationIndices());
        } else if (auto c = dyn_cast<memref::CollapseShapeOp>(v)) {
          // forward src -> smaller rank; inverse expands back to src (static).
          auto srcMr = mlir::cast<MemRefType>(srcTy);
          SmallVector<OpFoldResult> outShape;
          for (int64_t d : srcMr.getShape())
            outShape.push_back(builder.getIndexAttr(d));
          inv = memref::ExpandShapeOp::create(builder, v->getLoc(), srcTy, inv,
                                              c.getReassociationIndices(),
                                              outShape);
        } else { // memref::CastOp
          inv = memref::CastOp::create(builder, v->getLoc(), srcTy, inv);
        }
      }

      // `inv` now has the alloc's type and aliases the EP-owned buffer.
      allocOp.getResult().replaceAllUsesWith(inv);
      allocOp.erase();
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
