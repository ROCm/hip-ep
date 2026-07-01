/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- UseOutputAllocator.cpp - returned allocs -> hip.alloc_output -------===//
//
// Every graph output must come from the EP's output allocator, so the runtime
// (not the model) owns its buffer. This pass makes sure that is true for every
// value the graph returns.
//
// `hip.alloc_output` is the op that asks the EP for an output buffer. Its
// `out_idx` is the output's position in `func.return` (= the graph output
// index). Any dynamic sizes it needs are passed as operands.
//
// The pass runs in two steps:
//
//   Step 1 (ReplaceOutputAllocPattern): if a returned buffer is a plain
//   `memref.alloc` (returned directly, or through one `memref.cast` that only
//   re-labels a dim as dynamic to match the ONNX output type), turn that alloc
//   into a `hip.alloc_output` and drop any dealloc of it. The cast, if present,
//   stays and keeps feeding the return.
//
//   Step 2 (serveUnbackedOutputs): every OTHER returned value is not backed by
//   its own output buffer yet -- e.g. a returned function input (passthrough),
//   a reshape/collapse/expand view, a constant, or the same buffer returned at
//   more than one index. For each of these, make a fresh `hip.alloc_output` for
//   that output index and `memref.copy` the value into it. Without this the EP
//   allocator callback would never fire for that output and the runtime aborts
//   (it checks that every output index is served exactly once).
//
// Scope: only public (graph-entry) functions are touched. Private helpers (e.g.
// outlined `onnx.Loop` bodies) also take a `!hip.context` arg 0 and return
// allocs, but their buffers stay inside the DLL and are never EP outputs, so
// rewriting them would hand out a wrong `out_idx`. Functions without a
// `!hip.context` first argument are skipped too (no runtime handle to build the
// op). Allocs that are not returned (intermediates) are left alone. The
// function signature and the number of `func.return` operands are not changed
// here -- `convert-hip-to-llvm` builds the `-> i32` entry wrapper later.
//
// Before:
//   func.func @main_graph(%ctx: !hip.context, %in: memref<?x?xf16>)
//       -> (memref<?x?xf16>, memref<?x?xf16>) {
//     %out = memref.alloc(%M, %N) : memref<?x?xf16>      // computed output 0
//     hip.sigmoid(%ctx) ins(%in) outs(%out)
//     return %out, %in : memref<?x?xf16>, memref<?x?xf16> // out 1 == input
//   }
//
// After:
//   func.func @main_graph(%ctx: !hip.context, %in: memref<?x?xf16>)
//       -> (memref<?x?xf16>, memref<?x?xf16>) {
//     %out = hip.alloc_output(%ctx, %M, %N) {out_idx = 0 : i64}  // Step 1
//          : memref<?x?xf16>
//     hip.sigmoid(%ctx) ins(%in) outs(%out)
//     %d0 = memref.dim %in, %c0 ; %d1 = memref.dim %in, %c1       // Step 2:
//     %pt = hip.alloc_output(%ctx, %d0, %d1) {out_idx = 1 : i64}  //
//     passthrough
//         : memref<?x?xf16>
//     memref.copy %in, %pt
//     return %out, %pt : memref<?x?xf16>, memref<?x?xf16>
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
#include "llvm/ADT/Sequence.h"

#define DEBUG_TYPE "hip-use-output-allocator"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_USEOUTPUTALLOCATORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Step 1: turn a returned `memref.alloc` (in a public function with a
// `!hip.context` arg 0) into a `hip.alloc_output`, and drop any dealloc of it
// (the EP owns the buffer now). If the same alloc is returned at more than one
// index, this matches it once and uses its FIRST return index; one match per
// alloc is what keeps it from being rewritten twice.
struct ReplaceOutputAllocPattern : public OpRewritePattern<memref::AllocOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::AllocOp allocOp,
                                PatternRewriter &rewriter) const override {
    // Is this alloc returned (a graph output), and at which index? It may be
    // returned DIRECTLY, or through one `memref.cast` that only re-labels a dim
    // as dynamic to match the ONNX output type (e.g. a matmul whose middle dim
    // is known to be 256 in-graph, `memref<?x256x2560>`, but declared
    // `memref<?x?x2560>` in the ONNX output). The cast only changes the shape
    // label, not the data, so the alloc is still the real output buffer and
    // must become a `hip.alloc_output`. If we miss it, the buffer gets pooled
    // or freed and the EP allocator callback never fires for that output
    // (runtime aborts: "output ... was never allocated by the model.dll").
    func::ReturnOp returnOp = nullptr;
    int64_t outIdx = -1;
    for (OpOperand &use : allocOp->getUses()) {
      Operation *owner = use.getOwner();
      if (auto ret = dyn_cast<func::ReturnOp>(owner)) {
        returnOp = ret;
        outIdx = use.getOperandNumber();
        break; // first use wins (same buffer returned twice shares one rewrite)
      }
      if (auto castOp = dyn_cast<memref::CastOp>(owner)) {
        for (OpOperand &castUse : castOp->getUses()) {
          if (auto ret = dyn_cast<func::ReturnOp>(castUse.getOwner())) {
            returnOp = ret;
            outIdx = castUse.getOperandNumber();
            break;
          }
        }
        if (returnOp)
          break;
      }
    }
    if (!returnOp)
      return failure(); // not a graph output

    auto func = allocOp->getParentOfType<func::FuncOp>();
    // Only public (graph-entry) functions have EP outputs. Private helpers
    // (e.g. outlined onnx.Loop bodies) also take a !hip.context arg 0 and
    // return allocs, but their buffers stay inside the DLL -> skip non-public.
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

// Step 2: give an output buffer to every returned value that Step 1 did not
// already handle (Step 1 only rewrites returned `memref.alloc`s). For each such
// `func.return` operand, make a fresh `hip.alloc_output` for that output index,
// `memref.copy` the returned value into it, and point the return at the new
// buffer. This covers a returned input (passthrough), a reshape/collapse/expand
// view, a constant, and the same buffer returned at a second index. Does
// nothing when every output already has its own matching-index
// `hip.alloc_output` (the common case).
static void serveUnbackedOutputs(func::FuncOp func) {
  // Only public (graph-entry) functions have EP outputs; need !hip.context arg
  // 0 to build hip.alloc_output. Same check as Step 1.
  if (!func.isPublic() || func.getNumArguments() == 0 ||
      !isa<ContextType>(func.getArgument(0).getType()))
    return;
  Value ctx = func.getArgument(0);

  func.walk([&](func::ReturnOp ret) {
    // Snapshot operands first: the loop mutates the return in place.
    SmallVector<Value> operands(ret.getOperands());
    OpBuilder b(ret);
    for (auto [i, val] : llvm::enumerate(operands)) {
      auto memTy = dyn_cast<MemRefType>(val.getType());
      if (!memTy)
        continue; // graph outputs are always memrefs; defensively skip others
      // Already has its own matching-index alloc_output? Then leave it be. This
      // covers Step 1's direct rewrite AND its cast path: when the output is
      // returned through a `memref.cast`, Step 1 turns the alloc under the cast
      // into alloc_output and keeps the cast feeding the return, so look
      // through one leading cast before checking. If we skip this look-through
      // we make a SECOND alloc_output for the same output index (the cast
      // result is not itself an alloc_output), and the runtime aborts because
      // it sees that output filled twice.
      Value backing = val;
      if (auto castOp = backing.getDefiningOp<memref::CastOp>())
        backing = castOp.getSource();
      if (auto ao = backing.getDefiningOp<AllocOutputOp>())
        if (ao.getOutIdx() == static_cast<int64_t>(i))
          continue;

      // Dynamic output extents come from the returned value itself (its type
      // equals the function result type at this index, so dims line up).
      SmallVector<Value> dynSizes;
      for (int64_t d : llvm::seq<int64_t>(0, memTy.getRank()))
        if (memTy.isDynamicDim(d))
          dynSizes.push_back(memref::DimOp::create(b, ret.getLoc(), val, d));

      auto out = AllocOutputOp::create(b, ret.getLoc(), memTy, ctx, dynSizes,
                                       b.getI64IntegerAttr(i));
      memref::CopyOp::create(b, ret.getLoc(), val, out.getResult());
      ret.setOperand(static_cast<unsigned>(i), out.getResult());
    }
  });
}

struct UseOutputAllocatorPass
    : impl::UseOutputAllocatorPassBase<UseOutputAllocatorPass> {
  void runOnOperation() override {
    // Step 1: returned memref.alloc -> hip.alloc_output.
    RewritePatternSet patterns(&getContext());
    patterns.add<ReplaceOutputAllocPattern>(&getContext());
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();

    // Step 2: give every remaining output (passthrough / view / shared) its own
    // alloc_output + copy.
    serveUnbackedOutputs(getOperation());
  }
};

} // namespace

} // namespace hip
} // namespace mlir
