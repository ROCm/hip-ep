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
//   - leaves the view ops in place -- only the alloc op itself changes,
//   - when the output is returned through a rank-reducing collapse_shape,
//     stamps `hipdnn.abi_shape` / `hipdnn.abi_groups` so the HIP->LLVM lowering
//     issues the output-allocator callback at the RETURNED (ONNX) rank rather
//     than the higher internal compute rank (see stampAbiCollapseAttrs).
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
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "hip-use-output-allocator"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_USEOUTPUTALLOCATORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// If the graph output is returned through a rank-reducing
// `memref.collapse_shape` (the internal compute buffer is higher-rank than the
// ONNX / func.return output -- e.g. a vision encoder produces
// memref<?x?x2560> internally but its `image_features` output is rank-2
// memref<?x2560>), stamp the collapse mapping onto `allocOutput` as the
// discardable attrs `hipdnn.abi_shape` + `hipdnn.abi_groups`. The HIP->LLVM
// lowering (`AllocOutputOpLowering`) reads them to emit the EP output-allocator
// callback at the RETURNED rank instead of the internal rank -- otherwise ORT
// rejects the pre-bound output OrtValue with a shape-mismatch
// ("has shape {252,2560} but the computed output shape ... is {1,252,2560}").
//
// Why here and not in the lowering: the reassociation must be read while
// collapse_shape is still intact. By the time `convert-hip-to-llvm` runs,
// `expand-strided-metadata` has decomposed collapse_shape into
// extract_strided_metadata + reinterpret_cast, which erases the reassociation
// and re-defines the external dims AFTER this alloc (referencing them there
// would break SSA dominance). Stamping static attrs now lets the lowering
// re-derive each external dim from the internal alloc sizes (which dominate).
//
// Only a single collapse (optionally wrapped by rank-preserving `memref.cast`)
// is handled. Any other view (expand_shape / subview / reinterpret_cast, a
// second collapse, or a non-view producer) leaves the attrs unset and the
// lowering keeps the internal rank -- unchanged behavior, no new regression.
//
// Before (IR reaching this pass; the alloc is already an EP output):
//   %out = hip.alloc_output(%ctx, %d0, %d1) {out_idx = 0}
//        : memref<?x?x2560xf16>
//   %r = memref.collapse_shape %out [[0, 1], [2]]
//        : memref<?x?x2560xf16> into memref<?x2560xf16>
//   return %r : memref<?x2560xf16>
// After (attrs added; op type + operands unchanged):
//   %out = hip.alloc_output(%ctx, %d0, %d1)
//            {out_idx = 0,
//             hipdnn.abi_shape  = array<i64: -9223372036854775808, 2560>,
//             hipdnn.abi_groups = array<i64: 2, 1>} : memref<?x?x2560xf16>
static void stampAbiCollapseAttrs(AllocOutputOp allocOutput, Value retVal,
                                  OpBuilder &builder) {
  Value root = allocOutput.getResult();
  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!rootType)
    return;

  memref::CollapseShapeOp collapse;
  Value cur = retVal;
  while (cur != root) {
    Operation *def = cur.getDefiningOp();
    if (!def)
      return; // block-arg / input passthrough -- no alloc-rooted view chain.
    if (auto c = dyn_cast<memref::CollapseShapeOp>(def)) {
      if (collapse)
        return; // more than one collapse on the path -- unsupported.
      collapse = c;
      cur = c.getSrc();
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      cur = castOp.getSource(); // rank-preserving -- dims unchanged.
      continue;
    }
    return; // expand_shape / subview / reinterpret_cast / other -- bail.
  }
  if (!collapse)
    return; // returned directly or via casts only -- rank already matches.

  auto extType = cast<MemRefType>(retVal.getType());

  // groups[e] = #consecutive internal dims collapsed into external dim e.
  // collapse_shape reassociation is contiguous and ordered, so the lowering
  // consumes the internal sizes sequentially with a running index.
  SmallVector<int64_t> groups;
  int64_t total = 0;
  for (ArrayRef<int64_t> g : collapse.getReassociationIndices()) {
    groups.push_back(static_cast<int64_t>(g.size()));
    total += static_cast<int64_t>(g.size());
  }

  // Guard the invariants the lowering relies on; bail (leave attrs unset) on
  // any mismatch rather than risk a wrong callback shape.
  if (total != rootType.getRank() ||
      static_cast<int64_t>(groups.size()) != extType.getRank() ||
      extType.getRank() == rootType.getRank())
    return;

  allocOutput->setAttr(kAbiShapeAttrName,
                       builder.getDenseI64ArrayAttr(extType.getShape()));
  allocOutput->setAttr(kAbiGroupsAttrName,
                       builder.getDenseI64ArrayAttr(groups));
}

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

    // The single func.return of this graph-entry function. Used to recover the
    // returned (ONNX ABI) value per output index for the collapse-shape ABI
    // adjustment below.
    func::ReturnOp returnOp;
    funcOp.walk([&](func::ReturnOp r) { returnOp = r; });

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

      // After RAUW the returned value's view chain is rooted at allocOutput.
      // Record the ONNX return shape when it is a rank-reduced collapse of the
      // internal compute buffer, so the lowering issues the output-allocator
      // callback at the returned rank (see stampAbiCollapseAttrs).
      if (returnOp && outIdx >= 0 &&
          outIdx < static_cast<int64_t>(returnOp.getNumOperands()))
        stampAbiCollapseAttrs(allocOutput, returnOp.getOperand(outIdx), builder);
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
