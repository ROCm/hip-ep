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
// Rank-adjusting outputs. An output alloc's shape can differ in RANK from what
// func.return hands back -- a rank-4 conv result collapsed to rank-2, or a flat
// buffer expanded to the ONNX rank. `convert-hip-to-llvm` lowers alloc_output
// so the EP allocator callback reports the op's own memref rank/shape to ORT,
// which IoBinding validates against the model's declared output. So the
// alloc_output MUST carry the EXTERNAL (func.return) type, not the internal
// compute type. When the ranks differ the rewrite allocates at the return type
// and hands the compute ops a `memref.reinterpret_cast` restored to the alloc
// type. One formula covers collapse AND expand alike: per reassociation group
// the total is the product of the internal dims in it, and the group's single
// dynamic external dim is that total divided by its static siblings (a group
// with more than one dynamic external dim is an underdetermined split and falls
// back to the direct path). Equal-rank returns (plain casts) are unchanged.
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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/STLExtras.h"

#define DEBUG_TYPE "hip-use-output-allocator"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_USEOUTPUTALLOCATORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// The alloc's per-dim sizes as mixed values: a static dim is an index attr, a
// dynamic dim is its size operand. Dim order matches the memref type.
static SmallVector<OpFoldResult> allocMixedSizes(OpBuilder &builder,
                                                 memref::AllocOp allocOp) {
  MemRefType type = allocOp.getType();
  SmallVector<OpFoldResult> sizes;
  sizes.reserve(type.getRank());
  unsigned dynIdx = 0;
  for (int64_t d = 0, e = type.getRank(); d < e; ++d) {
    if (type.isDynamicDim(d))
      sizes.push_back(allocOp.getDynamicSizes()[dynIdx++]);
    else
      sizes.push_back(builder.getIndexAttr(type.getDimSize(d)));
  }
  return sizes;
}

// a * b over index-typed fold results, constant-folded, with x*1 == x.
static OpFoldResult mulFold(OpBuilder &builder, Location loc, OpFoldResult a,
                            OpFoldResult b) {
  std::optional<int64_t> ca = getConstantIntValue(a);
  std::optional<int64_t> cb = getConstantIntValue(b);
  if (ca && cb)
    return builder.getIndexAttr(*ca * *cb);
  if (ca && *ca == 1)
    return b;
  if (cb && *cb == 1)
    return a;
  Value va = getValueOrCreateConstantIndexOp(builder, loc, a);
  Value vb = getValueOrCreateConstantIndexOp(builder, loc, b);
  return arith::MulIOp::create(builder, loc, va, vb).getResult();
}

// a / b over index-typed fold results, constant-folded, with x/1 == x. The
// division is always exact here (b is a product of static factors of a).
static OpFoldResult divFold(OpBuilder &builder, Location loc, OpFoldResult a,
                            OpFoldResult b) {
  std::optional<int64_t> ca = getConstantIntValue(a);
  std::optional<int64_t> cb = getConstantIntValue(b);
  if (cb && *cb == 1)
    return a;
  if (ca && cb && *cb != 0)
    return builder.getIndexAttr(*ca / *cb);
  Value va = getValueOrCreateConstantIndexOp(builder, loc, a);
  Value vb = getValueOrCreateConstantIndexOp(builder, loc, b);
  return arith::DivUIOp::create(builder, loc, va, vb).getResult();
}

// Solve the reassociation groups for the EXTERNAL (func.return) dynamic extents
// given the INTERNAL (alloc) sizes, appending them in result-dim order (the
// AllocOutputOp operand convention). ONE formula spans collapse and expand:
// getReassociationIndicesForReshape groups the higher-rank type's dims, one
// group per lower-rank dim, so for each group the total is the product of the
// internal dims it covers, and the group's single dynamic external dim is that
// total divided by the product of its static external siblings.
//   - collapse (internal higher): each group's external side is one dim, so its
//     "static siblings" product is 1 and the dim is just the internal product.
//   - expand (external higher): the group's external side is several dims; the
//     static ones come from the type and the one dynamic dim absorbs the rest.
// Returns false when a group has more than one dynamic external dim -- an
// underdetermined split the internal sizes alone cannot recover; the caller
// then keeps the direct (internal-type) path.
static bool solveExternalDynSizes(OpBuilder &builder, Location loc,
                                  MemRefType allocType, MemRefType returnType,
                                  ArrayRef<OpFoldResult> internalSizes,
                                  ArrayRef<ReassociationIndices> reassoc,
                                  SmallVectorImpl<Value> &externalDynSizes) {
  bool internalIsHigher = allocType.getRank() > returnType.getRank();
  SmallVector<OpFoldResult> externalSizes(returnType.getRank(), OpFoldResult());
  for (auto [k, group] : llvm::enumerate(reassoc)) {
    // The higher-rank side of this group is `group`; the lower-rank side is the
    // single dim `k`. Point internal/external at whichever side each is.
    SmallVector<int64_t, 4> internalDims, externalDims;
    if (internalIsHigher) {
      internalDims.assign(group.begin(), group.end());
      externalDims.push_back(static_cast<int64_t>(k));
    } else {
      internalDims.push_back(static_cast<int64_t>(k));
      externalDims.assign(group.begin(), group.end());
    }
    // Group total = product of the (all-known) internal dims in the group.
    OpFoldResult total = builder.getIndexAttr(1);
    for (int64_t i : internalDims)
      total = mulFold(builder, loc, total, internalSizes[i]);
    // Distribute the total across the external dims: statics come from the
    // type, the one dynamic dim = total / (product of the static siblings).
    int64_t staticProd = 1, dynDim = -1, dynCount = 0;
    for (int64_t f : externalDims) {
      if (returnType.isDynamicDim(f)) {
        ++dynCount;
        dynDim = f;
      } else {
        int64_t s = returnType.getDimSize(f);
        staticProd *= s;
        externalSizes[f] = builder.getIndexAttr(s);
      }
    }
    if (dynCount > 1)
      return false;
    if (dynCount == 1)
      externalSizes[dynDim] =
          divFold(builder, loc, total, builder.getIndexAttr(staticProd));
  }
  for (int64_t e = 0, r = returnType.getRank(); e < r; ++e)
    if (returnType.isDynamicDim(e))
      externalDynSizes.push_back(
          getValueOrCreateConstantIndexOp(builder, loc, externalSizes[e]));
  return true;
}

// View the EP output buffer (allocated at the external type, any rank) back as
// the internal alloc type so downstream compute ops see the shape they write.
// offset 0, row-major strides derived from the internal sizes. reinterpret_cast
// allows the source and result ranks to differ, so this restores both a
// collapsed (rank-up) and an expanded (rank-down) internal view uniformly.
static Value buildRankRestoringView(OpBuilder &builder, Location loc,
                                    Value buffer, MemRefType allocType,
                                    ArrayRef<OpFoldResult> internalSizes) {
  int64_t rank = allocType.getRank();
  SmallVector<OpFoldResult> strides(rank, builder.getIndexAttr(1));
  for (int64_t d = rank - 2; d >= 0; --d)
    strides[d] = mulFold(builder, loc, strides[d + 1], internalSizes[d + 1]);
  SmallVector<OpFoldResult> sizes(internalSizes.begin(), internalSizes.end());
  return memref::ReinterpretCastOp::create(builder, loc, allocType, buffer,
                                           /*offset=*/builder.getIndexAttr(0),
                                           sizes, strides)
      .getResult();
}

struct UseOutputAllocatorPass
    : impl::UseOutputAllocatorPassBase<UseOutputAllocatorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    // The pass creates hip.alloc_output (HipDialect). BufferViewFlowAnalysis
    // follows the memref view ops (cast, collapse_shape, expand_shape, subview)
    // through their ViewLikeOpInterface, which MemRefDialect provides. A
    // rank-adjusting output also needs memref.reinterpret_cast (MemRefDialect)
    // plus arith.muli / arith.divui (ArithDialect) for the extent math.
    registry.insert<hip::HipDialect, memref::MemRefDialect,
                    arith::ArithDialect>();
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
    ArrayRef<Type> resultTypes = funcOp.getFunctionType().getResults();
    for (auto [allocOp, outIdx] : outputs) {
      // The EP owns this buffer now, so drop any dealloc of it. A returned
      // buffer normally has none, but remove one if present -- the EP-owned
      // output must never be freed by the graph.
      for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          dealloc.erase();

      builder.setInsertionPoint(allocOp);
      Location loc = allocOp.getLoc();
      MemRefType allocType = allocOp.getType();
      // The external (ORT-visible) type is this output's func result type; it
      // equals the returned value's type (a view of the alloc).
      auto returnType = dyn_cast<MemRefType>(resultTypes[outIdx]);

      // A rank change means the returned view collapses or expands the alloc.
      // One reassociation describes either direction (the helper groups the
      // higher-rank type's dims); equal-rank returns need no shape rewrite.
      std::optional<SmallVector<ReassociationIndices>> reassoc;
      if (returnType && returnType != allocType &&
          returnType.getRank() != allocType.getRank())
        reassoc = getReassociationIndicesForReshape(allocType, returnType);

      Value replacement;
      if (reassoc) {
        SmallVector<OpFoldResult> internalSizes =
            allocMixedSizes(builder, allocOp);
        SmallVector<Value> externalDynSizes;
        if (solveExternalDynSizes(builder, loc, allocType, returnType,
                                  internalSizes, *reassoc, externalDynSizes)) {
          // Allocate the EP buffer at the EXTERNAL type so the runtime callback
          // reports the ORT-visible rank/shape; give compute ops a reinterpret
          // view restored to the internal alloc type.
          auto allocOutput = AllocOutputOp::create(
              builder, loc, returnType, ctx, externalDynSizes,
              builder.getI64IntegerAttr(outIdx));
          replacement = buildRankRestoringView(
              builder, loc, allocOutput.getResult(), allocType, internalSizes);
        }
      }
      if (!replacement) {
        // Equal-rank returns (plain casts) and unsupported splits keep the
        // internal type -- the alloc's own dynamic sizes carry over unchanged.
        auto allocOutput = AllocOutputOp::create(
            builder, loc, allocType, ctx, allocOp.getDynamicSizes(),
            builder.getI64IntegerAttr(outIdx));
        replacement = allocOutput.getResult();
      }
      allocOp.getResult().replaceAllUsesWith(replacement);
      allocOp.erase();
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
