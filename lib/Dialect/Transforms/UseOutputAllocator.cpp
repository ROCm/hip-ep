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
//   - when the output is returned through a rank-changing view (collapse_shape
//     or expand_shape), stamps `hipdnn.abi_shape` / `hipdnn.abi_groups` so the
//     HIP->LLVM lowering issues the output-allocator callback at the returned
//     graph rank rather than the internal compute rank (see
//     stampAbiReshapeAttrs).
//
// How outputs are found. Rather than listing every view op by hand, the pass
// asks `BufferViewFlowAnalysis` one question per alloc: does any value derived
// from the alloc (through any chain of view ops) show up in func.return? That
// covers every case with no per-op special-casing -- a direct return, a single
// `memref.cast`, or a reshaped output such as a rank-4 conv result collapsed to
// rank-2 before the return.
//
// Only public (graph-entry) functions are rewritten. Private helper functions
// may also take a `!hip.context` arg and return allocs, but those buffers stay
// inside the DLL and are not EP outputs --
// rewriting them would hand out an `out_idx` that clashes with the real
// outputs. Functions whose arg 0 is not `!hip.context` are skipped too (no
// runtime handle to pass to the new op). A pass-through output (a returned
// value that comes from a block argument or a view of an input, not from an
// alloc) is copied into a fresh `hip.alloc_output`: the EP output allocator
// requires every graph output to be allocated in-graph, and an aliased input
// never calls that callback. The function signature is not touched here --
// `convert-hip-to-llvm` builds the `-> i32` entry wrapper later.
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
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dominance.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#define DEBUG_TYPE "hip-use-output-allocator"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_USEOUTPUTALLOCATORPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Stamp `hipdnn.abi_shape` / `hipdnn.abi_groups` when the graph output is
// returned through a single rank-changing view (collapse_shape or expand_shape,
// optionally wrapped in memref.cast) of the EP-owned buffer.
//
// Collapse (internal rank > return rank): abi_groups[e] = # internal dims
// folded into external dim e (sum = internal rank, len = external rank).
//
// Expand (internal rank < return rank, e.g. DETR logits): abi_groups[i] = #
// external dims expanded from internal dim i (sum = external rank, len =
// internal rank).
static void stampAbiReshapeAttrs(AllocOutputOp allocOutput, Value retVal,
                                 OpBuilder &builder) {
  Value root = allocOutput.getResult();
  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!rootType)
    return;

  memref::CollapseShapeOp collapse;
  memref::ExpandShapeOp expand;
  Value cur = retVal;
  while (cur != root) {
    Operation *def = cur.getDefiningOp();
    if (!def)
      return;
    if (auto c = dyn_cast<memref::CollapseShapeOp>(def)) {
      if (collapse || expand)
        return;
      collapse = c;
      cur = c.getSrc();
      continue;
    }
    if (auto e = dyn_cast<memref::ExpandShapeOp>(def)) {
      if (collapse || expand)
        return;
      expand = e;
      cur = e.getSrc();
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      cur = castOp.getSource();
      continue;
    }
    return;
  }
  if (!collapse && !expand)
    return;

  auto extType = cast<MemRefType>(retVal.getType());
  if (extType.getRank() == rootType.getRank())
    return;

  if (auto module = allocOutput->getParentOfType<ModuleOp>()) {
    auto outputShapes =
        module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
    int64_t outIdx = allocOutput.getOutIdx();
    if (outputShapes && outIdx >= 0 &&
        outIdx < static_cast<int64_t>(outputShapes.size())) {
      if (auto metaShape = dyn_cast<DenseI64ArrayAttr>(
              outputShapes[static_cast<unsigned>(outIdx)])) {
        if (static_cast<int64_t>(metaShape.size()) != extType.getRank())
          return;
      }
    }
  }

  SmallVector<int64_t> groups;
  if (collapse) {
    int64_t total = 0;
    for (ArrayRef<int64_t> g : collapse.getReassociationIndices()) {
      groups.push_back(static_cast<int64_t>(g.size()));
      total += static_cast<int64_t>(g.size());
    }
    if (total != rootType.getRank() ||
        static_cast<int64_t>(groups.size()) != extType.getRank())
      return;
  } else {
    int64_t total = 0;
    for (ArrayRef<int64_t> g : expand.getReassociationIndices()) {
      groups.push_back(static_cast<int64_t>(g.size()));
      total += static_cast<int64_t>(g.size());
    }
    if (total != extType.getRank() ||
        static_cast<int64_t>(groups.size()) != rootType.getRank())
      return;
  }

  allocOutput->setAttr(kAbiShapeAttrName,
                       builder.getDenseI64ArrayAttr(extType.getShape()));
  allocOutput->setAttr(kAbiGroupsAttrName,
                       builder.getDenseI64ArrayAttr(groups));
}

/// Return the single rank-increasing expand_shape in the view chain from
/// \p retVal back to \p root. Casts are transparent; any other view shape, or
/// more than one reshape, is deliberately rejected.
static memref::ExpandShapeOp findSingleExpandShape(Value root, Value retVal) {
  memref::ExpandShapeOp expand;
  Value cur = retVal;
  while (cur != root) {
    Operation *def = cur.getDefiningOp();
    if (!def)
      return {};
    if (auto e = dyn_cast<memref::ExpandShapeOp>(def)) {
      if (expand)
        return {};
      expand = e;
      cur = e.getSrc();
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      cur = castOp.getSource();
      continue;
    }
    return {};
  }
  return expand;
}

/// A rank-increasing expand can split one internal dynamic dimension into
/// multiple external dynamic dimensions. The existing abi_groups lowering
/// cannot reconstruct those individual extents from their product. Allocate
/// the graph output at the external rank instead, then collapse it back to the
/// internal compute rank. Both descriptors alias the same contiguous storage:
///
///   alloc_output [B, S, H] -> collapse [B*S, H] -> compute -> expand [B,S,H]
///
/// Returns null when the ordinary internal-rank allocation + abi_groups path
/// remains sufficient.
static Value createExternalRankAllocationForMultiDynExpand(
    memref::AllocOp allocOp, Value retVal, Value ctx, int64_t outIdx,
    DominanceInfo &dominance, OpBuilder &builder) {
  memref::ExpandShapeOp expand =
      findSingleExpandShape(allocOp.getResult(), retVal);
  if (!expand)
    return {};

  auto internalType = allocOp.getType();
  auto externalType = expand.getResultType();
  if (externalType.getRank() <= internalType.getRank())
    return {};

  bool hasMultiDynamicGroup = llvm::any_of(
      expand.getReassociationIndices(), [&](ArrayRef<int64_t> group) {
        return llvm::count_if(group, [&](int64_t dim) {
                 return externalType.isDynamicDim(dim);
               }) > 1;
      });
  if (!hasMultiDynamicGroup)
    return {};

  ValueRange externalDynamicSizes = expand.getOutputShape();
  if (static_cast<int64_t>(externalDynamicSizes.size()) !=
      externalType.getNumDynamicDims()) {
    allocOp.emitError() << "rank-increasing graph output has "
                        << externalType.getNumDynamicDims()
                        << " dynamic dimensions but expand_shape carries "
                        << externalDynamicSizes.size() << " runtime sizes";
    return {};
  }
  for (Value size : externalDynamicSizes) {
    if (!dominance.dominates(size, allocOp)) {
      allocOp.emitError()
          << "external graph-output dimension does not dominate allocation";
      return {};
    }
  }

  builder.setInsertionPoint(allocOp);
  auto allocOutput = AllocOutputOp::create(
      builder, allocOp.getLoc(), externalType, ctx, externalDynamicSizes,
      builder.getI64IntegerAttr(outIdx));
  auto collapsedType = expand.getSrcType();
  auto internalView = memref::CollapseShapeOp::create(
      builder, allocOp.getLoc(), collapsedType, allocOutput.getResult(),
      expand.getReassociationIndices());
  Value replacement = internalView.getResult();
  if (collapsedType != internalType) {
    if (!memref::CastOp::areCastCompatible(collapsedType, internalType)) {
      allocOp.emitError() << "cannot cast collapsed graph-output view from "
                          << collapsedType << " to allocation type "
                          << internalType;
      return {};
    }
    replacement = memref::CastOp::create(builder, allocOp.getLoc(),
                                         internalType, replacement);
  }
  assert(replacement.getType() == internalType &&
         "graph-output replacement must match the allocation type");
  return replacement;
}

/// Walk cast / reshape / subview chains back to the buffer they view.
static Value skipViews(Value value) {
  while (Operation *def = value.getDefiningOp()) {
    if (isa<memref::CastOp, memref::CollapseShapeOp, memref::ExpandShapeOp,
            memref::SubViewOp, memref::ReshapeOp>(def) &&
        def->getNumOperands() >= 1) {
      value = def->getOperand(0);
      continue;
    }
    break;
  }
  return value;
}

/// Copy a returned buffer that is not rooted at `hip.alloc_output` into a new
/// one and return the copy. The source is typically a graph input returned
/// directly or through a view. The copy is what makes the output allocator run
/// for this output index.
static void materializePassthroughOutput(func::ReturnOp returnOp, Value ctx,
                                         int64_t outIdx, OpBuilder &builder) {
  Value source = returnOp.getOperand(outIdx);
  if (skipViews(source).getDefiningOp<AllocOutputOp>())
    return;

  auto type = dyn_cast<MemRefType>(source.getType());
  if (!type)
    return;

  builder.setInsertionPoint(returnOp);
  Location loc = returnOp.getLoc();
  SmallVector<Value> dynamicSizes;
  for (int64_t dim = 0; dim < type.getRank(); ++dim)
    if (type.isDynamicDim(dim))
      dynamicSizes.push_back(memref::DimOp::create(builder, loc, source, dim));

  auto allocOutput = AllocOutputOp::create(
      builder, loc, type, ctx, dynamicSizes, builder.getI64IntegerAttr(outIdx));
  memref::CopyOp::create(builder, loc, source, allocOutput.getResult());
  returnOp.setOperand(outIdx, allocOutput.getResult());
}

struct UseOutputAllocatorPass
    : impl::UseOutputAllocatorPassBase<UseOutputAllocatorPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    // The pass creates hip.alloc_output (HipDialect). BufferViewFlowAnalysis
    // follows the memref view ops (cast, collapse_shape, expand_shape, subview)
    // through their ViewLikeOpInterface, which MemRefDialect provides.
    // memref.dim materializes an arith.constant for a static dimension index.
    registry
        .insert<hip::HipDialect, memref::MemRefDialect, arith::ArithDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    // Only public (graph-entry) functions own EP outputs. Private helper
    // functions may also carry a !hip.context arg 0 and return memref.allocs,
    // but their outputs are DLL-internal -> skip non-public.
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
    // returned graph value per output index for the collapse-shape ABI
    // adjustment below.
    func::ReturnOp returnOp;
    funcOp.walk([&](func::ReturnOp r) { returnOp = r; });

    // Phase 2 -- rewrite (IR mutation only; the analysis is no longer queried).
    OpBuilder builder(funcOp.getContext());
    DominanceInfo dominance(funcOp);
    for (auto [allocOp, outIdx] : outputs) {
      // The EP owns this buffer now, so drop any dealloc of it. A returned
      // buffer normally has none, but remove one if present -- the EP-owned
      // output must never be freed by the graph.
      for (Operation *user : llvm::make_early_inc_range(allocOp->getUsers()))
        if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
          dealloc.erase();

      Value retVal =
          returnOp && outIdx >= 0 &&
                  outIdx < static_cast<int64_t>(returnOp.getNumOperands())
              ? returnOp.getOperand(outIdx)
              : Value{};
      Value replacement;
      if (retVal)
        replacement = createExternalRankAllocationForMultiDynExpand(
            allocOp, retVal, ctx, outIdx, dominance, builder);

      AllocOutputOp allocOutput;
      if (!replacement) {
        builder.setInsertionPoint(allocOp);
        allocOutput = AllocOutputOp::create(
            builder, allocOp.getLoc(), allocOp.getType(), ctx,
            allocOp.getDynamicSizes(), builder.getI64IntegerAttr(outIdx));
        replacement = allocOutput.getResult();
      }

      allocOp.getResult().replaceAllUsesWith(replacement);

      // After RAUW the returned value's view chain is rooted at allocOutput.
      // Record the returned graph shape when it is a rank-reduced collapse of
      // the internal compute buffer, so the lowering issues the
      // output-allocator callback at the returned graph rank (see
      // stampAbiReshapeAttrs).
      if (allocOutput && retVal)
        stampAbiReshapeAttrs(allocOutput, retVal, builder);

      allocOp.erase();
    }

    // Phase 3 -- pass-through outputs. A returned input (or a view of one)
    // never went through memref.alloc, so phase 2 did not allocate it. Copy it
    // into an EP-owned buffer so the output-allocator callback still runs.
    if (returnOp)
      for (int64_t outIdx = 0;
           outIdx < static_cast<int64_t>(returnOp.getNumOperands()); ++outIdx)
        materializePassthroughOutput(returnOp, ctx, outIdx, builder);
  }
};

} // namespace

} // namespace hip
} // namespace mlir
