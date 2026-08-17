/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- UseOutputAllocator.cpp - returned allocs -> hip.alloc_output -------===//
//
// Turns each graph-output `memref.alloc` into a `hip.alloc_output`. A graph
// output is a buffer that func.return hands back -- either the alloc directly,
// through any number of `memref.cast` ops, or through one representable
// `memref.collapse_shape` / `memref.expand_shape` (optionally with casts).
// A zero-offset, unit-step, rank-preserving subview returned as an
// identity-layout memref is copied into a fresh exact-shape output immediately
// before return. Other returned view chains are rejected before mutation: the
// allocator callback carries shape only and cannot represent a subview's
// offset/strides.
// `hip.alloc_output` gets its buffer from the EP's output allocator, so the EP
// owns and frees it. Allocs that are NOT returned are left alone and get freed
// / pooled later like normal temporaries.
//
// For each matching alloc the rewrite:
//   - sets `out_idx` to its position in func.return (the graph output index),
//   - reuses the alloc's dynamic-size operands unchanged,
//   - deletes any `memref.dealloc` of it (the EP frees it, not us),
//   - leaves the view ops in place -- only the alloc op itself changes,
//   - when the output is returned through a rank-changing view (collapse_shape
//     or expand_shape), stamps `hipdnn.abi_shape` / `hipdnn.abi_groups` so the
//     HIP->LLVM lowering issues the output-allocator callback at the RETURNED
//     (ONNX) rank rather than the internal compute rank.
//
// How outputs are found. `BufferViewFlowAnalysis` discovers every returned
// alias of an alloc. A separate backward walk then proves that the exact
// alloc-to-return chain is representable by the callback ABI. Discovery and
// validation finish for every output before any IR is changed.
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
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

struct AbiReshapePlan {
  bool stampAttrs = false;
  bool materializeExactCopy = false;
  SmallVector<int64_t> shape;
  SmallVector<int64_t> groups;
};

struct OutputPlan {
  memref::AllocOp alloc;
  int64_t outIdx;
  AbiReshapePlan abiReshape;
};

// Validate the exact alloc-to-return chain and plan
// `hipdnn.abi_shape` / `hipdnn.abi_groups` when the graph output is returned
// through a single rank-changing view (collapse_shape or expand_shape,
// optionally wrapped in memref.cast) of the EP-owned buffer.
//
// Collapse (internal rank > return rank): abi_groups[e] = # internal dims
// folded into external dim e (sum = internal rank, len = external rank).
//
// Expand (internal rank < return rank, e.g. DETR logits): abi_groups[i] = #
// external dims expanded from internal dim i (sum = external rank, len =
// internal rank).
static LogicalResult planReturnedViewChain(memref::AllocOp alloc, Value retVal,
                                           int64_t outIdx,
                                           func::ReturnOp returnOp,
                                           AbiReshapePlan &plan) {
  Value root = alloc.getResult();
  auto rootType = dyn_cast<MemRefType>(root.getType());
  if (!rootType)
    return success();

  memref::CollapseShapeOp collapse;
  memref::ExpandShapeOp expand;
  memref::SubViewOp subview;
  Value cur = retVal;
  while (cur != root) {
    Operation *def = cur.getDefiningOp();
    if (!def) {
      returnOp.emitError()
          << "output #" << outIdx
          << " has an unsupported returned output-view chain that does not "
             "trace back to its allocation";
      return failure();
    }
    if (auto c = dyn_cast<memref::CollapseShapeOp>(def)) {
      if (collapse || expand || subview) {
        returnOp.emitError()
            << "output #" << outIdx
            << " has an unsupported returned output-view chain mixing "
               "memref.subview or multiple rank-changing reshape ops";
        return failure();
      }
      collapse = c;
      cur = c.getSrc();
      continue;
    }
    if (auto e = dyn_cast<memref::ExpandShapeOp>(def)) {
      if (collapse || expand || subview) {
        returnOp.emitError()
            << "output #" << outIdx
            << " has an unsupported returned output-view chain mixing "
               "memref.subview or multiple rank-changing reshape ops";
        return failure();
      }
      expand = e;
      cur = e.getSrc();
      continue;
    }
    if (auto s = dyn_cast<memref::SubViewOp>(def)) {
      if (collapse || expand || subview) {
        returnOp.emitError()
            << "output #" << outIdx
            << " has an unsupported returned output-view chain mixing "
               "memref.subview or multiple rank-changing reshape ops";
        return failure();
      }
      subview = s;
      cur = s.getSource();
      continue;
    }
    if (auto castOp = dyn_cast<memref::CastOp>(def)) {
      cur = castOp.getSource();
      continue;
    }
    returnOp.emitError()
        << "output #" << outIdx
        << " has an unsupported returned output-view chain through '"
        << def->getName()
        << "'; the output allocator callback cannot represent view offsets or "
           "strides";
    return failure();
  }

  if (subview) {
    auto extType = dyn_cast<MemRefType>(retVal.getType());
    Type elemType = extType ? extType.getElementType() : Type();
    if (!extType || extType.getRank() != rootType.getRank()) {
      returnOp.emitError()
          << "output #" << outIdx
          << " has an unsupported rank-reducing returned memref.subview";
      return failure();
    }
    for (OpFoldResult offset : subview.getMixedOffsets()) {
      std::optional<int64_t> value = getConstantIntValue(offset);
      if (!value || *value != 0) {
        returnOp.emitError()
            << "output #" << outIdx
            << " has a returned memref.subview with a nonzero or dynamic "
               "offset; exact-output copying requires zero offsets";
        return failure();
      }
    }
    for (OpFoldResult stride : subview.getMixedStrides()) {
      std::optional<int64_t> value = getConstantIntValue(stride);
      if (!value || *value != 1) {
        returnOp.emitError()
            << "output #" << outIdx
            << " has a returned memref.subview with a non-unit or dynamic "
               "stride; exact-output copying requires unit slice strides";
        return failure();
      }
    }
    if (!extType.getLayout().isIdentity()) {
      returnOp.emitError()
          << "output #" << outIdx
          << " has a returned memref.subview without an identity-layout "
             "return type; exact-output copying cannot replace the return";
      return failure();
    }
    if (!elemType.isIntOrFloat() || elemType.getIntOrFloatBitWidth() % 8 != 0) {
      returnOp.emitError()
          << "output #" << outIdx
          << " has a returned memref.subview with an element type unsupported "
             "by exact-output copying";
      return failure();
    }
    plan.materializeExactCopy = true;
    return success();
  }

  if (!collapse && !expand)
    return success();

  auto extType = dyn_cast<MemRefType>(retVal.getType());
  if (!extType) {
    returnOp.emitError() << "output #" << outIdx
                         << " has a non-memref returned output-view type";
    return failure();
  }
  if (extType.getRank() == rootType.getRank())
    return success();

  if (auto module = alloc->getParentOfType<ModuleOp>()) {
    auto outputShapes =
        module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
    if (outputShapes && outIdx >= 0 &&
        outIdx < static_cast<int64_t>(outputShapes.size())) {
      if (auto metaShape = dyn_cast<DenseI64ArrayAttr>(
              outputShapes[static_cast<unsigned>(outIdx)])) {
        if (static_cast<int64_t>(metaShape.size()) != extType.getRank()) {
          returnOp.emitError()
              << "output #" << outIdx << " has output metadata rank "
              << metaShape.size() << " but its returned output-view rank is "
              << extType.getRank();
          return failure();
        }
      }
    }
  }

  if (collapse) {
    int64_t total = 0;
    for (ArrayRef<int64_t> g : collapse.getReassociationIndices()) {
      plan.groups.push_back(static_cast<int64_t>(g.size()));
      total += static_cast<int64_t>(g.size());
    }
    if (total != rootType.getRank() ||
        static_cast<int64_t>(plan.groups.size()) != extType.getRank()) {
      returnOp.emitError()
          << "output #" << outIdx
          << " has an unsupported memref.collapse_shape reassociation for the "
             "output allocator callback";
      return failure();
    }
  } else {
    int64_t total = 0;
    for (ArrayRef<int64_t> g : expand.getReassociationIndices()) {
      plan.groups.push_back(static_cast<int64_t>(g.size()));
      total += static_cast<int64_t>(g.size());
    }
    if (total != extType.getRank() ||
        static_cast<int64_t>(plan.groups.size()) != rootType.getRank()) {
      returnOp.emitError()
          << "output #" << outIdx
          << " has an unsupported memref.expand_shape reassociation for the "
             "output allocator callback";
      return failure();
    }

    int64_t externalDim = 0;
    for (int64_t groupSize : plan.groups) {
      int64_t dynamicDims = 0;
      for (int64_t i = 0; i < groupSize; ++i)
        dynamicDims +=
            ShapedType::isDynamic(extType.getDimSize(externalDim + i));
      if (dynamicDims > 1) {
        returnOp.emitError()
            << "output #" << outIdx
            << " has an unsupported memref.expand_shape group with multiple "
               "dynamic returned dimensions";
        return failure();
      }
      externalDim += groupSize;
    }
  }

  plan.stampAttrs = true;
  plan.shape.assign(extType.getShape().begin(), extType.getShape().end());
  return success();
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

    // Phase 1 -- classify and validate (analysis only, no IR mutation). For
    // each alloc that is a graph output, record (alloc, out_idx). resolve()
    // gives back the
    // alloc plus every value derived from it through view ops (cast / collapse
    // / expand / subview / ...), so a returned alloc is found even when it was
    // reshaped on the way to the return. getOperandNumber() on a func.return
    // use of any alias IS the graph output index; if the buffer is returned in
    // more than one slot (aliased multi-output), take the first (lowest).
    // Keeping every analysis query here -- before any rewrite -- means the
    // analysis (which caches Value handles) is never read after the IR it
    // describes has been mutated. Walk order is program order, so out_idx
    // values print in order in phase 2.
    SmallVector<OutputPlan> outputs;
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
        outputs.push_back({allocOp, outIdx, {}});
    });

    // The single func.return of this graph-entry function. Used to recover the
    // returned (ONNX ABI) value per output index for the collapse-shape ABI
    // adjustment below.
    func::ReturnOp returnOp;
    funcOp.walk([&](func::ReturnOp r) { returnOp = r; });

    // Validate every returned alias before rewriting even the first alloc.
    // This makes pass failure atomic with respect to this function: unsupported
    // subviews or mixed reshape chains cannot leave earlier outputs converted.
    bool invalid = false;
    for (OutputPlan &output : outputs) {
      if (!returnOp || output.outIdx < 0 ||
          output.outIdx >= static_cast<int64_t>(returnOp.getNumOperands())) {
        funcOp.emitError() << "cannot resolve func.return operand for output #"
                           << output.outIdx;
        invalid = true;
        continue;
      }
      if (failed(planReturnedViewChain(
              output.alloc, returnOp.getOperand(output.outIdx), output.outIdx,
              returnOp, output.abiReshape)))
        invalid = true;
    }
    if (invalid) {
      signalPassFailure();
      return;
    }

    // Phase 2 -- rewrite (IR mutation only; the analysis is no longer queried).
    OpBuilder builder(funcOp.getContext());
    for (OutputPlan &output : outputs) {
      memref::AllocOp allocOp = output.alloc;
      int64_t outIdx = output.outIdx;
      if (output.abiReshape.materializeExactCopy) {
        Value returned = returnOp.getOperand(outIdx);
        auto outputType = cast<MemRefType>(returned.getType());
        builder.setInsertionPoint(returnOp);
        SmallVector<Value> dynamicSizes;
        for (int64_t dim = 0; dim < outputType.getRank(); ++dim)
          if (outputType.isDynamicDim(dim))
            dynamicSizes.push_back(memref::DimOp::create(
                builder, returnOp.getLoc(), returned, dim));
        auto exactOutput = AllocOutputOp::create(
            builder, returnOp.getLoc(), outputType, ctx, dynamicSizes,
            builder.getI64IntegerAttr(outIdx));
        memref::CopyOp::create(builder, returnOp.getLoc(), returned,
                               exactOutput.getResult());
        returnOp->setOperand(outIdx, exactOutput.getResult());
        continue;
      }

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

      if (output.abiReshape.stampAttrs) {
        allocOutput->setAttr(kAbiShapeAttrName, builder.getDenseI64ArrayAttr(
                                                    output.abiReshape.shape));
        allocOutput->setAttr(kAbiGroupsAttrName, builder.getDenseI64ArrayAttr(
                                                     output.abiReshape.groups));
      }
    }
  }
};

} // namespace

} // namespace hip
} // namespace mlir
