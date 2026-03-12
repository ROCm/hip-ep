/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LowerAllocs.cpp - memref.alloc -> hip.alloc + hip.free -------------===//
//
// Replaces memref.alloc with hip.alloc (device allocation via hipMalloc) and
// inserts hip.free for buffers not returned from the function.  Bridges
// one-shot-bufferize output to HIP-specific lowering.
//
// Alias-aware free placement uses BufferViewFlowAnalysis to transitively
// follow view-like ops so that hip.free is placed after all consumers of
// any derived view, preventing use-after-free.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/BufferUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-lower-allocs"

STATISTIC(NumAllocsLowered, "Number of memref.alloc lowered to hip.alloc");
STATISTIC(NumFreesInserted, "Number of hip.free ops inserted");
STATISTIC(NumDeallocsConverted,
          "Number of memref.dealloc converted to hip.free");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_LOWERALLOCSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Walk \p val back through view-like alias ops and return the root
/// hip::AllocOp if the chain originates from one, or nullptr otherwise.
static AllocOp traceToHipAlloc(Value val) {
  while (val) {
    if (auto alloc = val.getDefiningOp<AllocOp>())
      return alloc;
    auto viewLike = dyn_cast_or_null<ViewLikeOpInterface>(val.getDefiningOp());
    if (!viewLike)
      return nullptr;
    val = viewLike.getViewSource();
  }
  return nullptr;
}

struct LowerAllocsPass : public impl::LowerAllocsPassBase<LowerAllocsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hip::HipDialect, memref::MemRefDialect>();
    arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override;
};

void LowerAllocsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.empty())
    return;
  // TODO: Generalize to multi-block functions using MLIR's Liveness analysis.
  if (!funcOp.getBody().hasOneBlock()) {
    funcOp.emitError("hip-lower-allocs requires single-block functions; "
                     "findLastAliasedUser uses isBeforeInBlock which does "
                     "not generalize to multi-block control flow");
    return signalPassFailure();
  }

  // Get !hip.context from function argument 0.
  Value ctx;
  if (!funcOp.getArguments().empty() &&
      isa<ContextType>(funcOp.getArgument(0).getType()))
    ctx = funcOp.getArgument(0);
  if (!ctx) {
    funcOp.emitError("no !hip.context argument found; "
                     "hip-add-context-arg must run first");
    return signalPassFailure();
  }

  OpBuilder builder(funcOp.getContext());

  // ---- Phase 1: Replace each memref.alloc with hip.alloc ----------------
  SmallVector<memref::AllocOp> allocs;
  funcOp.walk([&](memref::AllocOp op) { allocs.push_back(op); });

  SmallVector<AllocOp> hipAllocs;
  for (memref::AllocOp allocOp : allocs) {
    builder.setInsertionPoint(allocOp);
    auto hipAlloc =
        AllocOp::create(builder, allocOp.getLoc(), allocOp.getType(), ctx,
                        allocOp.getDynamicSizes());
    allocOp.replaceAllUsesWith(hipAlloc.getResult());
    allocOp.erase();
    hipAllocs.push_back(hipAlloc);
    ++NumAllocsLowered;
  }

  // ---- Phase 2: Build alias analysis on the *modified* IR ---------------
  //
  // CRITICAL: BufferViewFlowAnalysis must be constructed AFTER the
  // memref.alloc -> hip.alloc replacement above.  The analysis records
  // forward alias edges (source -> view) for ViewLikeOpInterface ops
  // (memref.view, memref.subview, memref.cast, etc.).  resolve(value)
  // then returns all downstream aliases of that value.
  //
  // If the analysis were built on the original IR (with memref.alloc),
  // then queried with the NEW hip.alloc SSA values, resolve() would
  // return only {hipAlloc} -- missing all downstream views.  This causes:
  //   - isAliasInSet: fails to detect that a returned memref.view aliases
  //     the pool, so hip.free is incorrectly inserted for returned buffers.
  //   - findLastAliasedUser: finds only the immediate SSA user (e.g. the
  //     memref.view op) rather than the last transitive consumer, placing
  //     hip.free before ops that still read/write through the view.
  //
  // Both are use-after-free bugs triggered in the pool-allocs -> lower-allocs
  // pipeline, where the single pool hip.alloc has memref.view aliases that
  // are returned from the function.
  BufferViewFlowAnalysis aliasAnalysis(funcOp);

  // Collect returned values AFTER all replacements so the set contains
  // the new hip.alloc results (not the erased memref.alloc results).
  DenseSet<Value> returnedValues;
  funcOp.walk([&](func::ReturnOp ret) {
    for (Value v : ret.getOperands())
      returnedValues.insert(v);
  });

  Block &entry = funcOp.getBody().front();

  // ---- Phase 3: Convert memref.dealloc -> hip.free ----------------------
  //
  // Explicit deallocs that trace back to a hip.alloc are converted first.
  // traceToHipAlloc walks through ViewLikeOpInterface ops to find the root.
  DenseSet<Value> deallocated;
  SmallVector<memref::DeallocOp> deallocs;
  funcOp.walk([&](memref::DeallocOp op) { deallocs.push_back(op); });
  for (memref::DeallocOp deallocOp : deallocs) {
    Value memref = deallocOp.getMemref();
    AllocOp rootAlloc = traceToHipAlloc(memref);
    if (!rootAlloc)
      continue;

    builder.setInsertionPoint(deallocOp);
    FreeOp::create(builder, deallocOp.getLoc(), ctx, memref);
    deallocated.insert(rootAlloc.getResult());
    deallocOp.erase();
    ++NumDeallocsConverted;
  }

  // ---- Phase 4: Insert hip.free for allocs without explicit dealloc -----
  //
  // Uses BufferViewFlowAnalysis to find all downstream aliases (views,
  // subviews, casts) of each hip.alloc.  hip.free is placed after the
  // last user of any alias, ensuring all transitive consumers have
  // completed before the memory is released.
  for (AllocOp hipAlloc : hipAllocs) {
    if (isAliasInSet(hipAlloc.getResult(), aliasAnalysis, returnedValues))
      continue;
    if (deallocated.contains(hipAlloc.getResult()))
      continue;

    Operation *lastUser =
        findLastAliasedUser(hipAlloc.getResult(), aliasAnalysis, entry);
    builder.setInsertionPointAfter(lastUser);
    FreeOp::create(builder, hipAlloc.getLoc(), ctx, hipAlloc.getResult());
    ++NumFreesInserted;
    LLVM_DEBUG(llvm::dbgs()
               << "  Inserted hip.free after " << *lastUser << "\n");
  }
}

} // namespace
} // namespace hip
} // namespace mlir
