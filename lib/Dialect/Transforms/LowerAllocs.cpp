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
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_LOWERALLOCSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Return true if \p op creates a memref alias (view) of one of its inputs
/// without allocating new memory.  Uses the MLIR ViewLikeOpInterface so that
/// any future view-like ops are automatically covered.  arith::SelectOp is
/// included because its result may alias either memref operand.
static bool isMemRefAlias(Operation *op) {
  return isa<ViewLikeOpInterface, arith::SelectOp>(op);
}

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

/// Find the last user of \p rootValue, walking transitively through
/// view-like (aliasing) ops.  Without this, hip.free can be placed after
/// a memref.view but before consumers of the view -- a use-after-free.
///
/// Users in nested regions (e.g. scf.for body) are resolved to their
/// ancestor in the entry block so that isBeforeInBlock remains valid.
static Operation *findLastTransitiveUser(Value rootValue, Block &entryBlock) {
  Operation *lastUser = rootValue.getDefiningOp();
  SmallVector<Value> worklist = {rootValue};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value val = worklist.pop_back_val();
    if (!visited.insert(val).second)
      continue;
    for (Operation *user : val.getUsers()) {
      Operation *resolved = user;
      if (resolved->getBlock() != &entryBlock) {
        resolved = entryBlock.findAncestorOpInBlock(*resolved);
        if (!resolved)
          continue;
      }
      if (lastUser->isBeforeInBlock(resolved))
        lastUser = resolved;
      if (isMemRefAlias(user)) {
        for (OpResult result : user->getResults())
          worklist.push_back(result);
      }
    }
  }
  return lastUser;
}

struct LowerAllocsPass : public impl::LowerAllocsPassBase<LowerAllocsPass> {
  void runOnOperation() override;
};

void LowerAllocsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.empty())
    return;
  if (!funcOp.getBody().hasOneBlock()) {
    funcOp.emitError("hip-lower-allocs requires single-block functions; "
                     "findLastTransitiveUser uses isBeforeInBlock which does "
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

  // Replace each memref.alloc with hip.alloc.
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
  }

  // Collect returned values AFTER all replacements so the set contains
  // the new hip.alloc results (not the erased memref.alloc results).
  DenseSet<Value> returnedValues;
  funcOp.walk([&](func::ReturnOp ret) {
    for (Value v : ret.getOperands())
      returnedValues.insert(v);
  });

  // Check whether any value in the transitive alias chain of \p root is
  // returned.  Needed because pooled allocs are returned via memref.view,
  // not directly.
  auto isTransitivelyReturned = [&](Value root) -> bool {
    SmallVector<Value> wl = {root};
    DenseSet<Value> seen;
    while (!wl.empty()) {
      Value v = wl.pop_back_val();
      if (!seen.insert(v).second)
        continue;
      if (returnedValues.contains(v))
        return true;
      for (Operation *user : v.getUsers()) {
        if (isMemRefAlias(user))
          for (OpResult r : user->getResults())
            wl.push_back(r);
      }
    }
    return false;
  };

  Block &entry = funcOp.getBody().front();

  // Replace memref.dealloc -> hip.free for buffers that trace back to a
  // hip.alloc.  Non-HIP deallocs (e.g. externally-owned memrefs) are left
  // untouched for other lowerings.
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
  }

  // Fallback: insert hip.free after last use for allocs that have no
  // memref.dealloc (e.g. when buffer-deallocation-pipeline is not in
  // the pass pipeline).  Uses transitive user walk to account for
  // alias-creating ops like memref.view / subview / cast.
  for (AllocOp hipAlloc : hipAllocs) {
    if (isTransitivelyReturned(hipAlloc.getResult()))
      continue;
    if (deallocated.contains(hipAlloc.getResult()))
      continue;

    Operation *lastUser = findLastTransitiveUser(hipAlloc.getResult(), entry);
    builder.setInsertionPointAfter(lastUser);
    FreeOp::create(builder, hipAlloc.getLoc(), ctx, hipAlloc.getResult());
  }
}

} // namespace
} // namespace hip
} // namespace mlir
