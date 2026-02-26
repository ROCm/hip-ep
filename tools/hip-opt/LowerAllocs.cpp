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

#include "HipDialect.h"
#include "HipPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_LOWERALLOCSPASS
#include "HipPasses.h.inc"

namespace {

struct LowerAllocsPass : public impl::LowerAllocsPassBase<LowerAllocsPass> {
  void runOnOperation() override;
};

void LowerAllocsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.empty())
    return;

  // Find the hip.handle produced by hip.create_handle.
  Value handle;
  funcOp.walk([&](CreateHandleOp op) { handle = op.getResult(); });
  if (!handle) {
    funcOp.emitRemark("no hip.create_handle found; skipping alloc lowering");
    return;
  }

  OpBuilder builder(funcOp.getContext());

  // Replace each memref.alloc with hip.alloc.
  SmallVector<memref::AllocOp> allocs;
  funcOp.walk([&](memref::AllocOp op) { allocs.push_back(op); });

  SmallVector<AllocOp> hipAllocs;
  for (memref::AllocOp allocOp : allocs) {
    builder.setInsertionPoint(allocOp);
    auto hipAlloc = AllocOp::create(builder, allocOp.getLoc(),
                                    allocOp.getType(), handle,
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

  // Move hip.destroy_handle to just before the terminator.  Passes like
  // buffer-results-to-out-params may insert ops (e.g., memref.copy) after
  // the original destroy_handle position; the handle must stay valid until
  // all hip.free ops have executed.
  Block& entry = funcOp.getBody().front();
  Operation* terminator = entry.getTerminator();
  Operation* destroyHandleOp = nullptr;
  funcOp.walk([&](DestroyHandleOp op) { destroyHandleOp = op; });
  if (destroyHandleOp)
    destroyHandleOp->moveBefore(terminator);

  // Replace memref.dealloc -> hip.free (placed by buffer-deallocation-pipeline).
  DenseSet<Value> deallocated;
  SmallVector<memref::DeallocOp> deallocs;
  funcOp.walk([&](memref::DeallocOp op) { deallocs.push_back(op); });
  for (memref::DeallocOp deallocOp : deallocs) {
    Value memref = deallocOp.getMemref();
    builder.setInsertionPoint(deallocOp);
    FreeOp::create(builder, deallocOp.getLoc(), handle, memref);
    deallocated.insert(memref);
    deallocOp.erase();
  }

  // Fallback: insert hip.free after last use for allocs that have no
  // memref.dealloc (e.g. when buffer-deallocation-pipeline is not in
  // the pass pipeline).
  for (AllocOp hipAlloc : hipAllocs) {
    if (returnedValues.contains(hipAlloc.getResult()))
      continue;
    if (deallocated.contains(hipAlloc.getResult()))
      continue;

    Operation* lastUser = hipAlloc;
    for (Operation* user : hipAlloc.getResult().getUsers()) {
      if (lastUser->isBeforeInBlock(user))
        lastUser = user;
    }

    builder.setInsertionPointAfter(lastUser);
    FreeOp::create(builder, hipAlloc.getLoc(), handle, hipAlloc.getResult());
  }
}

}  // namespace
}  // namespace hip
}  // namespace mlir
