/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LowerAllocs.cpp - memref.alloc -> hip.alloc + hip.free -------------===//
//
// Replaces memref.alloc with hip.alloc (device allocation via hipMalloc) and
// inserts hip.free for non-returned buffers.  Bridges one-shot-bufferize
// output to the HIP-specific lowering that ultimately produces
// hipMalloc / hipFree runtime calls.
//
// Ownership convention:
//   - Returned buffers are caller-owned: no hip.free is emitted.
//   - All other buffers are freed before hip.destroy_handle, because the
//     handle encapsulates the HIP runtime context required by hipFree.
//     If no destroy_handle exists, frees are placed before the terminator.
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

  // Find the insertion point for hip.free ops.  Prefer placing them just
  // before hip.destroy_handle; fall back to just before the terminator.
  Operation* freeInsertionPoint = nullptr;
  funcOp.walk([&](DestroyHandleOp op) { freeInsertionPoint = op; });
  if (!freeInsertionPoint) {
    Block& entry = funcOp.getBody().front();
    freeInsertionPoint = entry.getTerminator();
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

  // Insert hip.free for each buffer not returned to the caller.
  for (AllocOp hipAlloc : hipAllocs) {
    if (!returnedValues.contains(hipAlloc.getResult())) {
      builder.setInsertionPoint(freeInsertionPoint);
      FreeOp::create(builder, freeInsertionPoint->getLoc(), handle,
                     hipAlloc.getResult());
    }
  }
}

}  // namespace
}  // namespace hip
}  // namespace mlir
