/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HoistAllocSizeArith.cpp - Schedule allocation-size producers
//---------===//
//
// PoolAllocs does not move existing operations. A late size definition can
// therefore create a separate pool even when buffer lifetimes do not overlap.
// Move pure size producers before the earliest allocation their operands allow.
// A late read or non-speculatable operation limits movement; it does not
// prevent its pure consumers from moving within the remaining part of the
// block.
//
// Before:
//   %early = memref.alloc(%n) : memref<?xf32>
//   %late = func.call @read_extent() : () -> index
//   %work = memref.alloc(%late) : memref<?xf32>
//   // ... use %work ...
//   %quarter = arith.divsi %late, %c4 : index
//   %tail = memref.alloc(%quarter) : memref<?xf32>
//
// After:
//   %early = memref.alloc(%n) : memref<?xf32>
//   %late = func.call @read_extent() : () -> index
//   %quarter = arith.divsi %late, %c4 : index
//   %work = memref.alloc(%late) : memref<?xf32>
//   // ... use %work ...
//   %tail = memref.alloc(%quarter) : memref<?xf32>
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"

#include <algorithm>
#include <utility>

#define DEBUG_TYPE "hip-hoist-alloc-size-arith"

STATISTIC(NumDynAllocsExamined,
          "Number of memref.alloc ops with dynamic operands inspected");
STATISTIC(NumOpsHoisted,
          "Number of pure size producers moved before allocations");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HOISTALLOCSIZEARITHPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct HoistAllocSizeArithPass
    : public impl::HoistAllocSizeArithPassBase<HoistAllocSizeArithPass> {
  void runOnOperation() override;
};

void HoistAllocSizeArithPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty() || !funcOp.getBody().hasOneBlock())
    return;
  Block &block = funcOp.getBody().front();

  // Match PoolAllocs: only used allocations directly in the entry block.
  // These anchors remain in block order throughout the pass.
  SmallVector<memref::AllocOp> allocs;
  SmallVector<Value> worklist;
  // Index of the first allocation strictly after each operation. Allocation i
  // itself has frontier i + 1, so its users cannot move ahead of it.
  llvm::DenseMap<Operation *, size_t> nextAlloc;
  for (Operation &op : block) {
    if (auto alloc = dyn_cast<memref::AllocOp>(op);
        alloc && !alloc.getResult().use_empty()) {
      allocs.push_back(alloc);
      if (!alloc.getDynamicSizes().empty()) {
        ++NumDynAllocsExamined;
        llvm::append_range(worklist, alloc.getDynamicSizes());
      }
    }
    nextAlloc[&op] = allocs.size();
  }
  if (worklist.empty())
    return;

  llvm::SmallPtrSet<Operation *, 32> candidates;
  while (!worklist.empty()) {
    Operation *def = worklist.pop_back_val().getDefiningOp();
    if (!def || def->getBlock() != &block || nextAlloc.lookup(def) == 0 ||
        candidates.contains(def))
      continue;
    // isPure includes speculatability, not just absence of memory effects.
    // Regions may capture dependencies absent from the parent's operand list.
    if (isa<memref::AllocOp, memref::AllocaOp>(def) ||
        def->getNumRegions() != 0 || !mlir::isPure(def))
      continue;
    candidates.insert(def);
    llvm::append_range(worklist, def->getOperands());
  }

  // Plan in SSA order, propagating each producer's new frontier to its users.
  // Apply moves only after planning: moveBefore invalidates MLIR's block order,
  // so interleaving moves with order queries can repeatedly rescan the block.
  SmallVector<std::pair<Operation *, Operation *>> moves;
  for (Operation &op : block) {
    if (!candidates.contains(&op))
      continue;
    size_t target = 0;
    for (Value operand : op.getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && def->getBlock() == &block)
        target = std::max(target, nextAlloc.lookup(def));
    }
    if (target < nextAlloc.lookup(&op)) {
      moves.emplace_back(&op, allocs[target]);
      nextAlloc[&op] = target;
    }
  }
  if (moves.empty()) {
    markAllAnalysesPreserved();
    return;
  }
  for (auto [op, anchor] : moves)
    op->moveBefore(anchor);
  NumOpsHoisted += moves.size();
}

} // namespace
} // namespace hip
} // namespace mlir
