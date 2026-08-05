/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HoistAllocSizeArith.cpp - Hoist alloc-size LICM for pool-allocs ----===//
//
// `--hip-hoist-alloc-size-arith` — hoists pure producers of
// `memref.alloc` dynamic operands above the earliest used `memref.alloc` in the
// function's single entry block. After this pass, every hoistable dynamic-size
// value dominates every allocation that `--hip-pool-allocs` may absorb.
//
// Why this pass exists
// --------------------
// `--hip-pool-allocs` is deliberately single-purpose: it anchors each
// domain's `hip.get_pool` and pool-size arithmetic at one block
// position and does NOT move existing ops.  That position must dominate
// every pooled alloc while being dominated by every alloc's dyn-operand
// defs; if an alloc's dyn-operand def appears below the earliest pooled
// alloc, pool-allocs splits a new domain rather than silently
// mis-placing arithmetic.
//
// On bufferised graphs that condition is occasionally violated by
// pure arithmetic that canonicalisation left interleaved with
// allocs (e.g. `%6 = arith.muli %dim, %dim_0` after
// `%alloc = memref.alloc(%dim, %dim_0)`).  This pass moves exactly that
// arithmetic up using MLIR's `mlir::isPure` predicate, the same standard used
// by upstream LICM. This excludes memory effects and operations that cannot be
// speculated, such as `arith.divsi` with a runtime-zero divisor. Region-bearing
// producers are rejected because regions may capture values that are not
// explicit operands of the parent operation.
//
// Algorithm
// ---------
// Single-entry-block functions only (multi-block functions are skipped,
// matching `--hip-pool-allocs`).  Only allocs that are direct children of
// the entry block are considered; allocs nested inside region-carrying
// ops are left untouched.
//
//   1. Collect every used entry-block `memref.alloc`, and identify the subset
//      with dynamic operands.
//   2. Pick `earliestAlloc` = the earliest used allocation in the block (per
//      `Operation::isBeforeInBlock`).
//   3. For each dynamic allocation, walk SSA backward from each dynamic
//      operand.
//      An op is hoistable when:
//         - it lives in the entry block,
//         - it is below `earliestAlloc` (otherwise it already dominates),
//         - it is not a `memref.alloc` / `memref.alloca`,
//         - it has no regions and satisfies `mlir::isPure(op)`,
//         - and every transitive operand is itself hoistable (or already
//           dominates `earliestAlloc`).
//      Hoistable ops are inserted into a SetVector.  Insertion order is
//      operand-before-use because each op is inserted AFTER its operands
//      have been recursed into.
//   4. Forward iteration of the SetVector calls `op->moveBefore(
//      earliestAlloc)` on each op.  Each `moveBefore` displaces prior
//      moves up by one slot, so the deepest operands end up at the top
//      of the hoist region and the closest-to-the-alloc ops at the
//      bottom.  Final layout above the earliest alloc is a topological
//      slice of the pure predecessor cone.
//
// Why not just mark hoistable=true unconditionally?
// -------------------------------------------------
// A chain `%loaded = memref.load ... ; %doubled = arith.muli %loaded,
// %c2 ; %alloc(%doubled)` looks half-hoistable — `%doubled` is pure, but its
// operand `%loaded` is not. Moving `%doubled`
// above `%alloc` while `%loaded` stays below would break SSA dominance
// at the new `%doubled` site.  The recursive check ensures we hoist
// `%doubled` only when its entire transitive operand cone is also
// hoistable (or already dominates `earliestAlloc`).
//
// Idempotence
// -----------
// A second invocation finds nothing below `earliestAlloc` that still
// needs moving (everything in the cone now dominates), so the pass is a
// no-op.  Verified by a LIT fixture.
//
// Example IR (the canonical case the regression tests pin down)
// -------------------------------------------------------------
//
// Before:
//
//   %dim   = memref.dim %arg, %c0 : memref<?x?xi64>
//   %dim_0 = memref.dim %arg, %c1 : memref<?x?xi64>
//   %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xui8>
//   // ... unrelated ops ...
//   %6 = arith.muli %dim, %dim_0      : index
//   %7 = arith.muli %6,   %c4096      : index
//   %alloc_6 = memref.alloc(%7) : memref<3x?xi64>
//
// After (`%6` and `%7` hoisted to just before `%alloc`):
//
//   %dim   = memref.dim %arg, %c0 : memref<?x?xi64>
//   %dim_0 = memref.dim %arg, %c1 : memref<?x?xi64>
//   %6 = arith.muli %dim, %dim_0      : index
//   %7 = arith.muli %6,   %c4096      : index
//   %alloc = memref.alloc(%dim, %dim_0) : memref<?x?xui8>
//   // ... unrelated ops ...
//   %alloc_6 = memref.alloc(%7) : memref<3x?xi64>
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-hoist-alloc-size-arith"

STATISTIC(NumDynAllocsExamined,
          "Number of memref.alloc ops with dynamic operands inspected");
STATISTIC(NumOpsHoisted,
          "Number of pure ops moved above the earliest used alloc");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_HOISTALLOCSIZEARITHPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Recursively determine whether \p value can be made to dominate
/// \p earliestAlloc by hoisting pure predecessors that are
/// currently below it.  Side-effect: every op on the path that needs to
/// move is inserted into \p toMove (a SetVector — operand-before-use
/// order is preserved by inserting after the recursion).
///
/// Return values:
///   - `true`  — \p value already dominates `earliestAlloc`, OR is
///               defined out-of-block, OR every op on the pure
///               cone has been added to `toMove` (and will dominate
///               after the move loop).
///   - `false` — at least one op on the cone is not pure, has regions, or is
///               itself an allocation; nothing on the cone of \p value can be
///               safely hoisted.
///
/// `cache` memoises per-op verdicts within one candidate cone. Cycles cannot
/// occur in pure SSA, but the optimistic-`true` insertion before recursion
/// defends against pathological IR without infinite recursion.
static bool isReachableHoistable(Value value, Block *block,
                                 Operation *earliestAlloc,
                                 llvm::DenseMap<Operation *, bool> &cache,
                                 llvm::SetVector<Operation *> &toMove) {
  Operation *def = value.getDefiningOp();
  if (!def)
    return true;
  if (def->getBlock() != block)
    return true;
  if (def->isBeforeInBlock(earliestAlloc))
    return true;
  if (isa<memref::AllocOp, memref::AllocaOp>(def))
    return false;

  auto it = cache.find(def);
  if (it != cache.end())
    return it->second;

  // Recursively-speculatable region operations may capture values that are not
  // explicit operands of the parent op. Moving such an op based only on its
  // operand list can violate dominance inside the region.
  if (def->getNumRegions() != 0) {
    cache[def] = false;
    LLVM_DEBUG(llvm::dbgs() << "  [region-bearing] " << def->getName() << "\n");
    return false;
  }

  if (!mlir::isPure(def)) {
    cache[def] = false;
    LLVM_DEBUG(llvm::dbgs() << "  [not pure] " << def->getName() << "\n");
    return false;
  }

  cache[def] = true;
  for (Value operand : def->getOperands()) {
    if (!isReachableHoistable(operand, block, earliestAlloc, cache, toMove)) {
      cache[def] = false;
      return false;
    }
  }

  toMove.insert(def);
  LLVM_DEBUG(llvm::dbgs() << "  [hoist candidate] " << def->getName() << "\n");
  return true;
}

struct HoistAllocSizeArithPass
    : public impl::HoistAllocSizeArithPassBase<HoistAllocSizeArithPass> {

  void runOnOperation() override;
};

void HoistAllocSizeArithPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;
  if (!funcOp.getBody().hasOneBlock())
    return;
  Block &block = funcOp.getBody().front();

  // Only used entry-block allocs are candidates. getOps (vs. block.walk) is
  // non-recursive on purpose: a recursive walk would also collect allocs
  // nested inside region-carrying ops (scf.for/scf.if), whose results live
  // in a different block — and the earliest-alloc selection and dominance
  // checks below use Operation::isBeforeInBlock, which asserts same-block.
  // An entry-block alloc's dynamic-size operands are themselves entry-block
  // (or block args) by SSA dominance, so the operand walk also stays in-block.
  SmallVector<memref::AllocOp> pooledAllocs;
  SmallVector<memref::AllocOp> dynAllocs;
  for (memref::AllocOp allocOp : block.getOps<memref::AllocOp>()) {
    if (allocOp.getResult().use_empty())
      continue;
    pooledAllocs.push_back(allocOp);
    if (!allocOp.getDynamicSizes().empty())
      dynAllocs.push_back(allocOp);
  }
  if (dynAllocs.empty())
    return;
  NumDynAllocsExamined += dynAllocs.size();

  Operation *earliestAlloc = *llvm::min_element(
      pooledAllocs, [](memref::AllocOp a, memref::AllocOp b) {
        return a->isBeforeInBlock(b);
      });

  LLVM_DEBUG({
    llvm::dbgs() << "[" DEBUG_TYPE "] @" << funcOp.getSymName() << ": "
                 << dynAllocs.size()
                 << " dynamic alloc(s); earliest used alloc = "
                 << earliestAlloc->getName() << "\n";
  });

  llvm::SetVector<Operation *> toMove;
  for (memref::AllocOp allocOp : dynAllocs) {
    for (Value dynOp : allocOp.getDynamicSizes()) {
      // Build each predecessor cone transactionally. If any dependency is not
      // hoistable, discard the complete cone instead of moving a successful
      // prefix that does not make the allocation size available any earlier.
      llvm::DenseMap<Operation *, bool> coneCache;
      llvm::SetVector<Operation *> cone;
      if (isReachableHoistable(dynOp, &block, earliestAlloc, coneCache, cone))
        for (Operation *op : cone)
          toMove.insert(op);
    }
  }
  if (toMove.empty()) {
    LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] no ops to hoist\n");
    markAllAnalysesPreserved();
    return;
  }

  // Forward iteration of toMove visits operands before uses (operands are
  // inserted after the recursion descends into them).  Each moveBefore
  // displaces the prior moves up by one slot, so the final layout above
  // earliestAlloc is exactly the topologically-ordered hoist cone.
  for (Operation *op : toMove)
    op->moveBefore(earliestAlloc);
  NumOpsHoisted += toMove.size();
  LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] hoisted " << toMove.size()
                          << " op(s)\n");
}

} // namespace
} // namespace hip
} // namespace mlir
