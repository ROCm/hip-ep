/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- OptimizeMemRefs.cpp - Buffer reuse via liveness analysis -----------===//
//
// Greedy best-fit buffer reuse for single-block functions.  Replaces
// memref.alloc ops whose live ranges don't overlap with earlier compatible
// allocations, reducing peak device-memory usage.
//
// Algorithm (illustrated with single-head attention, 8 allocs after
// bufferization):
//
//   1. Index every op in the entry block sequentially: op0->0, op1->1, ...
//
//   2. For each memref.alloc, compute a live interval [def, lastUse]:
//        - def     = index of the alloc op
//        - lastUse = max index among all transitive users (follows aliasing
//                    ops via BufferViewFlowAnalysis)
//
//   3. Walk intervals in program order.  For each one, try to find a
//      previously-created "slot" whose lastUse < this def (i.e., dead):
//        - Among compatible dead slots, pick the one with smallest byte
//          waste (best-fit).  Insert memref.reinterpret_cast if shapes differ.
//        - If no compatible slot exists, create a new one.
//
//   4. Replace all reused allocs with their assigned slot buffer and erase
//      the dead alloc ops.
//
//   Result for attention: 8 allocs -> 4 allocs (50% reduction).
//
// Compatibility rules:
//   - Static shapes: any alloc whose byte-size fits inside an earlier slot
//     (same element type and memory space).  A memref.reinterpret_cast is
//     inserted when the shapes differ.
//   - Dynamic shapes: the MemRefType must be identical and every dynamic-size
//     operand must be the same SSA value (not merely value-equivalent --
//     proving value-equivalence in general would require alias analysis;
//     SSA identity is sound and cheap, and CSE should run first to
//     maximize opportunities).
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/BufferUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-optimize-memrefs"

STATISTIC(NumAllocsReused, "Number of allocations reused via slot assignment");
STATISTIC(NumSlotsCreated, "Number of unique memory slots created");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_OPTIMIZEMEMREFSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// A memref.alloc and its live range within the entry block.
struct AllocInterval {
  memref::AllocOp allocOp;
  unsigned defIndex;     ///< Block index where the alloc occurs.
  unsigned lastUseIndex; ///< Block index of the last (transitive) use.
};

/// A reusable memory slot: tracks the buffer, its type, when it becomes
/// dead, and (for dynamic shapes) the SSA values that determine its size.
struct Slot {
  Value buffer;
  MemRefType type;
  unsigned lastUseIndex; ///< Updated when the slot is reused.
  SmallVector<Value, 4> dynamicSizes;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Computes contiguous row-major strides for a statically-shaped memref.
static SmallVector<int64_t> getContiguousStrides(MemRefType type) {
  auto shape = type.getShape();
  SmallVector<int64_t> strides(shape.size());
  int64_t stride = 1;
  for (int64_t i = shape.size() - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= shape[i];
  }
  return strides;
}

/// Returns true if \p slot can serve an allocation of \p neededType with
/// the given dynamic-size operands.
static bool canReuse(const Slot &slot, MemRefType neededType,
                     OperandRange neededDynSizes) {
  if (slot.type.getElementType() != neededType.getElementType())
    return false;
  if (slot.type.getMemorySpace() != neededType.getMemorySpace())
    return false;

  int64_t slotBytes = getStaticByteSize(slot.type);
  int64_t neededBytes = getStaticByteSize(neededType);

  if (slotBytes > 0 && neededBytes > 0)
    return slotBytes >= neededBytes;

  if (slotBytes == 0 && neededBytes == 0 && slot.type == neededType) {
    if (slot.dynamicSizes.size() != static_cast<size_t>(neededDynSizes.size()))
      return false;
    for (auto [slotDyn, neededDyn] :
         llvm::zip(slot.dynamicSizes, neededDynSizes))
      if (slotDyn != neededDyn)
        return false;
    return true;
  }

  return false;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct OptimizeMemRefsPass
    : public impl::OptimizeMemRefsPassBase<OptimizeMemRefsPass> {

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<memref::MemRefDialect>();
    arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override;
};

void OptimizeMemRefsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.empty())
    return;

  // TODO: Generalize to multi-block functions using MLIR's Liveness analysis
  // instead of sequential op indices.
  if (!funcOp.getBody().hasOneBlock()) {
    funcOp.emitError("hip-optimize-memrefs requires single-block functions; "
                     "liveness analysis uses sequential op indices that do "
                     "not generalize to control flow");
    return signalPassFailure();
  }

  Block &block = funcOp.getBody().front();

  BufferViewFlowAnalysis aliasAnalysis(funcOp);

  // Assign each op a sequential index for interval ordering.
  DenseMap<Operation *, unsigned> opIndex;
  unsigned idx = 0;
  for (Operation &op : block)
    opIndex[&op] = idx++;
  unsigned blockSize = idx;

  // Collect alloc ops and compute their live intervals.
  SmallVector<AllocInterval> intervals;
  for (Operation &op : block) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    if (!allocOp)
      continue;

    Value result = allocOp.getResult();
    if (result.use_empty())
      continue;

    unsigned lastUse = findLastAliasedUseIndex(result, aliasAnalysis, block,
                                               opIndex, blockSize);
    intervals.push_back({allocOp, opIndex[&op], lastUse});
    LLVM_DEBUG(llvm::dbgs() << "  interval " << allocOp << " [" << opIndex[&op]
                            << ", " << lastUse << "]\n");
  }

  if (intervals.size() < 2)
    return;

  // Greedy best-fit slot assignment.
  SmallVector<Slot> slots;
  SmallVector<std::pair<Value, Value>> replacements;

  for (auto &interval : intervals) {
    MemRefType neededType = interval.allocOp.getType();
    auto neededDynSizes = interval.allocOp.getDynamicSizes();

    Slot *bestSlot = nullptr;
    int64_t bestWaste = INT64_MAX;
    for (auto &slot : slots) {
      if (slot.lastUseIndex >= interval.defIndex)
        continue;
      if (!canReuse(slot, neededType, neededDynSizes))
        continue;
      int64_t waste =
          getStaticByteSize(slot.type) - getStaticByteSize(neededType);
      if (waste < bestWaste) {
        bestWaste = waste;
        bestSlot = &slot;
        if (waste == 0)
          break;
      }
    }

    if (bestSlot) {
      Value replacement = bestSlot->buffer;
      if (bestSlot->type != neededType) {
        OpBuilder builder(interval.allocOp);
        replacement = memref::ReinterpretCastOp::create(
            builder, interval.allocOp.getLoc(), neededType, bestSlot->buffer,
            /*offset=*/int64_t(0),
            /*sizes=*/neededType.getShape(),
            /*strides=*/getContiguousStrides(neededType));
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "  Reusing slot (type=" << bestSlot->type
                 << ", waste=" << bestWaste << ") for " << neededType << "\n");
      replacements.emplace_back(interval.allocOp.getResult(), replacement);
      bestSlot->lastUseIndex = interval.lastUseIndex;
      ++NumAllocsReused;
    } else {
      LLVM_DEBUG(llvm::dbgs() << "  New slot for " << neededType << "\n");
      slots.push_back({interval.allocOp.getResult(), neededType,
                       interval.lastUseIndex,
                       SmallVector<Value, 4>(neededDynSizes.begin(),
                                             neededDynSizes.end())});
      ++NumSlotsCreated;
    }
  }

  // Map each alloc to its memref.dealloc (if any) so we can erase the
  // dealloc when the alloc it targets is replaced.
  DenseMap<Value, memref::DeallocOp> allocToDealloc;
  for (Operation &op : block) {
    if (auto deallocOp = dyn_cast<memref::DeallocOp>(op))
      allocToDealloc[deallocOp.getMemref()] = deallocOp;
  }

  // Apply all replacements and erase dead alloc ops.
  for (auto [oldVal, newVal] : replacements) {
    if (auto it = allocToDealloc.find(oldVal); it != allocToDealloc.end()) {
      it->second.erase();
      allocToDealloc.erase(it);
    }

    // When a returned alloc (no dealloc) is merged into a slot that has a
    // dealloc, the slot becomes the returned buffer and must not be freed.
    bool oldIsReturned = llvm::any_of(oldVal.getUsers(), [](Operation *user) {
      return isa<func::ReturnOp>(user);
    });
    if (oldIsReturned) {
      if (auto it = allocToDealloc.find(newVal); it != allocToDealloc.end()) {
        it->second.erase();
        allocToDealloc.erase(it);
      }
    }

    oldVal.replaceAllUsesWith(newVal);
    oldVal.getDefiningOp()->erase();
  }
}

} // namespace
} // namespace hip
} // namespace mlir
