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
//                    ops like subview, cast, reshape so that a derived view
//                    extends the source buffer's lifetime)
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

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

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

/// Returns the allocation size in bytes for a fully-static memref type.
/// Returns 0 when any dimension is dynamic.
static int64_t getStaticByteSize(MemRefType type) {
  if (!type.hasStaticShape())
    return 0;
  return type.getNumElements() * type.getElementTypeBitWidth() / 8;
}

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
///
/// Static shapes: the slot must have at least as many bytes (same element
/// type and memory space).  Examples (slot holds memref<2x64x64xf32>,
/// 32768 bytes):
///   canReuse(memref<64xf32>)       -> true  (256 <= 32768, same f32)
///   canReuse(memref<2x64x64xf32>)  -> true  (exact fit, no cast needed)
///   canReuse(memref<3x64x64xf32>)  -> false (49152 > 32768)
///   canReuse(memref<64xf16>)       -> false (different element type)
///
/// Dynamic shapes: the MemRefType must be identical and every dynamic-size
/// operand must be the same SSA value.
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
    for (unsigned i = 0; i < slot.dynamicSizes.size(); ++i)
      if (slot.dynamicSizes[i] != neededDynSizes[i])
        return false;
    return true;
  }

  return false;
}

/// Returns true if \p user produces a memref result that aliases its memref
/// operand.  Covers ViewLikeOpInterface (subview, cast, reshape, etc.) and
/// arith::SelectOp whose result may alias either operand.
static bool isMemRefAlias(Operation *user) {
  return isa<ViewLikeOpInterface, arith::SelectOp>(user);
}

/// Returns the highest operation index among all transitive users of \p value,
/// following view-like and select ops so that the liveness of the source
/// buffer accounts for all its derived views.
///
/// memref.dealloc ops are excluded: they are administrative, not data uses,
/// and counting them would extend every lifetime to the block's end when
/// buffer-deallocation-pipeline has been run, defeating buffer reuse.
///
/// Example:
///   %buf  = memref.alloc() : memref<2x64x64xf32>           // index 3
///   %sv   = memref.subview %buf[...] : ... to memref<...>   // alias, index 5
///   use(%sv)                                                 // index 15
///   -> buf.lastUse = 15 (not 3), preventing premature reuse
static unsigned
findLastTransitiveUseIndex(Value value, Block &block,
                           const DenseMap<Operation *, unsigned> &opIndex,
                           unsigned blockSize) {
  unsigned lastIdx = 0;
  SmallVector<Value> worklist = {value};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation *user : current.getUsers()) {
      if (isa<memref::DeallocOp>(user))
        continue;

      auto it = opIndex.find(user);
      unsigned userIdx;
      if (it != opIndex.end()) {
        userIdx = it->second;
      } else if (auto *ancestor = block.findAncestorOpInBlock(*user)) {
        userIdx = opIndex.lookup(ancestor);
      } else {
        userIdx = blockSize - 1;
      }
      lastIdx = std::max(lastIdx, userIdx);

      if (isMemRefAlias(user)) {
        for (Value result : user->getResults())
          if (isa<MemRefType>(result.getType()))
            worklist.push_back(result);
      }
    }
  }
  return lastIdx;
}

//===----------------------------------------------------------------------===//
// Pass
//===----------------------------------------------------------------------===//

struct OptimizeMemRefsPass
    : public impl::OptimizeMemRefsPassBase<OptimizeMemRefsPass> {
  void runOnOperation() override;
};

void OptimizeMemRefsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.empty())
    return;

  if (!funcOp.getBody().hasOneBlock())
    return;

  Block &block = funcOp.getBody().front();

  // Assign each op a sequential index for interval ordering.
  DenseMap<Operation *, unsigned> opIndex;
  unsigned idx = 0;
  for (Operation &op : block)
    opIndex[&op] = idx++;
  unsigned blockSize = idx;

  // Collect alloc ops (static and dynamic) and compute their live intervals.
  // Liveness follows view-like and select ops transitively so that derived
  // views extend the source buffer's lifetime.
  SmallVector<AllocInterval> intervals;
  for (Operation &op : block) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    if (!allocOp)
      continue;

    Value result = allocOp.getResult();
    if (result.use_empty())
      continue;

    intervals.push_back(
        {allocOp, opIndex[&op],
         findLastTransitiveUseIndex(result, block, opIndex, blockSize)});
  }

  if (intervals.size() < 2)
    return;

  // Greedy best-fit slot assignment.  For each interval in program order,
  // find a dead slot (slot.lastUse < interval.def) with the smallest byte
  // waste.  Best-fit minimizes internal fragmentation; early exit on exact
  // match (waste == 0) avoids unnecessary scanning.
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
      replacements.emplace_back(interval.allocOp.getResult(), replacement);
      bestSlot->lastUseIndex = interval.lastUseIndex;
    } else {
      slots.push_back({interval.allocOp.getResult(), neededType,
                       interval.lastUseIndex,
                       SmallVector<Value, 4>(neededDynSizes.begin(),
                                             neededDynSizes.end())});
    }
  }

  // Map each alloc to its memref.dealloc (if any) so we can erase the
  // dealloc when the alloc it targets is replaced.  Without this, RAUW
  // would turn `memref.dealloc %replaced` into `memref.dealloc %reuser`,
  // causing a double-free.
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
