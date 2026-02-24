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
// Compatibility rules:
//   - Static shapes: any alloc whose byte-size fits inside an earlier slot
//     (same element type and memory space).  A memref.reinterpret_cast is
//     inserted when the shapes differ.
//   - Dynamic shapes: the MemRefType must be identical and every dynamic-size
//     operand must be the same SSA value.
//
//===----------------------------------------------------------------------===//

#include "HipPasses.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_OPTIMIZEMEMREFSPASS
#include "HipPasses.h.inc"

namespace {

struct AllocInterval {
  memref::AllocOp allocOp;
  unsigned defIndex;
  unsigned lastUseIndex;
};

struct Slot {
  Value buffer;
  MemRefType type;
  unsigned lastUseIndex;
  SmallVector<Value, 4> dynamicSizes;
};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

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
/// type and memory space).
/// Dynamic shapes: the MemRefType must be identical and every dynamic-size
/// operand must be the same SSA value.
static bool canReuse(const Slot& slot, MemRefType neededType,
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
    if (slot.dynamicSizes.size() !=
        static_cast<size_t>(neededDynSizes.size()))
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
static bool isMemRefAlias(Operation* user) {
  return isa<ViewLikeOpInterface, arith::SelectOp>(user);
}

/// Returns the highest operation index among all transitive users of \p value,
/// following view-like and select ops so that the liveness of the source
/// buffer accounts for all its derived views.
static unsigned findLastTransitiveUseIndex(
    Value value, Block& block,
    const DenseMap<Operation*, unsigned>& opIndex,
    unsigned blockSize) {
  unsigned lastIdx = 0;
  SmallVector<Value> worklist = {value};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation* user : current.getUsers()) {
      auto it = opIndex.find(user);
      unsigned userIdx;
      if (it != opIndex.end()) {
        userIdx = it->second;
      } else if (auto* ancestor = block.findAncestorOpInBlock(*user)) {
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

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

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

  Block& block = funcOp.getBody().front();

  // Assign each op a sequential index for interval ordering.
  DenseMap<Operation*, unsigned> opIndex;
  unsigned idx = 0;
  for (Operation& op : block)
    opIndex[&op] = idx++;
  unsigned blockSize = idx;

  // Collect alloc ops (static and dynamic) and compute their live intervals.
  // Liveness follows view-like and select ops transitively so that derived
  // views extend the source buffer's lifetime.
  SmallVector<AllocInterval> intervals;
  for (Operation& op : block) {
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

  // Greedy best-fit slot assignment: pick the smallest sufficient dead slot
  // whose live range ended strictly before this alloc.
  SmallVector<Slot> slots;
  SmallVector<std::pair<Value, Value>> replacements;

  for (auto& interval : intervals) {
    MemRefType neededType = interval.allocOp.getType();
    auto neededDynSizes = interval.allocOp.getDynamicSizes();

    Slot* bestSlot = nullptr;
    int64_t bestWaste = INT64_MAX;
    for (auto& slot : slots) {
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
      slots.push_back(
          {interval.allocOp.getResult(), neededType, interval.lastUseIndex,
           SmallVector<Value, 4>(neededDynSizes.begin(),
                                 neededDynSizes.end())});
    }
  }

  // Apply all replacements and erase dead alloc ops.
  for (auto [oldVal, newVal] : replacements) {
    oldVal.replaceAllUsesWith(newVal);
    oldVal.getDefiningOp()->erase();
  }
}

}  // namespace
}  // namespace hip
}  // namespace mlir
