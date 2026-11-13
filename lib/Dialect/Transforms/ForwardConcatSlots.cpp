/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ForwardConcatSlots.cpp - Write producers into concat slots --------===//
//
// Bufferized concat copies every input into a subview of the concat buffer.
// hip.rocmlir and hip.resize already store a dense tensor from the pointer
// they are given, so a slot that is one packed span can be that pointer.
// The private output alloc and the memref.copy both go away.
//
// A concat whose buffer is only filled by such slots, and then copied in
// full into a packed prefix of another buffer, is rewritten to that prefix.
// Chained concats then write the outer buffer once.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_FORWARDCONCATSLOTSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// A packed row-major walk of `type` hits consecutive addresses. A size-1
// dimension may carry the parent's stride; every other stride must be the
// product of the sizes after it. That is the batch-1 channel slot, and any
// other concat slice with no gaps. A strided slot (batch > 1 on a middle
// axis, or a concat of rows) fails and keeps its copy.
static bool isPackedSlot(MemRefType type) {
  if (!type.hasStaticShape())
    return false;
  int64_t offset = 0;
  SmallVector<int64_t, 6> strides;
  if (failed(type.getStridesAndOffset(strides, offset)) ||
      ShapedType::isDynamic(offset))
    return false;
  int64_t expect = 1;
  for (int64_t i = type.getRank() - 1; i >= 0; --i) {
    int64_t stride = strides[i];
    int64_t size = type.getDimSize(i);
    if (ShapedType::isDynamic(stride) || (size != 1 && stride != expect))
      return false;
    expect *= size;
  }
  return true;
}

static bool sameShape(MemRefType a, MemRefType b) {
  return a.getRank() == b.getRank() && a.getShape() == b.getShape() &&
         a.getElementType() == b.getElementType();
}

static bool definedFrom(Value value, Value root) {
  while (value) {
    if (value == root)
      return true;
    auto sub = value.getDefiningOp<memref::SubViewOp>();
    if (!sub)
      return false;
    value = sub.getSource();
  }
  return false;
}

// True when every use writes through a subview (or is a dealloc). `ignored`
// is the full-buffer copy about to be erased; its source use does not count.
static bool isWriteOnlyView(Value value, Operation *ignored,
                            DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return true;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == ignored)
      continue;
    if (isa<memref::DeallocOp>(owner))
      continue;
    if (auto copy = dyn_cast<memref::CopyOp>(owner)) {
      if (copy.getTarget() == value)
        continue;
      return false;
    }
    if (auto sub = dyn_cast<memref::SubViewOp>(owner)) {
      if (sub.getSource() == value &&
          isWriteOnlyView(sub.getResult(), ignored, seen))
        continue;
    }
    return false;
  }
  return true;
}

static bool isSlotWriter(OpOperand &use) {
  if (!isa<RocMlirOp, ResizeOp>(use.getOwner()))
    return false;
  auto dst = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
  if (!dst)
    return false;
  for (int64_t i = 0, n = dst.getNumDpsInits(); i < n; ++i)
    if (dst.getDpsInitOperand(i) == &use)
      return true;
  return false;
}

static void eraseDeallocs(Value value) {
  SmallVector<Operation *> deallocs;
  for (OpOperand &use : value.getUses())
    if (isa<memref::DeallocOp>(use.getOwner()))
      deallocs.push_back(use.getOwner());
  for (Operation *op : deallocs)
    op->erase();
}

// Replace a concat buffer that is copied wholesale into a packed prefix.
// Subviews of the old buffer are rebuilt against the prefix so their result
// types carry the outer strides; swapping the source operand would leave the
// old type in place and fail verification.
static bool aliasPrefix(memref::CopyOp copy, IRRewriter &rewriter) {
  auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
  auto dest = copy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!alloc || !dest)
    return false;
  auto srcType = cast<MemRefType>(alloc.getType());
  auto destType = cast<MemRefType>(dest.getType());
  if (!sameShape(srcType, destType) || !isPackedSlot(destType))
    return false;
  if (definedFrom(dest.getResult(), alloc.getResult()))
    return false;
  DenseSet<Value> seen;
  if (!isWriteOnlyView(alloc.getResult(), copy, seen))
    return false;

  SmallVector<memref::SubViewOp> subs;
  for (OpOperand &use : alloc.getResult().getUses()) {
    auto sub = dyn_cast<memref::SubViewOp>(use.getOwner());
    if (sub && sub.getSource() == alloc.getResult())
      subs.push_back(sub);
  }
  for (memref::SubViewOp sub : subs) {
    // The prefix is defined at the outer copy, which may be after the slot
    // subview. Rebuild the slot there so the prefix dominates it.
    rewriter.setInsertionPointAfter(dest);
    auto rebuilt = memref::SubViewOp::create(
        rewriter, sub.getLoc(), dest.getResult(), sub.getMixedOffsets(),
        sub.getMixedSizes(), sub.getMixedStrides());
    Value slot = rebuilt.getResult();
    // A full view of the prefix is the prefix. Drop the extra subview.
    if (rebuilt.getType() == dest.getType()) {
      slot = dest.getResult();
      rewriter.eraseOp(rebuilt);
    }
    sub.getResult().replaceAllUsesWith(slot);
    // A slot copy can sit between the old subview and the prefix. It now
    // reads the rebuilt slot, so it has to follow that slot.
    if (Operation *slotOp = slot.getDefiningOp()) {
      SmallVector<Operation *> users(slotOp->getUsers());
      for (Operation *user : users)
        if (user->getBlock() == slotOp->getBlock() &&
            user->isBeforeInBlock(slotOp))
          user->moveAfter(slotOp);
    }
    rewriter.eraseOp(sub);
  }

  eraseDeallocs(alloc.getResult());
  copy.erase();
  alloc.erase();
  return true;
}

// Retarget hip.rocmlir / hip.resize at the packed slot they were copied into.
static bool forwardProducer(memref::CopyOp copy) {
  auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
  if (!alloc)
    return false;
  auto destType = dyn_cast<MemRefType>(copy.getTarget().getType());
  auto srcType = cast<MemRefType>(alloc.getType());
  if (!destType || !sameShape(srcType, destType) || !isPackedSlot(destType))
    return false;
  if (definedFrom(copy.getTarget(), alloc.getResult()))
    return false;

  Operation *writer = nullptr;
  for (OpOperand &use : alloc.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (owner == copy || isa<memref::DeallocOp>(owner))
      continue;
    if (!isSlotWriter(use) || writer)
      return false;
    writer = owner;
  }
  if (!writer)
    return false;

  // The slot is defined at the copy, which is after the writer. Move the
  // writer down so the slot dominates its output operand.
  writer->moveBefore(copy);
  auto dst = cast<DestinationStyleOpInterface>(writer);
  for (int64_t i = 0, n = dst.getNumDpsInits(); i < n; ++i)
    if (dst.getDpsInitOperand(i)->get() == alloc.getResult())
      dst.setDpsInitOperand(i, copy.getTarget());

  eraseDeallocs(alloc.getResult());
  copy.erase();
  if (alloc.use_empty())
    alloc.erase();
  return true;
}

struct ForwardConcatSlotsPass
    : public impl::ForwardConcatSlotsPassBase<ForwardConcatSlotsPass> {
  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(func.getContext());
    bool aliased = true;
    while (aliased) {
      aliased = false;
      SmallVector<memref::CopyOp> copies;
      func.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
      for (memref::CopyOp copy : copies) {
        if (aliasPrefix(copy, rewriter)) {
          aliased = true;
          break;
        }
      }
    }

    SmallVector<memref::CopyOp> copies;
    func.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
    for (memref::CopyOp copy : copies)
      (void)forwardProducer(copy);

    SmallVector<memref::SubViewOp> dead;
    func.walk([&](memref::SubViewOp sub) {
      if (sub.use_empty())
        dead.push_back(sub);
    });
    for (memref::SubViewOp sub : dead)
      sub.erase();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
