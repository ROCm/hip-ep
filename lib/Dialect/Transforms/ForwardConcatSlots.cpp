/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ForwardConcatSlots.cpp - Write producers into concat slots --------===//
//
// Bufferized concat copies every input into a subview of the concat buffer.
// A packed slot is one contiguous span. hip.packed_view names it with an
// identity memref whose aligned pointer is the first element of the slot,
// which is what every HIP lowering already loads, and
// hip-promote-strided-operands leaves an identity memref alone. The producer
// and every other reader of the private buffer are retargeted at the view,
// and the copy and the private alloc go away. A buffer that is copied into
// two concats is aliased into the first; the second copy stays.
//
// A concat whose buffer is only filled by such slots, and then copied in
// full into a packed prefix of another buffer, is rewritten to that prefix.
// Chained concats then write the outer buffer once.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <optional>
#include <utility>

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

// True when `use` writes `value` (a DPS init, the target of a copy, or a
// store). A source that nothing writes is uninitialized, and aliasing it
// would drop the copy that was its only contents.
static bool writesValue(OpOperand &use, Value value) {
  Operation *owner = use.getOwner();
  if (auto copy = dyn_cast<memref::CopyOp>(owner))
    return copy.getTarget() == value;
  if (auto store = dyn_cast<memref::StoreOp>(owner))
    return store.getMemRef() == value;
  auto dst = dyn_cast<DestinationStyleOpInterface>(owner);
  if (!dst)
    return false;
  for (int64_t i = 0, n = dst.getNumDpsInits(); i < n; ++i)
    if (dst.getDpsInitOperand(i) == &use)
      return true;
  return false;
}

// True when `base` is written only through a strict view, which is a concat
// filling its slots. A write to the whole buffer would clobber a slot that
// the aliased readers still expect to hold the source.
static bool onlySlotWrites(Value value, Value base, DenseSet<Value> &seen) {
  if (!seen.insert(value).second)
    return true;
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (isa<memref::DeallocOp, func::ReturnOp>(owner))
      continue;
    if (auto view = dyn_cast<ViewLikeOpInterface>(owner);
        view && view.getViewSource() == value) {
      for (Value result : owner->getResults())
        if (!onlySlotWrites(result, base, seen))
          return false;
      continue;
    }
    if (writesValue(use, value)) {
      if (value == base)
        return false;
      continue;
    }
    if (isa<memref::CopyOp, memref::LoadOp, DestinationStyleOpInterface>(owner))
      continue;
    auto effects = dyn_cast<MemoryEffectOpInterface>(owner);
    if (!effects || effects.getEffectOnValue<MemoryEffects::Write>(value))
      return false;
  }
  return true;
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

// The buffer under a chain of subviews and packed views, with the element
// offset of `value` from its first element. A subview's type offset already
// counts every subview below it, so only packed views add on top.
static std::optional<std::pair<Value, int64_t>> resolveRoot(Value value) {
  int64_t offset = 0;
  while (true) {
    if (auto sub = value.getDefiningOp<memref::SubViewOp>()) {
      int64_t subOffset = 0;
      SmallVector<int64_t, 6> strides;
      if (failed(cast<MemRefType>(sub.getType())
                     .getStridesAndOffset(strides, subOffset)) ||
          ShapedType::isDynamic(subOffset))
        return std::nullopt;
      offset += subOffset;
      value = sub.getSource();
      while (auto inner = value.getDefiningOp<memref::SubViewOp>())
        value = inner.getSource();
      continue;
    }
    if (auto view = value.getDefiningOp<PackedViewOp>()) {
      std::optional<int64_t> shift =
          getConstantIntValue(view.getElementOffset());
      if (!shift)
        return std::nullopt;
      offset += *shift;
      value = view.getSource();
      continue;
    }
    return std::make_pair(value, offset);
  }
}

struct Slot {
  memref::CopyOp copy;
  int64_t offset;
  int64_t size;
};

// Copies into packed spans of `value`, reached through subviews and packed
// views. Fails on a copy that is not a packed span in `block`.
static bool collectSlots(Value value, Block *block,
                         SmallVectorImpl<Slot> &slots) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (auto sub = dyn_cast<memref::SubViewOp>(owner)) {
      if (!collectSlots(sub.getResult(), block, slots))
        return false;
      continue;
    }
    if (auto view = dyn_cast<PackedViewOp>(owner)) {
      if (view.getSource() == value &&
          !collectSlots(view.getResult(), block, slots))
        return false;
      continue;
    }
    auto copy = dyn_cast<memref::CopyOp>(owner);
    if (!copy || copy.getTarget() != value)
      continue;
    auto type = cast<MemRefType>(value.getType());
    auto root = resolveRoot(value);
    if (copy->getBlock() != block || !root || !isPackedSlot(type))
      return false;
    slots.push_back({copy, root->second, type.getNumElements()});
  }
  return true;
}

// The last op in `block` that uses `value` or a view of it.
static Operation *lastUser(Value value, Block *block, Operation *latest) {
  for (Operation *user : value.getUsers()) {
    Operation *op = block->findAncestorOpInBlock(*user);
    if (op && (!latest || latest->isBeforeInBlock(op)))
      latest = op;
    if (isa<ViewLikeOpInterface>(user))
      for (Value result : user->getResults())
        latest = lastUser(result, block, latest);
  }
  return latest;
}

static bool writtenBetween(Value value, Operation *begin, Operation *end) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (!writesValue(use, value))
      continue;
    if (owner->getBlock() != begin->getBlock())
      return true;
    if (begin->isBeforeInBlock(owner) && owner->isBeforeInBlock(end))
      return true;
  }
  return false;
}

// A dense block concatenates every earlier output, so each concat repeats
// the previous one's inputs at the same offsets. Such a concat becomes a
// prefix of the larger one, whose duplicate copies go away; chained, the
// whole block shares one buffer and each input is copied once.
static bool foldIntoLargerConcat(memref::AllocOp alloc, IRRewriter &rewriter) {
  Value small = alloc.getResult();
  auto smallType = cast<MemRefType>(small.getType());
  if (!smallType.hasStaticShape() || !smallType.getLayout().isIdentity())
    return false;
  Block *block = alloc->getBlock();
  for (Operation *user : small.getUsers())
    if (isa<func::ReturnOp>(user))
      return false;
  SmallVector<Slot> smallSlots;
  if (!collectSlots(small, block, smallSlots) || smallSlots.empty())
    return false;
  llvm::sort(smallSlots,
             [](const Slot &a, const Slot &b) { return a.offset < b.offset; });
  int64_t covered = 0;
  for (const Slot &slot : smallSlots) {
    if (slot.offset != covered)
      return false;
    covered += slot.size;
  }
  int64_t smallElems = smallType.getNumElements();
  if (covered != smallElems)
    return false;
  DenseSet<Value> seenSmall;
  if (!onlySlotWrites(small, small, seenSmall))
    return false;

  Value first = smallSlots.front().copy.getSource();
  for (Operation *user : first.getUsers()) {
    auto other = dyn_cast<memref::CopyOp>(user);
    if (!other || other == smallSlots.front().copy ||
        other.getSource() != first)
      continue;
    auto root = resolveRoot(other.getTarget());
    if (!root || root->first == small)
      continue;
    Value large = root->first;
    auto largeType = dyn_cast<MemRefType>(large.getType());
    Operation *largeOp = large.getDefiningOp();
    if (!largeType || !largeType.hasStaticShape() ||
        !largeType.getLayout().isIdentity() || !largeOp ||
        largeOp->getBlock() != block ||
        !isa<memref::AllocOp, AllocOutputOp>(largeOp) ||
        largeType.getElementType() != smallType.getElementType() ||
        largeType.getNumElements() < smallElems)
      continue;
    SmallVector<Slot> largeSlots;
    if (!collectSlots(large, block, largeSlots))
      continue;

    // Every large slot inside the prefix repeats a small slot: the same
    // source and span, with the source unchanged between the two copies.
    // Both then land in the same bytes, so the later copy goes away.
    SmallVector<memref::CopyOp> duplicates;
    bool matches = true;
    for (Slot &slot : largeSlots) {
      if (slot.offset >= smallElems)
        continue;
      auto it = llvm::find_if(smallSlots, [&](Slot &s) {
        return s.offset == slot.offset && s.size == slot.size &&
               s.copy.getSource() == slot.copy.getSource();
      });
      if (it == smallSlots.end() || it->copy == slot.copy) {
        matches = false;
        break;
      }
      memref::CopyOp earlier = it->copy, later = slot.copy;
      if (later->isBeforeInBlock(earlier))
        std::swap(earlier, later);
      if (writtenBetween(slot.copy.getSource(), earlier, later)) {
        matches = false;
        break;
      }
      duplicates.push_back(later);
    }
    if (!matches || duplicates.size() != smallSlots.size())
      continue;
    DenseSet<Value> seenLarge;
    if (!onlySlotWrites(large, large, seenLarge))
      continue;
    bool dominates = llvm::all_of(largeOp->getOperands(), [&](Value v) {
      Operation *def = v.getDefiningOp();
      return !def || (def->getBlock() == block && def->isBeforeInBlock(alloc));
    });
    if (!dominates)
      continue;

    Operation *latest = lastUser(small, block, nullptr);
    if (!largeOp->isBeforeInBlock(alloc))
      largeOp->moveBefore(alloc);
    eraseDeallocs(small);
    rewriter.setInsertionPoint(alloc);
    Value zero = arith::ConstantIndexOp::create(rewriter, alloc.getLoc(), 0);
    auto view =
        PackedViewOp::create(rewriter, alloc.getLoc(), smallType, large, zero);
    small.replaceAllUsesWith(view.getResult());
    for (memref::CopyOp copy : duplicates)
      copy.erase();
    alloc.erase();
    for (Operation *user : llvm::to_vector(large.getUsers()))
      if (auto dealloc = dyn_cast<memref::DeallocOp>(user))
        if (latest && dealloc->isBeforeInBlock(latest))
          dealloc->moveAfter(latest);
    return true;
  }
  return false;
}

// Point every user of the copied alloc at an identity view of the packed
// slot, so a later reader sees the same bytes the concat does.
static bool aliasSlot(memref::CopyOp copy, IRRewriter &rewriter) {
  auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
  if (!alloc || copy->getBlock() != alloc->getBlock())
    return false;
  auto srcType = dyn_cast<MemRefType>(alloc.getType());
  auto slot = copy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!srcType || !srcType.getLayout().isIdentity() ||
      !srcType.hasStaticShape() || !slot)
    return false;
  auto slotType = cast<MemRefType>(slot.getType());
  if (!sameShape(srcType, slotType) || !isPackedSlot(slotType))
    return false;
  if (definedFrom(slot.getResult(), alloc.getResult()))
    return false;

  auto root = resolveRoot(slot.getResult());
  if (!root)
    return false;
  Value base = root->first;
  int64_t elementOffset = root->second;
  auto baseType = dyn_cast<MemRefType>(base.getType());
  Operation *baseOp = base.getDefiningOp();
  if (!baseType || !baseType.hasStaticShape() ||
      !baseType.getLayout().isIdentity() || !baseOp ||
      baseOp->getBlock() != alloc->getBlock())
    return false;
  if (!isa<memref::AllocOp, AllocOutputOp>(baseOp))
    return false;
  if (baseType.getElementType() != srcType.getElementType())
    return false;
  DenseSet<Value> seen;
  if (!onlySlotWrites(base, base, seen))
    return false;

  int64_t slotElems = srcType.getNumElements();
  int64_t baseElems = baseType.getNumElements();
  if (elementOffset < 0 || slotElems > baseElems - elementOffset)
    return false;

  Block *block = alloc->getBlock();
  bool sawWriter = false;
  Operation *earliest = nullptr;
  Operation *latest = nullptr;
  for (OpOperand &use : alloc.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (owner == copy || isa<memref::DeallocOp>(owner))
      continue;
    // Returning the private buffer would alias it with the concat result.
    if (isa<func::ReturnOp>(owner) || owner->getBlock() != block)
      return false;
    // A write after the copy would now land in the concat result.
    if (writesValue(use, alloc.getResult())) {
      if (!owner->isBeforeInBlock(copy))
        return false;
      sawWriter = true;
    }
    if (!earliest || owner->isBeforeInBlock(earliest))
      earliest = owner;
    if (!latest || latest->isBeforeInBlock(owner))
      latest = owner;
  }
  if (!sawWriter || !earliest)
    return false;

  // The view has to dominate the writer, which sits above the copy. The
  // concat buffer moves up with it. Its operands are a context and, for a
  // static buffer, nothing else, so they already dominate.
  for (Value operand : baseOp->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (def && (def->getBlock() != block || !def->isBeforeInBlock(earliest)))
      return false;
  }
  SmallVector<memref::DeallocOp> baseDeallocs;
  for (OpOperand &use : base.getUses()) {
    auto dealloc = dyn_cast<memref::DeallocOp>(use.getOwner());
    if (!dealloc)
      continue;
    if (dealloc->getBlock() != block)
      return false;
    baseDeallocs.push_back(dealloc);
  }

  if (!baseOp->isBeforeInBlock(earliest))
    baseOp->moveBefore(earliest);

  Location loc = earliest->getLoc();
  rewriter.setInsertionPoint(earliest);
  Value shift = arith::ConstantIndexOp::create(rewriter, loc, elementOffset);
  auto view = PackedViewOp::create(rewriter, loc, srcType, base, shift);

  SmallVector<OpOperand *> retarget;
  for (OpOperand &use : alloc.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (owner == copy || isa<memref::DeallocOp>(owner))
      continue;
    retarget.push_back(&use);
  }
  for (OpOperand *use : retarget)
    use->set(view.getResult());

  // Readers can sit after the concat buffer's dealloc. Keep the buffer live
  // through the last one.
  for (memref::DeallocOp dealloc : baseDeallocs)
    if (dealloc->isBeforeInBlock(latest))
      dealloc->moveAfter(latest);

  eraseDeallocs(alloc.getResult());
  copy.erase();
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

    bool folded = true;
    while (folded) {
      folded = false;
      SmallVector<memref::AllocOp> allocs;
      func.walk([&](memref::AllocOp alloc) { allocs.push_back(alloc); });
      for (memref::AllocOp alloc : allocs) {
        if (foldIntoLargerConcat(alloc, rewriter)) {
          folded = true;
          break;
        }
      }
    }

    SmallVector<memref::CopyOp> copies;
    func.walk([&](memref::CopyOp copy) { copies.push_back(copy); });
    for (memref::CopyOp copy : copies)
      (void)aliasSlot(copy, rewriter);

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
