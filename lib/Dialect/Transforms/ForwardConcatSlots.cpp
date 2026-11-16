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

using Span = std::pair<int64_t, int64_t>;

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

static bool overlaps(Span a, Span b) {
  return a.first < b.second && b.first < a.second;
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

static bool isViewOf(Operation *op, Value value) {
  auto view = dyn_cast<ViewLikeOpInterface>(op);
  return view && view.getViewSource() == value;
}

// False only when `use` provably leaves the bytes of its value unchanged.
// A view is not a write; its own uses are visited separately.
static bool mayWrite(OpOperand &use) {
  Value value = use.get();
  Operation *owner = use.getOwner();
  if (writesValue(use, value))
    return true;
  if (isViewOf(owner, value) ||
      isa<memref::DeallocOp, func::ReturnOp, memref::LoadOp, memref::DimOp,
          memref::CopyOp, DestinationStyleOpInterface>(owner))
    return false;
  auto effects = dyn_cast<MemoryEffectOpInterface>(owner);
  return !effects ||
         effects.getEffectOnValue<MemoryEffects::Write>(value).has_value();
}

// Every use of `value` and of every view derived from it.
static void collectAliasUses(Value value, SmallVectorImpl<OpOperand *> &uses) {
  for (OpOperand &use : value.getUses()) {
    uses.push_back(&use);
    if (isViewOf(use.getOwner(), value))
      for (Value result : use.getOwner()->getResults())
        collectAliasUses(result, uses);
  }
}

// The buffer `value` views, through any view op.
static Value viewRoot(Value value) {
  while (auto view = value.getDefiningOp<ViewLikeOpInterface>())
    value = view.getViewSource();
  return value;
}

static void eraseDeallocs(Value value) {
  SmallVector<Operation *> deallocs;
  for (OpOperand &use : value.getUses())
    if (isa<memref::DeallocOp>(use.getOwner()))
      deallocs.push_back(use.getOwner());
  for (Operation *op : deallocs)
    op->erase();
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

// The elements of `root` that `value` can reach: the first one and one past
// the last, which for a strided view bounds the gaps as well.
static std::optional<Span> spanIn(Value value, Value root) {
  auto resolved = resolveRoot(value);
  auto type = dyn_cast<MemRefType>(value.getType());
  if (!resolved || resolved->first != root || !type || !type.hasStaticShape())
    return std::nullopt;
  int64_t typeOffset = 0;
  SmallVector<int64_t, 6> strides;
  if (failed(type.getStridesAndOffset(strides, typeOffset)))
    return std::nullopt;
  int64_t begin = resolved->second;
  int64_t extent = 1;
  for (auto [size, stride] : llvm::zip(type.getShape(), strides)) {
    if (size == 0)
      return Span{begin, begin};
    if (ShapedType::isDynamic(stride))
      return std::nullopt;
    extent += (size - 1) * stride;
  }
  return Span{begin, begin + extent};
}

struct Write {
  Operation *op;
  Span span;
};

// Every op that may write `root`, through any view, with the elements it can
// reach. Fails when a write goes through a view whose span is unknown.
static bool collectWrites(Value root, SmallVectorImpl<Write> &writes) {
  SmallVector<OpOperand *> uses;
  collectAliasUses(root, uses);
  for (OpOperand *use : uses) {
    if (!mayWrite(*use))
      continue;
    std::optional<Span> span = spanIn(use->get(), root);
    if (!span)
      return false;
    writes.push_back({use->getOwner(), *span});
  }
  return true;
}

// True when anything may write the buffer under `value`, through any alias,
// between `begin` and `end`. Writes it cannot place count as in between.
static bool writtenBetween(Value value, Operation *begin, Operation *end) {
  Block *block = begin->getBlock();
  SmallVector<OpOperand *> uses;
  collectAliasUses(viewRoot(value), uses);
  for (OpOperand *use : uses) {
    if (!mayWrite(*use))
      continue;
    Operation *op = block->findAncestorOpInBlock(*use->getOwner());
    if (!op || (begin->isBeforeInBlock(op) && op->isBeforeInBlock(end)))
      return true;
  }
  return false;
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

// Replace a concat buffer that is copied wholesale into a packed prefix.
// The buffer must only be filled, through its own slot subviews, before that
// copy. The prefix view moves above the first fill and each slot is rebuilt
// against it, so the fills keep their place and order; swapping the source
// operand would leave the old subview type in place and fail verification.
static bool aliasPrefix(memref::CopyOp copy, IRRewriter &rewriter) {
  auto alloc = copy.getSource().getDefiningOp<memref::AllocOp>();
  auto dest = copy.getTarget().getDefiningOp<memref::SubViewOp>();
  if (!alloc || !dest || !dest->hasOneUse())
    return false;
  Block *block = copy->getBlock();
  if (alloc->getBlock() != block || dest->getBlock() != block)
    return false;
  auto srcType = cast<MemRefType>(alloc.getType());
  auto destType = cast<MemRefType>(dest.getType());
  if (!sameShape(srcType, destType) || !isPackedSlot(destType))
    return false;
  if (definedFrom(dest.getResult(), alloc.getResult()))
    return false;

  SmallVector<memref::SubViewOp> subs;
  Operation *earliest = nullptr;
  for (OpOperand &use : alloc.getResult().getUses()) {
    Operation *owner = use.getOwner();
    if (owner == copy || isa<memref::DeallocOp>(owner))
      continue;
    auto sub = dyn_cast<memref::SubViewOp>(owner);
    if (!sub || sub->getBlock() != block)
      return false;
    for (OpOperand &subUse : sub.getResult().getUses()) {
      auto fill = dyn_cast<memref::CopyOp>(subUse.getOwner());
      if (!fill || fill.getTarget() != sub.getResult() ||
          fill->getBlock() != block || !fill->isBeforeInBlock(copy))
        return false;
    }
    subs.push_back(sub);
    if (!earliest || sub->isBeforeInBlock(earliest))
      earliest = sub;
  }
  if (!earliest)
    return false;

  // The fills now land in the outer buffer before the copy would have, so
  // nothing else may touch the prefix until then.
  auto root = resolveRoot(dest.getResult());
  if (!root)
    return false;
  std::optional<Span> prefix = spanIn(dest.getResult(), root->first);
  if (!prefix)
    return false;
  SmallVector<OpOperand *> outerUses;
  collectAliasUses(root->first, outerUses);
  for (OpOperand *use : outerUses) {
    Operation *owner = use->getOwner();
    if (owner == copy || isViewOf(owner, use->get()) ||
        isa<memref::DeallocOp, func::ReturnOp, memref::DimOp>(owner))
      continue;
    Operation *op = block->findAncestorOpInBlock(*owner);
    if (!op)
      return false;
    if (!op->isBeforeInBlock(copy))
      continue;
    std::optional<Span> span = spanIn(use->get(), root->first);
    if (!span || overlaps(*span, *prefix))
      return false;
  }

  // The prefix has to dominate the first fill. Its source may be the outer
  // concat buffer, which moves up with it.
  SmallVector<Operation *> hoist;
  for (Value operand : dest->getOperands()) {
    Operation *def = operand.getDefiningOp();
    if (!def || def->getBlock() != block || def->isBeforeInBlock(earliest))
      continue;
    if (!isa<memref::AllocOp, AllocOutputOp>(def))
      return false;
    for (Value v : def->getOperands()) {
      Operation *inner = v.getDefiningOp();
      if (inner && inner->getBlock() == block &&
          !inner->isBeforeInBlock(earliest))
        return false;
    }
    hoist.push_back(def);
  }
  for (Operation *op : hoist)
    op->moveBefore(earliest);
  if (!dest->isBeforeInBlock(earliest))
    dest->moveBefore(earliest);

  for (memref::SubViewOp sub : subs) {
    rewriter.setInsertionPoint(sub);
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
    rewriter.eraseOp(sub);
  }

  eraseDeallocs(alloc.getResult());
  copy.erase();
  alloc.erase();
  return true;
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
  // The slot copies are the only writes, so the buffer holds exactly what
  // they copied.
  SmallVector<Write> smallWrites;
  if (!collectWrites(small, smallWrites) ||
      smallWrites.size() != smallSlots.size())
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
    DenseSet<Operation *> matched;
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
      matched.insert(slot.copy);
    }
    if (!matches || duplicates.size() != smallSlots.size())
      continue;
    // Nothing but those repeated copies may write the prefix.
    SmallVector<Write> largeWrites;
    if (!collectWrites(large, largeWrites) ||
        llvm::any_of(largeWrites, [&](const Write &write) {
          return write.span.first < smallElems && !matched.contains(write.op);
        }))
      continue;
    bool dominates = llvm::all_of(largeOp->getOperands(), [&](Value v) {
      Operation *def = v.getDefiningOp();
      return !def || (def->getBlock() == block && def->isBeforeInBlock(alloc));
    });
    if (!dominates)
      continue;

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
    // Readers of the old buffer can sit after the large buffer's dealloc.
    Operation *latest = lastUser(view.getResult(), block, nullptr);
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

  int64_t slotElems = srcType.getNumElements();
  int64_t baseElems = baseType.getNumElements();
  if (elementOffset < 0 || slotElems > baseElems - elementOffset)
    return false;
  // Any other write into the slot would land on the aliased source.
  Span slotSpan{elementOffset, elementOffset + slotElems};
  SmallVector<Write> baseWrites;
  if (!collectWrites(base, baseWrites) ||
      llvm::any_of(baseWrites, [&](const Write &write) {
        return write.op != copy && overlaps(write.span, slotSpan);
      }))
    return false;

  // Every use of the private buffer, through any view, stays in this block
  // and is not returned. A write after the copy would now land in the concat
  // result.
  Block *block = alloc->getBlock();
  bool sawWriter = false;
  Operation *earliest = nullptr;
  Operation *latest = nullptr;
  SmallVector<OpOperand *> uses;
  collectAliasUses(alloc.getResult(), uses);
  for (OpOperand *use : uses) {
    Operation *owner = use->getOwner();
    if (owner == copy || isa<memref::DeallocOp>(owner))
      continue;
    if (isa<func::ReturnOp>(owner) || owner->getBlock() != block)
      return false;
    if (mayWrite(*use)) {
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
