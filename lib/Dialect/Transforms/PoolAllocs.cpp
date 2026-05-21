/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PoolAllocs.cpp - Pack memref.alloc into a single i8 pool -----------===//
//
// Packs all memref.alloc ops in a single-block function into one contiguous
// byte pool (memref<?xi8>), replacing each original alloc with a memref.view
// at a computed byte offset. The pool is acquired via hip.get_pool(%ctx,
// %pool_size) so the runtime can grow on demand.
//
// Algorithm overview:
//   Phase 1 - Liveness:  assign [defIndex, lastUseIndex] to each alloc,
//             following view-like ops transitively via BufferViewFlowAnalysis.
//   Phase 2 - Partition:  split allocs into static (compile-time byte size)
//             and dynamic (runtime byte size) groups.
//   Phase 3 - Static packing:  greedy best-fit offset assignment inspired by
//             greedy best-fit strip packing.  Allocs whose
//             lifetimes don't overlap may share the same offset range.
//   Phase 4 - Dynamic packing:  group by structural byte-size key (same
//             static factor + same dynamic SSA operands).  Within each group,
//             bin non-overlapping lifetimes so they share an offset at runtime.
//   Phase 5 - IR emission:  create hip.get_pool, compute offsets via
//             arith ops, and replace each original alloc with memref.view.
//
// All sub-buffer offsets are aligned to the configured alignment (default 256
// bytes) to satisfy GPU coalesced-access requirements.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/BufferUtils.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Transforms/BufferViewFlowOpInterfaceImpl.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/MathExtras.h"

#include <functional>

#define DEBUG_TYPE "hip-pool-allocs"

STATISTIC(NumAllocsPooled, "Number of allocations pooled into byte buffer");
STATISTIC(NumStaticPacked, "Number of static allocations packed");
STATISTIC(NumDynBuckets, "Number of dynamic size buckets created");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_POOLALLOCSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

//===----------------------------------------------------------------------===//
// AllocInfo - per-alloc metadata collected in Phase 1
//===----------------------------------------------------------------------===//

struct AllocInfo {
  memref::AllocOp allocOp;
  unsigned defIndex;      ///< sequential index of the alloc op in the block
  unsigned lastUseIndex;  ///< highest index of any (transitive) user
  int64_t staticByteSize; ///< >0 for static shapes, 0 for dynamic
};

/// True when two allocs' [def, lastUse] intervals overlap.
static bool lifetimesOverlap(const AllocInfo &a, const AllocInfo &b) {
  return !(a.lastUseIndex < b.defIndex || b.lastUseIndex < a.defIndex);
}

//===----------------------------------------------------------------------===//
// Phase 3 - Static packing: greedy best-fit gap finding
//===----------------------------------------------------------------------===//
//
// Assigns a byte offset in a 1D address space to each static alloc.
// Two allocs whose lifetimes don't overlap may share the same address range.
//
// Example (static attention model, 4 allocs of 32768 bytes each):
//
//   Alloc  Lifetime      Offset assigned   Why
//   -----  ------------  ----------------  -------------------------------
//   A      [3, 16]       0                 first alloc, placed at start
//   B      [5, 13]       32768             overlaps A -> append after A
//   C      [7, 14]       65536             overlaps A,B -> append after B
//   D      [9, 11]       98304             overlaps A,B,C -> append after C
//
//   Pool: |----A----|----B----|----C----|----D----|  = 131072 bytes
//
// If alloc X's lifetime ends before alloc Y starts, Y can reuse X's
// address range.  The gap-finding loop looks for such opportunities.

/// A placed allocation in the linear address space.
struct Reservation {
  AllocInfo *info;
  int64_t offset;
  int64_t size; ///< aligned byte size
};

/// Assign byte offsets to static allocs using a greedy best-fit strategy.
///
/// For each alloc (processed largest-first), scan the existing reservations
/// whose lifetimes overlap, find the smallest gap that fits, and place it
/// there.  If no gap fits, append after the last overlapping reservation.
static SmallVector<std::pair<AllocInfo *, int64_t>>
packStaticAllocs(MutableArrayRef<AllocInfo> statics, int64_t alignment) {
  SmallVector<std::pair<AllocInfo *, int64_t>> assignments;
  SmallVector<Reservation, 16> reservations; // kept sorted by offset

  for (AllocInfo &info : statics) {
    int64_t size = llvm::alignTo(info.staticByteSize, alignment);
    int64_t bestOffset = -1;
    int64_t bestFit = INT64_MAX;

    // Walk existing reservations with overlapping lifetimes and look for
    // the smallest gap between them that fits this allocation.
    int64_t currentOffset = 0;
    for (auto &res : reservations) {
      if (!lifetimesOverlap(info, *res.info))
        continue;
      int64_t alignedOffset = llvm::alignTo(currentOffset, alignment);
      if (alignedOffset + size <= res.offset &&
          res.offset - alignedOffset < bestFit) {
        bestOffset = alignedOffset;
        bestFit = res.offset - alignedOffset;
      }
      currentOffset = std::max(currentOffset, res.offset + res.size);
    }

    // No gap found - append after all overlapping reservations.
    if (bestOffset < 0)
      bestOffset = llvm::alignTo(currentOffset, alignment);

    // Insert into the sorted reservation list.
    Reservation newRes{&info, bestOffset, size};
    auto insertIt = reservations.begin();
    while (insertIt != reservations.end() && insertIt->offset < newRes.offset)
      ++insertIt;
    reservations.insert(insertIt, newRes);
    assignments.emplace_back(&info, bestOffset);
  }

  return assignments;
}

//===----------------------------------------------------------------------===//
// Phase 4 - Dynamic packing: bucket by structural byte-size, bin by lifetime
//===----------------------------------------------------------------------===//
//
// For allocs with runtime-unknown sizes we cannot hardcode byte offsets.
// Instead we group allocs that provably have the same runtime byte size into
// "buckets", then within each bucket we bin allocs with non-overlapping
// lifetimes so they can share one offset slot.
//
// Two-level grouping:
//
//   Level 1 - Bucket by DynSizeKey:
//     A DynSizeKey = {staticFactor, [dynOperands...]}.
//     staticFactor = elementBytes * product_of_static_dims.
//     dynOperands  = SSA values for the dynamic dimensions.
//     Two allocs with the same key have the same runtime byte size:
//       byte_size = staticFactor * dynDim0 * dynDim1 * ...
//
//   Level 2 - Bin by lifetime:
//     Within a bucket, first-fit pack allocs whose lifetimes don't overlap
//     into bins.  All allocs in the same bin share one offset at runtime.

/// A bucket groups dynamic allocs that share the same runtime byte size.
/// Within a bucket, each "bin" holds allocs with non-overlapping lifetimes
/// that can share a single offset at runtime.
struct DynBucket {
  Value byteSizeValue; ///< SSA value for the unaligned byte size
  Value alignedSize;   ///< SSA value for the aligned byte size (cached)
  SmallVector<SmallVector<AllocInfo *>>
      bins; ///< bins of non-overlapping allocs
};

/// Structural key for grouping allocs with identical runtime byte sizes.
struct DynSizeKey {
  int64_t staticFactor;
  SmallVector<Value, 2> dynOperands;

  bool operator==(const DynSizeKey &other) const {
    return staticFactor == other.staticFactor &&
           dynOperands == other.dynOperands;
  }
};

/// Group dynamic allocs into buckets and bins.
static SmallVector<DynBucket>
packDynamicAllocs(MutableArrayRef<AllocInfo> dynamics, OpBuilder &builder,
                  Location loc, int64_t alignment) {
  struct KeyedInfo {
    DynSizeKey key;
    AllocInfo *info;
  };

  // Step 1: build structural keys.
  SmallVector<KeyedInfo> keyed;
  for (AllocInfo &info : dynamics) {
    MemRefType type = info.allocOp.getType();
    auto dynSizes = info.allocOp.getDynamicSizes();
    int64_t totalBits = type.getElementTypeBitWidth();
    int64_t staticFactor = 1;
    for (int64_t dim : type.getShape())
      if (!ShapedType::isDynamic(dim))
        staticFactor *= dim;
    staticFactor =
        static_cast<int64_t>(llvm::divideCeil(staticFactor * totalBits, 8));
    DynSizeKey key;
    key.staticFactor = staticFactor;
    unsigned dynIdx = 0;
    for (int64_t dim : type.getShape())
      if (ShapedType::isDynamic(dim))
        key.dynOperands.push_back(dynSizes[dynIdx++]);
    keyed.push_back({key, &info});
  }

  // Step 2: one byte-size SSA value per unique key.
  SmallVector<std::pair<DynSizeKey, Value>> uniqueKeys;
  auto findOrCreateByteSize = [&](const DynSizeKey &key) -> Value {
    for (auto &[k, v] : uniqueKeys)
      if (k == key)
        return v;
    Value byteSize =
        arith::ConstantIndexOp::create(builder, loc, key.staticFactor);
    for (Value dynDim : key.dynOperands)
      byteSize = builder.createOrFold<arith::MulIOp>(loc, byteSize, dynDim);
    uniqueKeys.push_back({key, byteSize});
    return byteSize;
  };

  // Step 3: group by byte-size SSA value.  Track the staticFactor alongside
  // each group so step 4 can skip alignment when the factor is already a
  // multiple of the alignment (e.g. staticFactor=256 with alignment=256).
  struct SizeGroup {
    int64_t staticFactor;
    SmallVector<AllocInfo *> infos;
  };
  llvm::MapVector<Value, SizeGroup> bySize;
  for (auto &[key, info] : keyed) {
    Value sizeVal = findOrCreateByteSize(key);
    auto &group = bySize[sizeVal];
    group.staticFactor = key.staticFactor;
    group.infos.push_back(info);
  }

  // Step 4: first-fit bin packing within each group.
  SmallVector<DynBucket> buckets;
  for (auto &[sizeVal, group] : bySize) {
    auto &infos = group.infos;
    DynBucket bucket;
    bucket.byteSizeValue = sizeVal;
    // When staticFactor is a multiple of alignment, the byte size
    // (staticFactor * dynDim0 * dynDim1 * ...) is guaranteed to be aligned
    // regardless of the dynamic dimensions, so emitAlignUp is a no-op.
    // Skipping it avoids an expensive divui + supporting arithmetic.
    if (group.staticFactor % alignment == 0)
      bucket.alignedSize = sizeVal;
    else
      bucket.alignedSize = emitAlignUp(builder, loc, sizeVal, alignment);
    for (AllocInfo *info : infos) {
      bool placed = false;
      for (auto &bin : bucket.bins) {
        bool conflicts = false;
        for (AllocInfo *existing : bin) {
          if (lifetimesOverlap(*info, *existing)) {
            conflicts = true;
            break;
          }
        }
        if (!conflicts) {
          bin.push_back(info);
          placed = true;
          break;
        }
      }
      if (!placed)
        bucket.bins.push_back({info});
    }
    buckets.push_back(std::move(bucket));
  }

  return buckets;
}

//===----------------------------------------------------------------------===//
// Pass entry point
//===----------------------------------------------------------------------===//

struct PoolAllocsPass : public impl::PoolAllocsPassBase<PoolAllocsPass> {
  using PoolAllocsPassBase::PoolAllocsPassBase;

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, memref::MemRefDialect>();
    arith::registerBufferViewFlowOpInterfaceExternalModels(registry);
  }

  void runOnOperation() override;
};

/// Recursively resolve `dim(src, i)` through any chain of
/// `memref.collapse_shape` / `memref.expand_shape`, emitting pure arithmetic
/// (`arith.muli` for collapse, `arith.divui` for the dynamic dim of an expand
/// group) over `memref.dim` of the chain's root memref. Both intermediate ops
/// are pure descriptor edits, so this rewrite preserves semantics; it pushes
/// every dim query down to the chain root, where the source is hoistable
/// (function arg, alloc, or memref.view).
///
/// IR example (collapse case — dim of a collapse_shape becomes a product
/// of dims of the source memref):
///
///   Before:
///     %c = memref.collapse_shape %src [[0], [1, 2]]
///             : memref<?x?x4xf16> into memref<?x?xf16>
///     %d = memref.dim %c, %c1 : memref<?x?xf16>     // i = 1
///
///   After (one step of the recursion, with K = 4 absorbed from the source):
///     %d_src = memref.dim %src, %c1 : memref<?x?x4xf16>
///     %k     = arith.constant 4 : index
///     %d     = arith.muli %d_src, %k : index
///
/// IR example (expand case — dynamic output dim becomes div of the input
/// dim by the static partner factor):
///
///   Before:
///     %e = memref.expand_shape %src [[0, 1]] output_shape [%n, %k]
///             : memref<?xf16> into memref<?x4xf16>     // K = 4
///     %d = memref.dim %e, %c0 : memref<?x4xf16>        // i = 0 (dynamic)
///
///   After:
///     %d_src = memref.dim %src, %c0 : memref<?xf16>
///     %k     = arith.constant 4 : index
///     %d     = arith.divui %d_src, %k : index
///
/// Bounded recursion: each step strips one reshape layer, so depth equals the
/// reshape-chain depth. In practice ≤ 2 (typical case: a same-rank dynamic
/// Reshape pair around a per-head norm op).
static Value resolveDimAtSource(OpBuilder &b, Location loc, Value src,
                                int64_t i) {
  Operation *def = src.getDefiningOp();
  // memref.dim(memref.alloc(d0, d1, ...), i) -> i-th dyn operand of alloc.
  // The dyn operand is the actual SSA value the alloc was sized by — its
  // defining op (typically `memref.dim` of a function arg or another alloc,
  // possibly via arith) is hoistable through the rest of the chain. Crucially
  // we DO NOT emit a fresh `memref.dim %alloc, i` here, because after Phase 5
  // the alloc gets replaced by a `memref.view` placed at the pool prelude;
  // hoisted dim ops would then reference a view that lives later in the
  // block (dominance error). Returning the dyn operand SSA value directly
  // sidesteps the entire problem.
  if (auto alloc = dyn_cast_or_null<memref::AllocOp>(def)) {
    auto srcType = cast<MemRefType>(src.getType());
    if (i < 0 || i >= srcType.getRank())
      return Value();
    if (!srcType.isDynamicDim(i))
      return arith::ConstantIndexOp::create(b, loc, srcType.getDimSize(i));
    int64_t dynIdx = 0;
    for (int64_t j = 0; j < i; ++j)
      if (srcType.isDynamicDim(j))
        ++dynIdx;
    auto dynSizes = alloc.getDynamicSizes();
    if (dynIdx < static_cast<int64_t>(dynSizes.size()))
      return dynSizes[dynIdx];
  }
  if (auto collapse = dyn_cast_or_null<memref::CollapseShapeOp>(def)) {
    auto reassoc = collapse.getReassociationIndices();
    if (i >= 0 && i < static_cast<int64_t>(reassoc.size())) {
      Value collSrc = collapse.getSrc();
      Value acc;
      for (int64_t srcDim : reassoc[i]) {
        Value d = resolveDimAtSource(b, loc, collSrc, srcDim);
        acc = acc ? b.createOrFold<arith::MulIOp>(loc, acc, d) : d;
      }
      if (acc)
        return acc;
    }
  } else if (auto expand = dyn_cast_or_null<memref::ExpandShapeOp>(def)) {
    auto resultType = cast<MemRefType>(expand.getResult().getType());
    if (i >= 0 && i < resultType.getRank() && !resultType.isDynamicDim(i))
      return arith::ConstantIndexOp::create(b, loc, resultType.getDimSize(i));
    auto reassoc = expand.getReassociationIndices();
    int64_t srcIdx = -1;
    for (int64_t j = 0; j < static_cast<int64_t>(reassoc.size()) && srcIdx < 0;
         ++j)
      if (llvm::is_contained(reassoc[j], i))
        srcIdx = j;
    if (srcIdx >= 0) {
      const auto &group = reassoc[srcIdx];
      int64_t staticProduct = 1;
      bool ok = true;
      for (int64_t outIdx : group) {
        if (outIdx == i)
          continue;
        if (resultType.isDynamicDim(outIdx)) {
          ok = false;
          break;
        }
        staticProduct *= resultType.getDimSize(outIdx);
      }
      if (ok) {
        Value srcDim = resolveDimAtSource(b, loc, expand.getSrc(), srcIdx);
        if (staticProduct == 1)
          return srcDim;
        Value div = arith::ConstantIndexOp::create(b, loc, staticProduct);
        return b.createOrFold<arith::DivUIOp>(loc, srcDim, div);
      }
    }
  }
  // Base case: emit memref.dim on src directly. The source is the chain root
  // (alloc / function arg / view / non-reshape op) — `memref.dim` on it is
  // hoistable.
  auto srcType = cast<MemRefType>(src.getType());
  if (!srcType.isDynamicDim(i))
    return arith::ConstantIndexOp::create(b, loc, srcType.getDimSize(i));
  Value idxVal = arith::ConstantIndexOp::create(b, loc, i);
  return memref::DimOp::create(b, loc, src, idxVal);
}

/// Fold `memref.dim(memref.collapse_shape(src), i)` and
/// `memref.dim(memref.expand_shape(src), i)` into pure arithmetic on
/// `memref.dim(root, ...)`. Required for PoolAllocs's hoisting to succeed
/// when dim queries originate from reshape chains produced by the same-rank
/// dynamic Reshape decomposition (expand_shape + collapse_shape).
///
/// MLIR's stock canonicalizer leaves these alone for dynamic dims; without
/// this fold the surviving `memref.dim %some_reshape` would be hoisted (it's
/// in `isHoistable`) but its operand `%some_reshape` is not hoistable —
/// yielding an SSA dominance error in Phase 4.
///
/// The replacement uses `arith.muli` (collapse) and `arith.divui` (expand
/// with absorbed static factor); both are in `isHoistable`.
static void foldDimOfReshape(func::FuncOp funcOp) {
  SmallVector<memref::DimOp> worklist;
  funcOp.walk([&](memref::DimOp op) {
    Operation *def = op.getSource().getDefiningOp();
    if (def &&
        (isa<memref::AllocOp>(def) || isa<memref::CollapseShapeOp>(def) ||
         isa<memref::ExpandShapeOp>(def)))
      worklist.push_back(op);
  });

  for (memref::DimOp dimOp : worklist) {
    auto idxAttr = getConstantIntValue(dimOp.getIndex());
    if (!idxAttr)
      continue; // dynamic index — can't fold structurally
    OpBuilder b(dimOp);
    Value replacement =
        resolveDimAtSource(b, dimOp.getLoc(), dimOp.getSource(), *idxAttr);
    if (replacement && replacement != dimOp.getResult()) {
      dimOp.replaceAllUsesWith(replacement);
      dimOp.erase();
    }
  }
}

void PoolAllocsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (alignment <= 0 || (alignment & (alignment - 1)) != 0) {
    funcOp.emitError("hip-pool-allocs: alignment must be a positive power of 2"
                     " (got ")
        << alignment << ")";
    return signalPassFailure();
  }

  if (funcOp.empty())
    return;

  // Pre-pass: simplify `memref.dim` of `memref.collapse_shape`/
  // `memref.expand_shape` so the hoist worklist below can ascend through to
  // the original source memref. Without this, a surviving dim-of-collapse
  // breaks Phase 4's SSA dominance for any same-rank dynamic Reshape
  // decomposed into expand_shape + collapse_shape.
  foldDimOfReshape(funcOp);
  // TODO: Generalize to multi-block functions using MLIR's Liveness analysis
  // instead of sequential op indices.
  if (!funcOp.getBody().hasOneBlock()) {
    funcOp.emitError("hip-pool-allocs requires single-block functions; "
                     "liveness analysis uses sequential op indices that do "
                     "not generalize to control flow");
    return signalPassFailure();
  }

  Block &block = funcOp.getBody().front();

  // ----- Phase 1: liveness analysis ------------------------------------

  BufferViewFlowAnalysis aliasAnalysis(funcOp);

  DenseMap<Operation *, unsigned> opIndex;
  unsigned idx = 0;
  for (Operation &op : block)
    opIndex[&op] = idx++;
  unsigned blockSize = idx;

  SmallVector<AllocInfo> allInfos;
  for (Operation &op : block) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    if (!allocOp)
      continue;
    Value result = allocOp.getResult();
    if (result.use_empty())
      continue;
    AllocInfo info;
    info.allocOp = allocOp;
    info.defIndex = opIndex[&op];
    info.lastUseIndex = findLastAliasedUseIndex(result, aliasAnalysis, block,
                                                opIndex, blockSize);
    info.staticByteSize = getStaticByteSize(allocOp.getType());
    LLVM_DEBUG(llvm::dbgs() << "  alloc " << allocOp << " [" << info.defIndex
                            << ", " << info.lastUseIndex << "] "
                            << info.staticByteSize << " bytes\n");
    allInfos.push_back(info);
  }

  if (allInfos.size() < 2) {
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    OpBuilder zeroBuilder(funcOp.getContext());
    moduleOp->setAttr("hipdnn.pool_size", zeroBuilder.getI64IntegerAttr(0));
    moduleOp->setAttr("hipdnn.buffer_count", zeroBuilder.getI64IntegerAttr(0));
    moduleOp->setAttr("hipdnn.buffer_offsets",
                      zeroBuilder.getArrayAttr(SmallVector<Attribute>{}));
    return;
  }

  // ----- Phase 2: partition into static / dynamic ----------------------

  SmallVector<AllocInfo> statics, dynamics;
  for (auto &info : allInfos) {
    if (info.staticByteSize > 0)
      statics.push_back(info);
    else
      dynamics.push_back(info);
  }

  // Largest-first ordering improves packing density.
  llvm::sort(statics, [](const AllocInfo &a, const AllocInfo &b) {
    return a.staticByteSize > b.staticByteSize;
  });

  // ----- Phase 3: pack statics ----------------------------------------

  int64_t align = alignment;
  auto staticAssignments = packStaticAllocs(statics, align);
  NumStaticPacked += staticAssignments.size();

  int64_t staticPoolSize = 0;
  for (auto &[info, offset] : staticAssignments) {
    int64_t end = offset + llvm::alignTo(info->staticByteSize, align);
    staticPoolSize = std::max(staticPoolSize, end);
  }

  // ----- Phase 4: pack dynamics ----------------------------------------

  Location loc = funcOp.getLoc();
  OpBuilder builder(funcOp.getContext());

  // Pool setup must be inserted before the first alloc. Dynamic allocs use
  // SSA values for their sizes (e.g. memref.dim extracting a dimension from
  // a function argument). These ops may appear between allocs in the block.
  // Move them before the first alloc so they dominate the pool size
  // computation. Safe because they only depend on function arguments.
  Operation *firstPooledAlloc = allInfos.front().allocOp.getOperation();
  for (auto &info : allInfos)
    if (info.allocOp->isBeforeInBlock(firstPooledAlloc))
      firstPooledAlloc = info.allocOp.getOperation();

  // Recursively hoist the def-chain that computes each dynamic size in
  // front of the first pooled alloc.  A single-level hoist is not enough:
  // dynamic sizes are typically computed as `arith.muli %dim, %const` where
  // %dim itself comes from `memref.dim %arg, %i` and %const from
  // `arith.constant`.  Moving only the muli but leaving its operands behind
  // breaks SSA dominance for the pool-size arithmetic emitted in Phase 5.
  //
  // Whitelist intentionally narrow: only side-effect-free pure ops whose
  // operands are themselves trivial (block args, constants, or other
  // whitelisted ops).  This guarantees the move never reorders observable
  // behavior — the ops are restricted to shape/index arithmetic, never any
  // op that might read or mutate buffers.
  auto isHoistable = [](Operation *op) {
    return isa<memref::DimOp, arith::ConstantOp, arith::ConstantIndexOp,
               arith::MulIOp, arith::AddIOp, arith::SubIOp, arith::IndexCastOp,
               // Division: needed when a Reshape is decomposed into
               // expand_shape(input, [..., dyn/K, K, ...]) + collapse_shape,
               // which emits `arith.divui %dim, K` for the dynamic absorber.
               // Same purity guarantee as MulIOp/AddIOp — pure index/integer
               // arithmetic, no side effects, safe to hoist.
               arith::DivUIOp, arith::DivSIOp>(op);
  };
  llvm::SetVector<Operation *> hoistWorklist;
  for (auto &info : dynamics)
    for (Value dynDim : info.allocOp.getDynamicSizes())
      if (auto *defOp = dynDim.getDefiningOp())
        hoistWorklist.insert(defOp);

  // Worklist grows as we walk operands of ops we visit; SetVector preserves
  // the discovery order while preventing duplicate visits.  We process
  // index-by-index instead of pop_back so additions during the loop are
  // visited before we exit.
  for (size_t idx = 0; idx < hoistWorklist.size(); ++idx) {
    Operation *op = hoistWorklist[idx];
    if (!isHoistable(op))
      continue; // Not hoistable; leave it alone — Phase 5's createOrFold
                // will see the original SSA value at its current position.
                // If that position post-dominates firstPooledAlloc the
                // verifier will catch the SSA error so we fail loudly.
    for (Value operand : op->getOperands())
      if (auto *defOp = operand.getDefiningOp())
        hoistWorklist.insert(defOp);
  }

  // Move ops so their final layout has each operand defined before its uses.
  // Each `moveBefore(firstPooledAlloc)` places the op immediately before
  // firstPooledAlloc, displacing any previously-moved ops up by one slot. So
  // the final top-to-bottom order in the prelude is the REVERSE of the move
  // order. To make operands dominate uses, we therefore want move order:
  //   uses (consumers) first, operands (producers) last.
  //
  // The previous implementation used `hoistWorklist` directly via reverse
  // iteration, which works for linear chains (BFS produces consumer-first,
  // reverse gives operand-first move order, final layout = operand-then-
  // consumer). It breaks for shared producers reached through multiple
  // consumers: BFS may add the consumer before the producer, but the
  // producer can also be added directly by a different alloc's worklist
  // entry, ending up later in `hoistWorklist`. Reverse iteration then moves
  // the producer EARLIER than the consumer that uses it, leaving the final
  // layout with the consumer above the producer — SSA dominance error.
  //
  // Fix: depth-first post-order topological sort over the hoistable subset,
  // then iterate the sort in reverse for the move order. DFS post-order
  // guarantees operands precede uses; reversing gives the consumer-first
  // move sequence we need.
  llvm::DenseSet<Operation *> hoistSet;
  for (Operation *op : hoistWorklist)
    if (isHoistable(op))
      hoistSet.insert(op);

  SmallVector<Operation *> sorted;
  llvm::DenseSet<Operation *> visited;
  std::function<void(Operation *)> visit = [&](Operation *op) {
    if (!op || !hoistSet.count(op))
      return;
    if (!visited.insert(op).second)
      return;
    for (Value operand : op->getOperands())
      visit(operand.getDefiningOp());
    sorted.push_back(op);
  };
  for (Operation *op : hoistWorklist)
    visit(op);

  // `sorted` is DFS post-order: operands precede uses. Iterate FORWARD so
  // operands are moved first; each subsequent `moveBefore(firstPooledAlloc)`
  // places the next op at -1 and displaces all earlier moves up by one.
  // Final layout: first-moved (operands) at the top, last-moved (uses) at
  // the bottom of the prelude — exactly the dominance order we need.
  for (auto it = sorted.begin(); it != sorted.end(); ++it) {
    Operation *defOp = *it;
    if (defOp->getBlock() != firstPooledAlloc->getBlock())
      continue; // Defined in a different region/block — already dominates.
    if (defOp == firstPooledAlloc)
      continue;
    if (defOp->isBeforeInBlock(firstPooledAlloc))
      continue; // Already in the right place.
    defOp->moveBefore(firstPooledAlloc);
  }

  builder.setInsertionPoint(firstPooledAlloc);

  auto dynBuckets = packDynamicAllocs(dynamics, builder, loc, align);
  NumDynBuckets += dynBuckets.size();

  // ----- Phase 5: emit pool + views -----------------------------------

  bool hasDynamic = !dynBuckets.empty();

  // 5a. Compute total pool size as an SSA value.
  // Uses createOrFold so that trivial ops (addi(x,0), muli(x,1)) are
  // folded away automatically by the arith dialect's fold methods.
  if (!hasDynamic && staticPoolSize == 0)
    return;

  Value poolSize;
  if (hasDynamic) {
    poolSize = arith::ConstantIndexOp::create(builder, loc, staticPoolSize);

    for (auto &bucket : dynBuckets) {
      Value numBinsVal = arith::ConstantIndexOp::create(
          builder, loc, static_cast<int64_t>(bucket.bins.size()));
      Value contribution = builder.createOrFold<arith::MulIOp>(
          loc, bucket.alignedSize, numBinsVal);
      poolSize =
          builder.createOrFold<arith::AddIOp>(loc, poolSize, contribution);
    }
  } else {
    poolSize = arith::ConstantIndexOp::create(builder, loc, staticPoolSize);
  }

  LLVM_DEBUG(llvm::dbgs() << "Pool: static=" << staticPoolSize << " bytes, "
                          << dynBuckets.size() << " dynamic buckets, "
                          << allInfos.size() << " total allocs\n");

  // 5b. Acquire the pool via hip.get_pool(%ctx, %pool_size).
  if (funcOp.getNumArguments() == 0 ||
      !isa<hip::ContextType>(funcOp.getArgument(0).getType())) {
    funcOp.emitError("function missing !hip.context as arg 0 — "
                     "run hip-add-context-arg before hip-pool-allocs");
    return signalPassFailure();
  }
  Value ctx = funcOp.getArgument(0);
  auto poolType =
      MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(8));
  Value pool = hip::GetPoolOp::create(builder, loc, poolType, ctx, poolSize);

  // 5c. Compute byte offsets for every alloc and store in allocToOffset.
  DenseMap<Operation *, Value> allocToOffset;

  for (auto &[info, offset] : staticAssignments) {
    Value offsetVal = arith::ConstantIndexOp::create(builder, loc, offset);
    allocToOffset[info->allocOp.getOperation()] = offsetVal;
  }

  if (hasDynamic) {
    Value currentBase =
        arith::ConstantIndexOp::create(builder, loc, staticPoolSize);

    for (auto &bucket : dynBuckets) {
      for (auto [binIdx, bin] : llvm::enumerate(bucket.bins)) {
        // offset = currentBase + binIdx * alignedBucketSize
        Value binIdxVal = arith::ConstantIndexOp::create(
            builder, loc, static_cast<int64_t>(binIdx));
        Value binContrib = builder.createOrFold<arith::MulIOp>(
            loc, bucket.alignedSize, binIdxVal);
        Value binOffset =
            builder.createOrFold<arith::AddIOp>(loc, currentBase, binContrib);
        for (AllocInfo *info : bin)
          allocToOffset[info->allocOp.getOperation()] = binOffset;
      }

      // Advance base past this entire bucket.
      Value numBinsVal = arith::ConstantIndexOp::create(
          builder, loc, static_cast<int64_t>(bucket.bins.size()));
      Value bucketTotal = builder.createOrFold<arith::MulIOp>(
          loc, bucket.alignedSize, numBinsVal);
      currentBase =
          builder.createOrFold<arith::AddIOp>(loc, currentBase, bucketTotal);
    }
  }

  // 5d. Replace each original alloc with a memref.view into the pool.
  SmallVector<int64_t> staticOffsets;
  for (auto &info : allInfos) {
    auto it = allocToOffset.find(info.allocOp.getOperation());
    if (it == allocToOffset.end())
      continue;

    Value offset = it->second;
    builder.setInsertionPoint(info.allocOp);

    MemRefType viewType = info.allocOp.getType();
    auto dynSizes = info.allocOp.getDynamicSizes();
    auto view =
        memref::ViewOp::create(builder, info.allocOp.getLoc(), viewType, pool,
                               offset, SmallVector<Value>(dynSizes));

    info.allocOp.replaceAllUsesWith(view.getResult());
    info.allocOp.erase();
    ++NumAllocsPooled;

    if (auto constOp = offset.getDefiningOp<arith::ConstantIndexOp>())
      staticOffsets.push_back(constOp.value());
    else
      staticOffsets.push_back(-1);
  }

  // 5d'. Erase deallocs that now target views — the pool is owned by the
  // runtime state (not this function), so no pool dealloc is inserted.
  SmallVector<memref::DeallocOp> orphanedDeallocs;
  funcOp.walk([&](memref::DeallocOp op) {
    if (op.getMemref().getDefiningOp<memref::ViewOp>())
      orphanedDeallocs.push_back(op);
  });
  for (auto op : orphanedDeallocs)
    op.erase();

  // 5e. Attach pool metadata to the module for GenerateInterface.
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  MLIRContext *mlirCtx = funcOp.getContext();
  moduleOp->setAttr("hipdnn.pool_size",
                    builder.getI64IntegerAttr(staticPoolSize));
  moduleOp->setAttr("hipdnn.buffer_count",
                    builder.getI64IntegerAttr((int64_t)allInfos.size()));
  SmallVector<Attribute> offsetAttrs;
  for (int64_t off : staticOffsets)
    offsetAttrs.push_back(builder.getI64IntegerAttr(off));
  moduleOp->setAttr("hipdnn.buffer_offsets",
                    ArrayAttr::get(mlirCtx, offsetAttrs));
}

} // namespace
} // namespace hip
} // namespace mlir
