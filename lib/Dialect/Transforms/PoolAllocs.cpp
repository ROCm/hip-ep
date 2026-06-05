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
//   Phase 3 - Static packing:  greedy best-fit offset assignment.  Allocs
//             whose lifetimes don't overlap may share the same offset range.
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
// findLatestLegalInsertionPoint - dominator-aware insertion in a single block
//===----------------------------------------------------------------------===//
//
// PoolAllocs::runOnOperation enforces single-block functions, so SSA
// dominance inside the function body collapses to "comes earlier in the
// block". The helper picks the latest position in the block consistent
// with two ordering constraints:
//
//   - every op in `requiredAfter`  must come strictly before the result;
//   - every op in `requiredBefore` must come strictly after the result.
//
// "Latest" (rather than "earliest") matters for Phase 5: emitting the
// per-bucket size arithmetic just below its dyn-operand defs (instead of
// hoisting it to block start) keeps the size SSA close to the IR that
// produced it and avoids reordering unrelated ops.

/// Latest position in `block` such that:
///   - every op in `requiredAfter` comes strictly before the result, AND
///   - every op in `requiredBefore` comes strictly after the result.
///
/// All input ops must already live in `block`; the caller is responsible
/// for filtering out cross-block / cross-region defs (those already
/// dominate any insertion point inside `block` and need not be considered).
///
/// Returns `std::nullopt` when the constraints are infeasible — i.e. some
/// op in `requiredAfter` is at-or-after some op in `requiredBefore`. That
/// situation cannot arise for the bucket cases we care about (the
/// alloc's dyn-operand defs always dominate the alloc itself), but a
/// future caller might pass an inconsistent set; signaling infeasibility
/// is cleaner than asserting.
///
/// Returns `Block::end()` when both lists are empty (a fresh empty block,
/// or a caller that passed nothing) — a degenerate but well-defined point.
///
/// Example:
///
/// ```mlir
/// // requiredAfter: [%size_def]   requiredBefore: [%alloc]
/// %size_def = arith.muli %dim, %c4 : index
/// %something_else = ...                  // ← returned iterator points here
/// %alloc = memref.alloc(%size_def) : memref<?xf16>
/// ```
///
/// The returned iterator is `std::next(%size_def's iterator)`, i.e. the
/// position immediately after the latest predecessor. Callers feed this
/// to `OpBuilder::setInsertionPoint(block, *iter)` (or, when the
/// requiredBefore list is non-empty, simply `setInsertionPoint(*iter)` is
/// equivalent because the iterator points at the operation we want to
/// insert before).
static std::optional<Block::iterator>
findLatestLegalInsertionPoint(Block &block, ArrayRef<Operation *> requiredAfter,
                              ArrayRef<Operation *> requiredBefore) {
  // `lo` = the latest op in `requiredAfter`. If non-empty, the result
  // must be strictly after `lo` (i.e. iterator = std::next(lo)).
  Operation *lo = nullptr;
  for (Operation *op : requiredAfter) {
    assert(op && op->getBlock() == &block &&
           "findLatestLegalInsertionPoint: requiredAfter op outside block");
    if (!lo || lo->isBeforeInBlock(op))
      lo = op;
  }

  // `hi` = the earliest op in `requiredBefore`. If non-empty, the result
  // must be strictly before `hi`.
  Operation *hi = nullptr;
  for (Operation *op : requiredBefore) {
    assert(op && op->getBlock() == &block &&
           "findLatestLegalInsertionPoint: requiredBefore op outside block");
    if (!hi || op->isBeforeInBlock(hi))
      hi = op;
  }

  // Both empty: degenerate, hand back end() so OpBuilder can no-op.
  if (!lo && !hi)
    return block.end();

  // Only `requiredBefore` constrained: insert immediately before the
  // earliest one.
  if (!lo)
    return Block::iterator(hi);

  // Only `requiredAfter` constrained: insert immediately after the
  // latest one. `Block::end()` is a valid result if `lo` is the
  // terminator.
  if (!hi)
    return std::next(Block::iterator(lo));

  // Both constrained: feasible iff lo is strictly before hi. The strict
  // comparison matches the contract ("strictly before"/"strictly after"):
  // when lo == hi (same op required to be both predecessor and successor)
  // there is no legal insertion point.
  if (!lo->isBeforeInBlock(hi))
    return std::nullopt;
  return std::next(Block::iterator(lo));
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
///
/// Phase 4 (`packDynamicAllocs`) populates the structural fields:
///   `staticFactor`, `dynOperands`, and `bins`.
///
/// Phase 5 (`emitBucketSize`) populates the SSA-value fields:
///   `byteSizeValue` (= staticFactor * product(dynOperands), inserted at
///                    the builder's current position), and
///   `alignedSize`   (= alignUp(byteSizeValue, alignment), or the same
///                    SSA value when staticFactor is already aligned).
///
/// Splitting structural analysis from SSA emission lets Phase 5 pick a
/// per-bucket insertion point — see `findLatestLegalInsertionPoint` — so
/// the size arithmetic naturally lands AFTER every dyn-operand def of
/// the bucket AND above the earliest pooled alloc in the function (the
/// pool acquisition itself is anchored there, and pool size depends on
/// every bucket's `alignedSize`).
struct DynBucket {
  /// Element bytes times the product of static dims of this bucket's
  /// MemRefType. Multiplying by `dynOperands` yields the runtime byte
  /// size; rounded up to alignment when not already a multiple.
  int64_t staticFactor;
  /// SSA values for the dynamic dimensions, in declaration order. All
  /// allocs in the bucket share the same dyn operands by construction.
  SmallVector<Value, 2> dynOperands;
  /// Bins of non-overlapping allocs sharing one runtime offset.
  SmallVector<SmallVector<AllocInfo *>> bins;

  /// Populated by `emitBucketSize` at Phase 5 emission time. Null until
  /// then.
  Value byteSizeValue;
  Value alignedSize;
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
///
/// Pure structural analysis: no SSA values are emitted. Each returned
/// `DynBucket` carries the `staticFactor` and `dynOperands` needed to
/// rebuild the size at any insertion point during Phase 5; `byteSizeValue`
/// and `alignedSize` start out null.
///
/// Allocs are grouped by the structural key `{staticFactor, dynOperands}`.
/// Within each group, allocs whose lifetimes do not overlap are bin-packed
/// (first-fit) so they can share a single offset at runtime.
static SmallVector<DynBucket>
packDynamicAllocs(MutableArrayRef<AllocInfo> dynamics) {
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

  // Step 2: group allocs by structural key. We keep insertion order so the
  // emit phase processes buckets in the order they were first encountered;
  // SmallVector + linear scan is fine because the number of unique keys is
  // small (≤ tens on real models).
  SmallVector<DynBucket> buckets;
  auto findOrCreateBucket = [&](const DynSizeKey &key) -> DynBucket & {
    for (auto &b : buckets)
      if (b.staticFactor == key.staticFactor &&
          b.dynOperands == key.dynOperands)
        return b;
    DynBucket b;
    b.staticFactor = key.staticFactor;
    b.dynOperands.assign(key.dynOperands.begin(), key.dynOperands.end());
    buckets.push_back(std::move(b));
    return buckets.back();
  };

  // Step 3: first-fit bin packing within each bucket. Two allocs share a
  // bin iff their lifetimes don't overlap; bins map 1:1 to runtime
  // offsets.
  for (auto &[key, info] : keyed) {
    DynBucket &bucket = findOrCreateBucket(key);
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

  return buckets;
}

/// Emit the SSA values describing a bucket's runtime byte size at the
/// builder's current insertion point. Populates `bucket.byteSizeValue`
/// and `bucket.alignedSize`.
///
/// Before:
/// ```mlir
/// // builder at some point dominated by all bucket.dynOperands
/// ```
/// After (general case):
/// ```mlir
/// %static = arith.constant <staticFactor> : index
/// %byte   = arith.muli %static, %dyn0 : index
/// %byte   = arith.muli %byte,   %dyn1 : index    // for each dyn operand
/// // alignedSize = alignUp(%byte, alignment), unless staticFactor is
/// // already a multiple of alignment — then alignedSize == byteSizeValue.
/// ```
static void emitBucketSize(OpBuilder &builder, Location loc, DynBucket &bucket,
                           int64_t alignment) {
  Value byteSize =
      arith::ConstantIndexOp::create(builder, loc, bucket.staticFactor);
  for (Value dynDim : bucket.dynOperands)
    byteSize = builder.createOrFold<arith::MulIOp>(loc, byteSize, dynDim);
  bucket.byteSizeValue = byteSize;
  // When staticFactor is a multiple of alignment, the byte size
  // (staticFactor * dynDim0 * dynDim1 * ...) is guaranteed to already
  // be a multiple of alignment regardless of the dynamic dimensions, so
  // alignUp would be a semantic no-op. Skip it explicitly to avoid
  // emitting the divui + muli + addi triple it would otherwise produce.
  if (bucket.staticFactor % alignment == 0)
    bucket.alignedSize = byteSize;
  else
    bucket.alignedSize = emitAlignUp(builder, loc, byteSize, alignment);
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
    // Nothing to pool: a single alloc would just be replaced by a
    // single-view-into-pool, which is strictly worse than leaving the
    // alloc in place. Still emit zeroed pool metadata so downstream
    // consumers (GenerateInterface) see the attributes regardless of
    // input shape — a missing attribute would crash the metadata
    // reader, while a zeroed one is a well-defined "no pool".
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

  // ----- Phase 4: pack dynamics (structural only — no SSA emission) ---

  Location loc = funcOp.getLoc();
  OpBuilder builder(funcOp.getContext());

  auto dynBuckets = packDynamicAllocs(dynamics);
  NumDynBuckets += dynBuckets.size();
  bool hasDynamic = !dynBuckets.empty();

  if (!hasDynamic && staticPoolSize == 0)
    return;

  // ----- Phase 5: emit pool + views -----------------------------------
  //
  // Per-bucket size arithmetic and the pool acquisition are placed at
  // dominator-derived points selected by `findLatestLegalInsertionPoint`.
  //
  // Critical invariant: there is exactly ONE `hip.get_pool` per function,
  // and every `memref.view` replacing a pooled alloc consumes its result.
  // The pool therefore must dominate every pooled alloc in the function
  // — its insertion point is anchored at the EARLIEST pooled alloc.
  // Pool size is `sum_buckets(alignedSize * numBins)`, so each bucket's
  // `alignedSize` must in turn dominate the pool, and therefore must
  // also live above the earliest alloc. We pass the same
  // `requiredBefore = allAllocOps` to both calls and let the helper
  // compute the per-call insertion point from each call's
  // `requiredAfter` set:
  //
  //   - Per bucket: requiredAfter = dyn-operand defs of THIS bucket.
  //     Position lands above the earliest alloc but at or below the
  //     latest dyn-operand def — naturally placing each bucket's size
  //     close to the IR that produced its inputs.
  //   - Pool acquisition: requiredAfter = every bucket's `alignedSize`.
  //     Position lands above the earliest alloc and at or below the
  //     LAST bucket size emitted.
  //   - View replacement: unchanged, at each alloc's original site.
  //     Offsets dominate by construction (helper's contract).
  //
  // Before:
  // ```mlir
  // %d0 = memref.dim %arg, %c0 : memref<?x?xf16>   // bucket dyn operand
  // ... unrelated ops ...
  // %a0 = memref.alloc(%d0) : memref<?xf16>        // earliest alloc
  // ... unrelated ops ...
  // %a1 = memref.alloc(%d0) : memref<?xf16>
  // ```
  // After (size + pool + views inserted at dominator-derived anchors):
  // ```mlir
  // %d0 = memref.dim %arg, %c0 : memref<?x?xf16>
  // ... unrelated ops ...
  // %static = arith.constant <staticFactor> : index
  // %byte   = arith.muli %static, %d0 : index
  // %aligned = ...                                  // when needed
  // %pool   = hip.get_pool(%ctx, %size) : memref<?xi8>
  // ... offset arithmetic ...
  // %v0 = memref.view %pool[%off0][%d0] : memref<?xi8> to memref<?xf16>
  // ... unrelated ops ...
  // %v1 = memref.view %pool[%off1][%d0] : memref<?xi8> to memref<?xf16>
  // ```

  // Cached `requiredBefore` for both insertion-point calls — every
  // pooled alloc in the function. The helper picks the earliest of the
  // set, so passing the full list (instead of just the earliest) keeps
  // the call correct under any future allocation reordering inside the
  // function body.
  SmallVector<Operation *> allAllocOps;
  allAllocOps.reserve(allInfos.size());
  for (auto &info : allInfos)
    allAllocOps.push_back(info.allocOp.getOperation());

  for (auto &bucket : dynBuckets) {
    SmallVector<Operation *> requiredAfter;
    for (Value dynOp : bucket.dynOperands)
      if (auto *def = dynOp.getDefiningOp())
        if (def->getBlock() == &block)
          requiredAfter.push_back(def);
    auto ip = findLatestLegalInsertionPoint(block, requiredAfter, allAllocOps);
    if (!ip) {
      // Infeasible iff some dyn-operand def of this bucket lives
      // at-or-after the earliest pooled alloc in the function. That
      // requires the dyn-operand to be defined BETWEEN two pooled
      // allocs (e.g. `memref.dim` of an earlier alloc consumed by a
      // later alloc) — a pattern a Stage 0 IR audit confirmed does not
      // occur in practice on lowered transformer graphs. The legacy
      // hoist path supported it via code motion; the dominator-emit
      // path intentionally does not, so we surface it as a hard
      // failure to flag any future regression.
      funcOp.emitError(
          "hip-pool-allocs: cannot place bucket size arithmetic at a "
          "legal point in the block (a dyn-operand def is at-or-after "
          "the earliest pooled alloc)");
      return signalPassFailure();
    }
    builder.setInsertionPoint(&block, *ip);
    emitBucketSize(builder, loc, bucket, align);
  }

  // Pool acquisition insertion point: after every bucket's aligned size
  // SSA value, before every alloc.
  {
    SmallVector<Operation *> requiredAfter;
    for (auto &bucket : dynBuckets) {
      // alignedSize is the LATEST op in the bucket's size chain. When
      // staticFactor is already aligned, alignedSize == byteSizeValue
      // (still the latest). Block args and folded constants have no
      // defining op and need no constraint.
      Value v = bucket.alignedSize ? bucket.alignedSize : bucket.byteSizeValue;
      if (!v)
        continue;
      if (auto *def = v.getDefiningOp())
        if (def->getBlock() == &block)
          requiredAfter.push_back(def);
    }
    auto ip = findLatestLegalInsertionPoint(block, requiredAfter, allAllocOps);
    if (!ip) {
      funcOp.emitError(
          "hip-pool-allocs: cannot place hip.get_pool at a legal point "
          "(some bucket size arithmetic is at-or-after an alloc)");
      return signalPassFailure();
    }
    builder.setInsertionPoint(&block, *ip);
  }

  // Pool size = staticPoolSize + sum_buckets(alignedSize * numBins).
  // Uses createOrFold so trivial ops (addi(x, 0), muli(x, 1)) are folded
  // away by the arith dialect's fold methods.
  Value poolSize = arith::ConstantIndexOp::create(builder, loc, staticPoolSize);
  if (hasDynamic) {
    for (auto &bucket : dynBuckets) {
      Value numBinsVal = arith::ConstantIndexOp::create(
          builder, loc, static_cast<int64_t>(bucket.bins.size()));
      Value contribution = builder.createOrFold<arith::MulIOp>(
          loc, bucket.alignedSize, numBinsVal);
      poolSize =
          builder.createOrFold<arith::AddIOp>(loc, poolSize, contribution);
    }
  }

  LLVM_DEBUG(llvm::dbgs() << "Pool: static=" << staticPoolSize << " bytes, "
                          << dynBuckets.size() << " dynamic buckets, "
                          << allInfos.size() << " total allocs\n");

  // Acquire the pool via hip.get_pool(%ctx, %pool_size).
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

  // Compute byte offsets for every alloc and store in allocToOffset.
  // Static offsets are emitted as constants right after the pool. Dynamic
  // offsets are walked bucket-by-bucket, base advancing past each bucket
  // so distinct buckets never alias.
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
        Value binIdxVal = arith::ConstantIndexOp::create(
            builder, loc, static_cast<int64_t>(binIdx));
        Value binContrib = builder.createOrFold<arith::MulIOp>(
            loc, bucket.alignedSize, binIdxVal);
        Value binOffset =
            builder.createOrFold<arith::AddIOp>(loc, currentBase, binContrib);
        for (AllocInfo *info : bin)
          allocToOffset[info->allocOp.getOperation()] = binOffset;
      }
      Value numBinsVal = arith::ConstantIndexOp::create(
          builder, loc, static_cast<int64_t>(bucket.bins.size()));
      Value bucketTotal = builder.createOrFold<arith::MulIOp>(
          loc, bucket.alignedSize, numBinsVal);
      currentBase =
          builder.createOrFold<arith::AddIOp>(loc, currentBase, bucketTotal);
    }
  }

  // Replace each original alloc with a memref.view into the pool.
  // The view lands at the alloc's original site — offsets dominate by
  // construction (helper-anchored above the pool acquisition).
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

  // Erase deallocs that now target views — the pool is owned by the
  // runtime state (not this function), so no pool dealloc is inserted.
  SmallVector<memref::DeallocOp> orphanedDeallocs;
  funcOp.walk([&](memref::DeallocOp op) {
    if (op.getMemref().getDefiningOp<memref::ViewOp>())
      orphanedDeallocs.push_back(op);
  });
  for (auto op : orphanedDeallocs)
    op.erase();

  // Attach pool metadata to the module for GenerateInterface.
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
