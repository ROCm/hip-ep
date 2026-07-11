/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PoolAllocs.cpp - Pack memref.alloc into per-domain i8 pools --------===//
//
// Packs all memref.alloc ops in a single-block function into one or more
// contiguous byte pools (memref<?xi8>), replacing each original alloc with a
// memref.view at a computed byte offset. Each pool is acquired via
// hip.get_pool(%ctx, %pool_size) so the runtime can grow on demand.
//
// Single-domain output (the canonical case where every pooled alloc's
// dyn-operand defs sit above the earliest alloc) is bit-identical to the
// pre-multi-domain output: one hip.get_pool, one set of views, the legacy
// `hipdnn.pool_size` / `hipdnn.buffer_count` / `hipdnn.buffer_offsets`
// module attributes only.
//
// Multi-domain output emerges only when some alloc's dyn-operand chain comes
// from values defined BELOW an earlier pooled alloc — typically a
// `memref.load` of a host-staged scalar that itself sits between two pooled
// allocs and cannot be hoisted above the earlier one. Each domain gets its
// own hip.get_pool (one anchor per domain) plus per-domain offsets; the
// multi-domain runtime maps each domain to a separate growable buffer.
//
// Algorithm overview:
//   Phase 1   - Liveness: assign [defIndex, lastUseIndex] to each alloc,
//               following view-like ops transitively via
//               BufferViewFlowAnalysis.
//   Phase 1.5 - Domain partition: greedy textual-order clustering of allocs
//               by dominance feasibility. Allocs whose dyn-operand chain is
//               defined below the earliest alloc of every existing domain
//               open a new domain.
//   Phase 2..5 (PER DOMAIN):
//     Phase 2 - Partition: split this domain's allocs into static and dynamic.
//     Phase 3 - Static packing: greedy best-fit offset assignment within
//               this domain.
//     Phase 4 - Dynamic packing: color runtime-sized allocs by lifetime so
//               lifetime-disjoint allocs share a pool slab (Phase 4 header).
//     Phase 5 - IR emission: insert this domain's hip.get_pool and views;
//               offsets are local to this domain's pool.
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
STATISTIC(NumDomains, "Number of dominance domains (separate pools) created");
STATISTIC(NumStaticFragBytes,
          "Static-pool bytes beyond the max-load lower bound (fragmentation)");
STATISTIC(NumDynFragUnits,
          "Dynamic-pool staticFactor units beyond the max-load lower bound "
          "(single-F domains only, pre-scale by the dynamic factor)");
STATISTIC(NumSmallBucketExcessBins,
          "Small-bucket bins beyond the peak concurrent count");

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

/// Byte size of a memref's STATIC dims: elementBytes * product(static dims).
/// For a static memref this is the whole byte size; for a dynamic one it is the
/// per-alloc `staticFactor` -- multiply by product(dynOperands) for the runtime
/// byte size. Shared by the dynamic packer (as `staticFactor`) and the
/// fragmentation probe (which compares aligned groups in these units, since the
/// dynamic factor is common and symbolic within a group).
static int64_t staticFactorBytes(MemRefType type) {
  int64_t staticElems = 1;
  for (int64_t dim : type.getShape())
    if (!ShapedType::isDynamic(dim))
      staticElems *= dim;
  return static_cast<int64_t>(
      llvm::divideCeil(staticElems * type.getElementTypeBitWidth(), 8));
}

/// Max-load lower bound: the peak sum of `sizeOf` over members live at a single
/// point in time. No contiguous packing of these intervals can be smaller, so a
/// packer's highwater minus this is its fragmentation. The load only rises when
/// an interval starts, so the peak occurs at some member's def point; summing
/// the members live AT each such point (not merely overlapping the anchor's
/// interval -- two members can each overlap the anchor yet never coexist) is
/// exact. O(n^2), fine here (n = allocs per domain, tens in practice).
static int64_t
maxConcurrentLoad(ArrayRef<const AllocInfo *> members,
                  llvm::function_ref<int64_t(const AllocInfo *)> sizeOf) {
  int64_t peak = 0;
  for (const AllocInfo *anchor : members) {
    unsigned point = anchor->defIndex;
    int64_t load = 0;
    for (const AllocInfo *other : members)
      if (other->defIndex <= point && point <= other->lastUseIndex)
        load += sizeOf(other); // live at `point` (includes anchor itself)
    peak = std::max(peak, load);
  }
  return peak;
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
// Phase 1.5 - Dominance-domain partition
//===----------------------------------------------------------------------===//
//
// A Domain holds a subset of pooled allocs that share a common dominance
// feasibility region — i.e. there is at least one block position that
// strictly dominates every alloc in the domain AND is strictly dominated by
// every dyn-operand def of every alloc in the domain. Such a position is
// the only place where a single hip.get_pool can be inserted that's both:
//
//   - reachable from every alloc's view replacement (pool dominates uses), AND
//   - reachable by every alloc's dyn-size SSA (dyn-defs dominate pool size).
//
// Allocs whose dyn-defs are defined BELOW the earliest pooled alloc in a
// domain cannot share that domain's pool; they require their own.

struct Domain {
  /// Pooled allocs assigned to this domain, in textual (defIndex) order.
  SmallVector<AllocInfo *> allocs;
};

/// Advisory threshold on the number of domains a single function produces.
/// This is NOT a cap — the runtime grows its per-domain pool arrays on demand
/// with no upper bound, so any domain count compiles and runs correctly. But a
/// well-canonicalised graph produces 1 domain (the typical case once size
/// arithmetic has been hoisted to block top) or 2 (when a host-load-dependent
/// alloc forces a second domain). A count far above that usually means the
/// earlier `--hip-hoist-alloc-size-arith` pass did not run, or size arithmetic
/// it should have lifted is still pinned below its allocs — so we emit a
/// non-fatal remark to flag the likely pre-condition gap, then proceed
/// normally.
static constexpr unsigned kDomainCountAdvisoryThreshold = 8;

/// Greedy textual-order clustering of pooled allocs into dominance domains.
///
/// Algorithm:
///
///   for each alloc A in textual order:
///     for each existing domain D, most-recently-created first:
///       speculatively add A to D
///       if D ∪ {A} still admits a single common insertion point: keep
///       else: roll back
///     if no D accepted A: open a new domain {A}
///
/// Most-recent-first iteration order is deliberate: domains created later
/// have later first-allocs and so their dominance window admits the
/// strictest constraints. If alloc A's dyn-operand chain lives above the
/// latest-created domain D_recent, the merge into D_recent succeeds; if A's
/// dyn-defs are below D_recent's earliest alloc, A is structurally below
/// every earlier domain's earliest alloc too (their first-allocs are
/// earlier) and merging there will also fail. So the first feasible domain
/// is always the most recent one — but we still walk in reverse-creation
/// order to remain robust against future reorderings.
///
/// Single-domain output (the typical post-hoist case): every alloc's
/// dyn-operand defs dominate every pooled alloc, so the first probe of D0
/// always succeeds and partitionByDominanceDomain returns one domain
/// containing every alloc — exactly the single-domain input shape the
/// downstream packing phases expect.
static SmallVector<Domain>
partitionByDominanceDomain(MutableArrayRef<AllocInfo> allInfos, Block &block) {
  SmallVector<AllocInfo *> ordered;
  ordered.reserve(allInfos.size());
  for (AllocInfo &info : allInfos)
    ordered.push_back(&info);
  llvm::sort(ordered, [](const AllocInfo *a, const AllocInfo *b) {
    return a->defIndex < b->defIndex;
  });

  // Feasibility probe: collect the union of requiredAfter and requiredBefore
  // sets across `allocs` and ask findLatestLegalInsertionPoint whether any
  // common strict insertion point exists.
  auto domainFeasible = [&block](ArrayRef<AllocInfo *> allocs) -> bool {
    SmallVector<Operation *> requiredAfter;
    SmallVector<Operation *> requiredBefore;
    for (AllocInfo *info : allocs) {
      requiredBefore.push_back(info->allocOp.getOperation());
      for (Value dynOp : info->allocOp.getDynamicSizes())
        if (auto *def = dynOp.getDefiningOp())
          if (def->getBlock() == &block)
            requiredAfter.push_back(def);
    }
    return findLatestLegalInsertionPoint(block, requiredAfter, requiredBefore)
        .has_value();
  };

  SmallVector<Domain> domains;
  for (AllocInfo *info : ordered) {
    bool merged = false;
    for (Domain &domain : llvm::reverse(domains)) {
      domain.allocs.push_back(info);
      if (domainFeasible(domain.allocs)) {
        merged = true;
        break;
      }
      domain.allocs.pop_back();
    }
    if (!merged) {
      Domain d;
      d.allocs.push_back(info);
      domains.push_back(std::move(d));
    }
  }
  return domains;
}

//===----------------------------------------------------------------------===//
// Phase 3 - Static packing: greedy best-fit gap finding
//===----------------------------------------------------------------------===//
//
// Assigns a byte offset in a 1D address space to each static alloc.
// Two allocs whose lifetimes don't overlap may share the same address range.
//
// Example (4 static allocs of 32768 bytes each, fully overlapping lifetimes):
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

/// Greedy best-fit gap packer over a lifetime-annotated set.
///
/// `items` MUST be pre-sorted largest-first (by `sizeOf`) for good packing.
/// For each item, scan the reservations whose lifetimes overlap, find the
/// smallest gap that fits, and place it there; otherwise append after the
/// last overlapping reservation. Returns `(item, alignedByteOffset)` and sets
/// `highwater` = total span (aligned). `sizeOf` gives each item's UNALIGNED
/// byte size (the packer aligns it).
///
/// Shared by Phase 3 (static packing, `sizeOf = staticByteSize`) and Phase 4's
/// aligned-dynamic groups (`sizeOf = staticFactor`, offsets later scaled by the
/// group's common dynamic factor -- see `packDynamicAllocs`).
static SmallVector<std::pair<AllocInfo *, int64_t>>
packGreedyBestFit(ArrayRef<AllocInfo *> items,
                  llvm::function_ref<int64_t(const AllocInfo *)> sizeOf,
                  int64_t alignment, int64_t &highwater) {
  SmallVector<std::pair<AllocInfo *, int64_t>> assignments;
  SmallVector<Reservation, 16> reservations; // kept sorted by offset
  highwater = 0;

  for (AllocInfo *info : items) {
    int64_t size = llvm::alignTo(sizeOf(info), alignment);
    int64_t bestOffset = -1;
    int64_t bestFit = INT64_MAX;

    // Walk existing reservations with overlapping lifetimes and look for
    // the smallest gap between them that fits this allocation.
    int64_t currentOffset = 0;
    for (auto &res : reservations) {
      if (!lifetimesOverlap(*info, *res.info))
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
    Reservation newRes{info, bestOffset, size};
    auto insertIt = reservations.begin();
    while (insertIt != reservations.end() && insertIt->offset < newRes.offset)
      ++insertIt;
    reservations.insert(insertIt, newRes);
    assignments.emplace_back(info, bestOffset);
    highwater = std::max(highwater, bestOffset + size);
  }

  return assignments;
}

/// Assign byte offsets to static allocs using greedy best-fit (thin wrapper
/// over `packGreedyBestFit`). `statics` is pre-sorted largest-first by the
/// caller.
static SmallVector<std::pair<AllocInfo *, int64_t>>
packStaticAllocs(MutableArrayRef<AllocInfo> statics, int64_t alignment) {
  SmallVector<AllocInfo *> items;
  items.reserve(statics.size());
  for (AllocInfo &info : statics)
    items.push_back(&info);
  int64_t highwater = 0;
  return packGreedyBestFit(
      items, [](const AllocInfo *i) { return i->staticByteSize; }, alignment,
      highwater);
}

//===----------------------------------------------------------------------===//
// Phase 4 - Dynamic packing: factored static packing + small-bucket fallback
//===----------------------------------------------------------------------===//
//
// For allocs with runtime-unknown sizes we cannot hardcode byte offsets. Each
// alloc's byte size is `staticFactor * F`, where staticFactor = elementBytes *
// product(static dims) and F = product(dynOperands) is a runtime value.
//
// Default (lifetime-only): every runtime-sized alloc is its own group, then a
// group-level lifetime coloring lets ANY two lifetime-disjoint allocs share one
// pool slab (slab width = max of the member footprints) regardless of their
// dynOperands. A slab's cost is therefore the MAX of its concurrently-live
// members, not the SUM of every alloc routed to it. Groups are ordered
// largest-staticFactor-first -- the only compile-time size proxy, since F is
// symbolic -- so the biggest members dominate each slab width. Only
// proven-disjoint allocs ever share, so overlapping address ranges is always
// safe (no mis-pairing).
//
// Each group carries its own F, emitted as `product(dynOperands)`. An alloc's
// byte offset is `slabBase + unitOffset * F`. When staticFactor is a multiple
// of the alignment the packed unit offsets are too, so the scaled offset stays
// aligned for any F; a slab that also holds a non-multiple alloc rounds its
// width up so the next slab's base stays aligned.
//
// Fallback (lifetime-only = false): group by dynOperands and best-fit-pack the
// integer staticFactors within a group (offsets/span later scaled by the common
// F), then run the same cross-group coloring. Non-alignment-multiple allocs go
// to per-size lifetime bins in byte space (`DynBucket`) so their offsets are
// not rounded up and scaled by F. This is strictly <= the default's per-group
// footprint and never mis-pairs; kept as a conservative escape hatch.

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

/// An aligned-dynamic group: allocs that share the same `dynOperands` and whose
/// `staticFactor` is a multiple of the pool alignment.  All member byte sizes
/// are `staticFactor_i * F` with a common `F = product(dynOperands)`.  We pack
/// the integer `staticFactor`s with the static greedy packer -- non-overlap is
/// scale-invariant, so scaling every offset/size by the common `F` preserves it
/// -- then multiply offsets and the span by `F` at emit.  Because every
/// `staticFactor` is a multiple of `alignment`, the packed unit offsets are
/// too, so `unitOffset * F` stays aligned for any integer `F` (no per-alloc
/// alignment padding).
///
/// `assignments` = (alloc, unitOffset in staticFactor bytes); `spanUnits` =
/// highwater in staticFactor bytes.  Emitted byte offset = base + unitOffset*F.
struct AlignedDynGroup {
  SmallVector<Value, 2> dynOperands;
  SmallVector<std::pair<AllocInfo *, int64_t>> assignments;
  int64_t spanUnits = 0;
};

/// Phase-4 result: aligned groups (factored static packing -- the memory-
/// dominant case) plus small buckets (`staticFactor` not a multiple of the
/// alignment -> per-size bucket/bin packing in byte space).  Small allocs stay
/// off the factoring path because rounding their unit offsets up to the pool
/// alignment would inflate each reservation by a factor of F; keeping them
/// byte-aligned avoids that, and their byte contribution is negligible.
struct DynPacking {
  SmallVector<AlignedDynGroup> alignedGroups;
  SmallVector<DynBucket> smallBuckets;
  /// Lifetime coloring of the aligned groups: each entry lists the indices
  /// (into `alignedGroups`) of groups that are pairwise fully lifetime-disjoint
  /// and so may **share one pool slab** -- the slab's width is the max of its
  /// members' footprints, not their sum. Groups whose lifetimes overlap land in
  /// different slabs. This sharing spans groups with different `dynOperands`,
  /// so it is not limited to a single dynamic factor.
  ///
  ///   Stack every group           Disjoint groups share a slab
  ///     [ G0 ][ G1 ][ G2 ]          [ G0 / G2 ][ G1 ]     (G0, G2 disjoint)
  ///     size = f0 + f1 + f2         size = max(f0, f2) + f1
  ///
  /// A slab with a single member reproduces plain stacking (max of one).
  SmallVector<SmallVector<unsigned>> alignedSuperBins;
  bool empty() const { return alignedGroups.empty() && smallBuckets.empty(); }
};

/// Group dynamic allocs, color them by lifetime, and record which groups may
/// share a slab (see the Phase 4 header, AlignedDynGroup, and DynPacking).
///
/// Default (`lifetimeOnly`): one group per alloc, so the cross-group coloring
/// can share a slab between any two lifetime-disjoint allocs regardless of
/// their `dynOperands`. Fallback: group by `dynOperands` and best-fit-pack the
/// integer `staticFactor`s within each group before the same coloring, sending
/// non-alignment-multiple allocs to per-size byte-space bins instead.
static DynPacking packDynamicAllocs(MutableArrayRef<AllocInfo> dynamics,
                                    int64_t alignment, bool lifetimeOnly) {
  // staticFactor (elem bytes * product of static dims) + dynOperands per alloc.
  auto computeKey = [](AllocInfo &info) {
    MemRefType type = info.allocOp.getType();
    auto dynSizes = info.allocOp.getDynamicSizes();
    int64_t staticFactor = staticFactorBytes(type);
    SmallVector<Value, 2> dynOperands;
    unsigned dynIdx = 0;
    for (int64_t dim : type.getShape())
      if (ShapedType::isDynamic(dim))
        dynOperands.push_back(dynSizes[dynIdx++]);
    return std::make_pair(staticFactor, dynOperands);
  };

  DynPacking result;
  DenseMap<const AllocInfo *, int64_t> staticFactorOf;

  // ALIGNED allocs: collect per `dynOperands` group (SSA identity) before
  // packing, so each group can be sorted largest-first.
  struct AlignedTmp {
    SmallVector<Value, 2> dynOperands;
    SmallVector<AllocInfo *> allocs;
  };
  SmallVector<AlignedTmp> alignedTmp;
  auto findAligned = [&](ArrayRef<Value> ops) -> AlignedTmp & {
    for (auto &t : alignedTmp)
      if (ArrayRef<Value>(t.dynOperands) == ops)
        return t;
    alignedTmp.push_back({SmallVector<Value, 2>(ops.begin(), ops.end()), {}});
    return alignedTmp.back();
  };

  // SMALL allocs: bucket by `{staticFactor, dynOperands}` + first-fit lifetime
  // bins (per-size packing in byte space).
  auto findOrCreateBucket = [&](int64_t sf,
                                ArrayRef<Value> ops) -> DynBucket & {
    for (auto &b : result.smallBuckets)
      if (b.staticFactor == sf && ArrayRef<Value>(b.dynOperands) == ops)
        return b;
    DynBucket b;
    b.staticFactor = sf;
    b.dynOperands.assign(ops.begin(), ops.end());
    result.smallBuckets.push_back(std::move(b));
    return result.smallBuckets.back();
  };
  auto binPack = [](DynBucket &bucket, AllocInfo *info) {
    for (auto &bin : bucket.bins) {
      bool conflicts = false;
      for (AllocInfo *e : bin)
        if (lifetimesOverlap(*info, *e)) {
          conflicts = true;
          break;
        }
      if (!conflicts) {
        bin.push_back(info);
        return;
      }
    }
    bucket.bins.push_back({info});
  };

  for (AllocInfo &info : dynamics) {
    auto [sf, ops] = computeKey(info);
    staticFactorOf[&info] = sf;
    // In the default, every dynamic alloc (aligned or not) is packed by the
    // lifetime coloring below -- no size bucketing. The small-bucket path is
    // used only by the fallback (lifetimeOnly == false).
    if (lifetimeOnly || sf % alignment == 0)
      findAligned(ops).allocs.push_back(&info);
    else
      binPack(findOrCreateBucket(sf, ops), &info);
  }

  if (lifetimeOnly) {
    // Each aligned alloc is its own group; the cross-group coloring below then
    // shares slabs across any lifetime-disjoint allocs. Order largest-
    // staticFactor-first so bigger members dominate each slab width (the only
    // compile-time size proxy, since F is symbolic).
    for (AlignedTmp &t : alignedTmp)
      for (AllocInfo *a : t.allocs) {
        AlignedDynGroup g;
        g.dynOperands = t.dynOperands;
        g.assignments.push_back({a, /*unitOffset=*/0});
        g.spanUnits = staticFactorOf.lookup(a);
        result.alignedGroups.push_back(std::move(g));
      }
    llvm::stable_sort(result.alignedGroups,
                      [](const AlignedDynGroup &a, const AlignedDynGroup &b) {
                        return a.spanUnits > b.spanUnits;
                      });
  } else {
    // Fallback: best-fit-pack each dynOperands group's integer staticFactors,
    // largest-first (offsets scaled by the common F at emit).
    for (AlignedTmp &t : alignedTmp) {
      llvm::stable_sort(t.allocs, [&](AllocInfo *a, AllocInfo *b) {
        return staticFactorOf.lookup(a) > staticFactorOf.lookup(b);
      });
      AlignedDynGroup g;
      g.dynOperands = std::move(t.dynOperands);
      g.assignments = packGreedyBestFit(
          t.allocs,
          [&](const AllocInfo *i) { return staticFactorOf.lookup(i); },
          alignment, g.spanUnits);
      result.alignedGroups.push_back(std::move(g));
    }
  }

  // Cross-group lifetime coloring: place each group in the first slab whose
  // members are ALL lifetime-disjoint from it, else start a new slab. Only
  // proven-disjoint groups ever share, so overlapping their address ranges is
  // always safe. Deterministic (groups visited in build order).
  auto groupsDisjoint = [](const AlignedDynGroup &a, const AlignedDynGroup &b) {
    return llvm::none_of(a.assignments, [&](const auto &ea) {
      return llvm::any_of(b.assignments, [&](const auto &eb) {
        return lifetimesOverlap(*ea.first, *eb.first);
      });
    });
  };
  for (auto [gi, group] : llvm::enumerate(result.alignedGroups)) {
    auto *slab =
        llvm::find_if(result.alignedSuperBins, [&](ArrayRef<unsigned> members) {
          return llvm::all_of(members, [&](unsigned other) {
            return groupsDisjoint(group, result.alignedGroups[other]);
          });
        });
    if (slab == result.alignedSuperBins.end())
      result.alignedSuperBins.push_back({static_cast<unsigned>(gi)});
    else
      slab->push_back(static_cast<unsigned>(gi));
  }

  return result;
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

  // Bail only when there is nothing to pool. A SINGLE surviving alloc must
  // still be pooled: it lowers to hip.alloc -> hip_device_malloc, which has no
  // runtime definition (the pipeline assumes every transient is either pooled
  // or written through to an out-param). The canonical single-alloc trigger is
  // a one-op graph whose op needs a destination temp it can't write through,
  // e.g. a lone rank-3 Conv (the collapse_shape before the return blocks
  // write-through). Pooling a single alloc is cheap (pool_size == that alloc's
  // size) and keeps it on the RuntimeState-managed pool instead of an
  // undefined per-call malloc.
  if (allInfos.empty()) {
    // Still emit zeroed pool metadata so downstream consumers
    // (GenerateInterface) see the attributes regardless of input shape — a
    // missing attribute would crash the metadata reader, while a zeroed one
    // is a well-defined "no pool".
    ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
    OpBuilder zeroBuilder(funcOp.getContext());
    moduleOp->setAttr("hipdnn.pool_size", zeroBuilder.getI64IntegerAttr(0));
    moduleOp->setAttr("hipdnn.buffer_count", zeroBuilder.getI64IntegerAttr(0));
    moduleOp->setAttr("hipdnn.buffer_offsets",
                      zeroBuilder.getArrayAttr(SmallVector<Attribute>{}));
    return;
  }

  // Validate function signature once: hip.context arg 0 is required for
  // every domain's hip.get_pool emission below.
  if (funcOp.getNumArguments() == 0 ||
      !isa<hip::ContextType>(funcOp.getArgument(0).getType())) {
    funcOp.emitError("function missing !hip.context as arg 0 — "
                     "run hip-add-context-arg before hip-pool-allocs");
    return signalPassFailure();
  }

  // ----- Phase 1.5: partition allocs into dominance domains ------------

  SmallVector<Domain> domains = partitionByDominanceDomain(allInfos, block);
  NumDomains += domains.size();

  // The domain count is unbounded — the runtime grows its per-domain pool
  // arrays on demand — so a high count is correct, just suspicious. Emit a
  // non-fatal remark (compilation continues) pointing at the likely missing
  // pipeline pre-condition, rather than failing the pass.
  if (domains.size() > kDomainCountAdvisoryThreshold) {
    funcOp.emitRemark("hip-pool-allocs: dominance partition produced ")
        << domains.size() << " domains (advisory threshold "
        << kDomainCountAdvisoryThreshold
        << "); this usually means the earlier --hip-hoist-alloc-size-arith "
           "pass did not run, or canonicalization left unhoistable size "
           "arithmetic in place. Compilation continues.";
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Domains: " << domains.size() << "\n";
    for (auto [i, domain] : llvm::enumerate(domains))
      llvm::dbgs() << "  domain " << i << ": " << domain.allocs.size()
                   << " allocs\n";
  });

  // Position-in-allInfos lookup, used to write per-alloc metadata into
  // parallel vectors keyed on textual order. Built once before any erase.
  DenseMap<Operation *, unsigned> allocToPos;
  for (auto [i, info] : llvm::enumerate(allInfos))
    allocToPos[info.allocOp.getOperation()] = static_cast<unsigned>(i);

  // Per-alloc results, indexed by allInfos position. `bufferOffsets` is
  // -1 for dynamic-offset allocs (legacy semantics) and a non-negative
  // byte offset for constant-offset allocs. `bufferDomains` records which
  // pool each alloc consumes.
  SmallVector<int64_t> bufferOffsets(allInfos.size(), -1);
  SmallVector<unsigned> bufferDomains(allInfos.size(), 0);
  // Per-domain static prefix size (the constant `staticPoolSize` baked
  // into each domain's pool acquisition). Dynamic contributions are
  // baked into the runtime arith chain feeding hip.get_pool's size
  // operand and are NOT reflected here.
  SmallVector<int64_t> domainPoolStaticSizes;
  domainPoolStaticSizes.reserve(domains.size());

  // ----- Phases 2..5 (per domain) --------------------------------------

  Location loc = funcOp.getLoc();
  OpBuilder builder(funcOp.getContext());
  int64_t align = alignment;
  Value ctx = funcOp.getArgument(0);
  auto poolType =
      MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(8));

  for (auto [domainId, domain] : llvm::enumerate(domains)) {
    // Phase 2: split this domain's allocs into static / dynamic.
    SmallVector<AllocInfo> statics, dynamics;
    for (AllocInfo *info : domain.allocs) {
      if (info->staticByteSize > 0)
        statics.push_back(*info);
      else
        dynamics.push_back(*info);
    }
    llvm::sort(statics, [](const AllocInfo &a, const AllocInfo &b) {
      return a.staticByteSize > b.staticByteSize;
    });

    // Phase 3: pack statics within this domain.
    auto staticAssignments = packStaticAllocs(statics, align);
    NumStaticPacked += staticAssignments.size();

    int64_t staticPoolSize = 0;
    for (auto &[info, offset] : staticAssignments) {
      int64_t end = offset + llvm::alignTo(info->staticByteSize, align);
      staticPoolSize = std::max(staticPoolSize, end);
    }

    // Phase 4: bucket dynamics within this domain.
    auto dynPacking = packDynamicAllocs(dynamics, align, lifetimeOnly);
    // Count distinct dynamic slabs (lifetime-shared super-bins + small
    // buckets), i.e. the runtime footprint, not the per-alloc group count.
    NumDynBuckets +=
        dynPacking.alignedSuperBins.size() + dynPacking.smallBuckets.size();
    bool hasDynamic = !dynPacking.empty();
    // F = product(dynOperands) per aligned group; emitted below (Phase 5) and
    // reused for both the pool-size and per-alloc offset arithmetic.
    SmallVector<Value> groupFactors(dynPacking.alignedGroups.size());

    // Debug-only fragmentation probe. Reads the packing result (emits no IR,
    // changes no offsets) and compares each part of the domain footprint to its
    // max-load lower bound -- the peak sum of concurrently-live sizes, which no
    // contiguous packing can beat. The surplus is recoverable fragmentation; a
    // zero surplus means the packer reached the theoretical floor.
    //
    // Static bytes and small-bucket bins compare exactly. The dynamic pool
    // compares in staticFactor units: the emitted coefficient is the sum over
    // slabs of the max member footprint, and the floor is the max-load over all
    // aligned allocs. That is only meaningful when every aligned group shares
    // one dynamic factor F ("single-F"); a mixed-F domain is reported as such.
    // Runs unconditionally (accumulates statistics); only the per-domain remark
    // is gated behind the emit-fragmentation-report option.
    {
      SmallVector<const AllocInfo *> members;
      members.reserve(statics.size());
      for (const AllocInfo &s : statics)
        members.push_back(&s);
      int64_t staticLb = maxConcurrentLoad(members, [&](const AllocInfo *i) {
        return llvm::alignTo(i->staticByteSize, align);
      });
      int64_t staticFrag = staticPoolSize - staticLb;
      NumStaticFragBytes += staticFrag;

      ArrayRef<AlignedDynGroup> groups = dynPacking.alignedGroups;
      bool singleF = llvm::all_of(groups, [&](const AlignedDynGroup &g) {
        return ArrayRef<Value>(g.dynOperands) ==
               ArrayRef<Value>(groups.front().dynOperands);
      });
      int64_t dynPoolUnits = 0;
      for (ArrayRef<unsigned> slab : dynPacking.alignedSuperBins) {
        int64_t width = 0;
        for (unsigned gi : slab)
          width = std::max(width, groups[gi].spanUnits);
        dynPoolUnits += width;
      }
      members.clear();
      for (const AlignedDynGroup &g : groups)
        for (const auto &[info, unitOffset] : g.assignments)
          members.push_back(info);
      int64_t dynLbUnits = maxConcurrentLoad(members, [](const AllocInfo *i) {
        memref::AllocOp op = i->allocOp; // copy handle to drop constness
        return staticFactorBytes(op.getType());
      });
      int64_t dynFrag = singleF ? dynPoolUnits - dynLbUnits : 0;
      NumDynFragUnits += dynFrag;

      int64_t smallBins = 0, smallMinBins = 0;
      for (const DynBucket &bucket : dynPacking.smallBuckets) {
        members.clear();
        for (const auto &bin : bucket.bins)
          members.append(bin.begin(), bin.end());
        smallBins += static_cast<int64_t>(bucket.bins.size());
        smallMinBins += maxConcurrentLoad(
            members, [](const AllocInfo *) { return int64_t{1}; });
      }
      NumSmallBucketExcessBins += smallBins - smallMinBins;

      if (emitFragmentationReport) {
        InFlightDiagnostic remark = funcOp.emitRemark();
        remark << "hip-pool-allocs fragmentation: domain " << domainId
               << ": static " << staticPoolSize << "/" << staticLb << " B ("
               << staticFrag << " frag); ";
        if (singleF)
          remark << "dyn-pool " << dynPoolUnits << "/" << dynLbUnits
                 << " units (" << dynFrag << " frag); ";
        else
          remark << "dyn-pool mixed-F; ";
        remark << "small-buckets " << smallBins << "/" << smallMinBins
               << " bins (" << (smallBins - smallMinBins) << " excess)";
      }
    }

    domainPoolStaticSizes.push_back(staticPoolSize);

    if (!hasDynamic && staticPoolSize == 0)
      continue;

    // Phase 5: emit per-bucket sizes, the pool, offsets, and views — all
    // constrained to THIS domain's allocs as `requiredBefore`. Per-domain
    // `requiredBefore` is the key change vs. the pre-multi-domain code:
    // domain N's emission point need only dominate domain N's allocs, not
    // every pooled alloc in the function.
    SmallVector<Operation *> domainAllocOps;
    domainAllocOps.reserve(domain.allocs.size());
    for (AllocInfo *info : domain.allocs)
      domainAllocOps.push_back(info->allocOp.getOperation());

    // Move the builder to the latest legal point for `dynOperands`-derived
    // size arithmetic: strictly after every dyn-operand def, strictly before
    // every alloc in this domain. Shared by the aligned-group F emit and the
    // small-bucket size emit. Returns false (after emitting an error) when no
    // such point exists -- impossible after a successful Phase 1.5 partition,
    // whose feasibility check is a superset of this one.
    auto seekSizeInsertionPoint = [&](ArrayRef<Value> dynOperands) -> bool {
      SmallVector<Operation *> requiredAfter;
      for (Value dynOp : dynOperands)
        if (auto *def = dynOp.getDefiningOp())
          if (def->getBlock() == &block)
            requiredAfter.push_back(def);
      auto ip =
          findLatestLegalInsertionPoint(block, requiredAfter, domainAllocOps);
      if (!ip) {
        funcOp.emitError(
            "hip-pool-allocs: cannot place dynamic size arithmetic at a legal "
            "point in the block (a dyn-operand def is at-or-after the earliest "
            "pooled alloc in its domain)");
        return false;
      }
      builder.setInsertionPoint(&block, *ip);
      return true;
    };

    // Emit each aligned group's F = product(dynOperands).
    for (auto [gi, group] : llvm::enumerate(dynPacking.alignedGroups)) {
      if (!seekSizeInsertionPoint(group.dynOperands))
        return signalPassFailure();
      Value f = arith::ConstantIndexOp::create(builder, loc, 1);
      for (Value d : group.dynOperands)
        f = builder.createOrFold<arith::MulIOp>(loc, f, d);
      groupFactors[gi] = f;
    }

    // Emit each small bucket's aligned byte size.
    for (auto &bucket : dynPacking.smallBuckets) {
      if (!seekSizeInsertionPoint(bucket.dynOperands))
        return signalPassFailure();
      emitBucketSize(builder, loc, bucket, align);
    }

    // Pool acquisition insertion point: after every bucket's aligned size
    // SSA value, before every alloc IN THIS DOMAIN.
    {
      SmallVector<Operation *> requiredAfter;
      for (Value f : groupFactors)
        if (f)
          if (auto *def = f.getDefiningOp())
            if (def->getBlock() == &block)
              requiredAfter.push_back(def);
      for (auto &bucket : dynPacking.smallBuckets) {
        Value v =
            bucket.alignedSize ? bucket.alignedSize : bucket.byteSizeValue;
        if (!v)
          continue;
        if (auto *def = v.getDefiningOp())
          if (def->getBlock() == &block)
            requiredAfter.push_back(def);
      }
      auto ip =
          findLatestLegalInsertionPoint(block, requiredAfter, domainAllocOps);
      if (!ip) {
        funcOp.emitError(
            "hip-pool-allocs: cannot place hip.get_pool at a legal point "
            "(some dynamic size arithmetic is at-or-after an alloc in its "
            "domain)");
        return signalPassFailure();
      }
      builder.setInsertionPoint(&block, *ip);
    }

    // Per-group footprint = spanUnits * F; per-slab width = max of member
    // footprints (lifetime-disjoint groups share a slab). Emitted before
    // hip.get_pool so they dominate every alloc, and reused for the per-alloc
    // offsets below.
    SmallVector<Value> groupFootprint(dynPacking.alignedGroups.size());
    for (auto [gi, group] : llvm::enumerate(dynPacking.alignedGroups)) {
      Value spanVal =
          arith::ConstantIndexOp::create(builder, loc, group.spanUnits);
      groupFootprint[gi] =
          builder.createOrFold<arith::MulIOp>(loc, groupFactors[gi], spanVal);
    }
    SmallVector<Value> binWidth(dynPacking.alignedSuperBins.size());
    for (auto [bi, slab] : llvm::enumerate(dynPacking.alignedSuperBins)) {
      Value w = groupFootprint[slab[0]];
      // A slab holding a non-alignment-multiple alloc (possible only in
      // lifetime-only mode, which packs small allocs here too) rounds its width
      // up so the next slab's base stays aligned. Alignment-multiple-only slabs
      // skip the round-up (it is a no-op there).
      auto notAligned = [&](unsigned gi) {
        return dynPacking.alignedGroups[gi].spanUnits % align != 0;
      };
      bool needAlign = notAligned(slab[0]);
      for (unsigned gi : llvm::drop_begin(slab)) {
        w = builder.createOrFold<arith::MaxUIOp>(loc, w, groupFootprint[gi]);
        needAlign |= notAligned(gi);
      }
      if (needAlign)
        w = emitAlignUp(builder, loc, w, align);
      binWidth[bi] = w;
    }

    // Pool size = staticPoolSize
    //           + sum_slabs(max member footprint)
    //           + sum_smallBuckets(alignedSize * numBins).
    Value poolSize =
        arith::ConstantIndexOp::create(builder, loc, staticPoolSize);
    for (Value w : binWidth)
      poolSize = builder.createOrFold<arith::AddIOp>(loc, poolSize, w);
    for (auto &bucket : dynPacking.smallBuckets) {
      Value numBinsVal = arith::ConstantIndexOp::create(
          builder, loc, static_cast<int64_t>(bucket.bins.size()));
      Value contribution = builder.createOrFold<arith::MulIOp>(
          loc, bucket.alignedSize, numBinsVal);
      poolSize =
          builder.createOrFold<arith::AddIOp>(loc, poolSize, contribution);
    }

    // Tag the get_pool with this domain's id so the runtime grows the right
    // backing buffer. domain_id == 0 round-trips as the default attribute and
    // is elided by the printer, keeping single-domain output bit-identical to
    // the pre-multi-domain IR.
    Value pool = hip::GetPoolOp::create(
        builder, loc, poolType, ctx, poolSize,
        builder.getI64IntegerAttr(static_cast<int64_t>(domainId)));

    // Per-alloc offsets in this domain's pool. Static offsets are emitted
    // as constants right after the pool; dynamic offsets walk the buckets
    // with `currentBase` advancing past each so distinct buckets never
    // alias within this pool.
    DenseMap<Operation *, Value> allocToOffset;
    for (auto &[info, offset] : staticAssignments) {
      Value offsetVal = arith::ConstantIndexOp::create(builder, loc, offset);
      allocToOffset[info->allocOp.getOperation()] = offsetVal;
    }
    if (hasDynamic) {
      Value currentBase =
          arith::ConstantIndexOp::create(builder, loc, staticPoolSize);
      // Aligned slabs: groups sharing a slab (proven lifetime-disjoint) share
      // its base; within a group, alloc offset = base + unitOffset * F. The
      // base advances by the slab width (max member footprint), so disjoint
      // groups overlap in address space -- safe, they are never live together.
      for (auto [bi, slab] : llvm::enumerate(dynPacking.alignedSuperBins)) {
        for (unsigned gi : slab) {
          Value f = groupFactors[gi];
          for (auto &[info, unitOffset] :
               dynPacking.alignedGroups[gi].assignments) {
            Value off = currentBase;
            if (unitOffset != 0) {
              Value uo =
                  arith::ConstantIndexOp::create(builder, loc, unitOffset);
              Value scaled = builder.createOrFold<arith::MulIOp>(loc, f, uo);
              off =
                  builder.createOrFold<arith::AddIOp>(loc, currentBase, scaled);
            }
            allocToOffset[info->allocOp.getOperation()] = off;
          }
        }
        currentBase =
            builder.createOrFold<arith::AddIOp>(loc, currentBase, binWidth[bi]);
      }
      // Small buckets: bin offset = currentBase + binIdx * alignedSize.
      for (auto &bucket : dynPacking.smallBuckets) {
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

    LLVM_DEBUG(llvm::dbgs()
               << "  Pool[" << domainId << "]: static=" << staticPoolSize
               << " bytes, " << dynPacking.alignedGroups.size()
               << " aligned dyn groups, " << dynPacking.smallBuckets.size()
               << " small buckets, " << domain.allocs.size() << " allocs\n");

    // Replace this domain's allocs with views into THIS domain's pool.
    // Metadata (offset + domain id) is recorded BEFORE erasing the alloc
    // so the Operation* key into `allocToPos` stays valid.
    for (AllocInfo *info : domain.allocs) {
      auto it = allocToOffset.find(info->allocOp.getOperation());
      if (it == allocToOffset.end())
        continue;
      Value offset = it->second;

      auto posIt = allocToPos.find(info->allocOp.getOperation());
      assert(posIt != allocToPos.end() && "alloc missing from allocToPos");
      unsigned pos = posIt->second;
      bufferDomains[pos] = static_cast<unsigned>(domainId);
      if (auto constOp = offset.getDefiningOp<arith::ConstantIndexOp>())
        bufferOffsets[pos] = constOp.value();
      else
        bufferOffsets[pos] = -1;

      builder.setInsertionPoint(info->allocOp);
      MemRefType viewType = info->allocOp.getType();
      auto dynSizes = info->allocOp.getDynamicSizes();
      auto view =
          memref::ViewOp::create(builder, info->allocOp.getLoc(), viewType,
                                 pool, offset, SmallVector<Value>(dynSizes));
      info->allocOp.replaceAllUsesWith(view.getResult());
      info->allocOp.erase();
      ++NumAllocsPooled;
    }
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

  // ----- Module metadata for GenerateInterface -------------------------
  //
  // Layout:
  //
  //   Always emitted (legacy contract — single-domain models are
  //   bit-identical to pre-multi-domain output):
  //     hipdnn.pool_size      = i64           // domain 0's static prefix
  //     hipdnn.buffer_count   = i64           // total pooled allocs
  //     hipdnn.buffer_offsets = array<i64>    // -1 for dynamic offsets
  //
  //   Emitted only when domain_count > 1 (informational/code-generation
  //   metadata; runtime domain selection is carried by each lowered
  //   hip.get_pool call's domain_id and size operands):
  //     hipdnn.domain_count   = i64
  //     hipdnn.pool_sizes     = array<i64>    // per-domain static prefix
  //     hipdnn.buffer_domains = array<i64>    // per-buffer domain id
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  MLIRContext *mlirCtx = funcOp.getContext();
  int64_t legacyPoolSize =
      domainPoolStaticSizes.empty() ? 0 : domainPoolStaticSizes[0];
  moduleOp->setAttr("hipdnn.pool_size",
                    builder.getI64IntegerAttr(legacyPoolSize));
  moduleOp->setAttr("hipdnn.buffer_count",
                    builder.getI64IntegerAttr((int64_t)allInfos.size()));
  SmallVector<Attribute> offsetAttrs;
  for (int64_t off : bufferOffsets)
    offsetAttrs.push_back(builder.getI64IntegerAttr(off));
  moduleOp->setAttr("hipdnn.buffer_offsets",
                    ArrayAttr::get(mlirCtx, offsetAttrs));

  if (domains.size() > 1) {
    moduleOp->setAttr(
        "hipdnn.domain_count",
        builder.getI64IntegerAttr(static_cast<int64_t>(domains.size())));
    SmallVector<Attribute> poolSizeAttrs;
    for (int64_t s : domainPoolStaticSizes)
      poolSizeAttrs.push_back(builder.getI64IntegerAttr(s));
    moduleOp->setAttr("hipdnn.pool_sizes",
                      ArrayAttr::get(mlirCtx, poolSizeAttrs));
    SmallVector<Attribute> domainAttrs;
    for (unsigned d : bufferDomains)
      domainAttrs.push_back(builder.getI64IntegerAttr(static_cast<int64_t>(d)));
    moduleOp->setAttr("hipdnn.buffer_domains",
                      ArrayAttr::get(mlirCtx, domainAttrs));
  }
}

} // namespace
} // namespace hip
} // namespace mlir
