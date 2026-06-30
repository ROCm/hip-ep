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
//     Phase 4 - Dynamic packing: bucket by structural byte-size key + bin
//               by lifetime within this domain.
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
/// well-canonicalised graph produces 1 domain (the typical case after upstream
/// hoisting) or 2 (when a host-load-dependent alloc forces a second domain).
/// A count far above that usually means upstream hoisting is missing or some
/// size arithmetic that `--hip-hoist-alloc-size-arith` should have lifted is
/// still pinned below its allocs — so we emit a non-fatal remark to flag the
/// likely pre-condition gap, then proceed normally.
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
    // Defensive: only device-space allocations belong in the GPU pool.
    // Absorbing any explicit non-device space (host/pinned/managed) would
    // lower to the undefined hip_device_malloc. (Unspecified / legacy integer
    // spaces are transitional and still pooled.) Non-device transfer buffers
    // are stack allocas today, so none should reach here; skip any that slip
    // through.
    if (auto sp = dyn_cast_or_null<MemorySpaceAttr>(
            allocOp.getType().getMemorySpace()))
      if (sp.getKind() != MemorySpaceKind::Device)
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

  // The domain count is unbounded — the runtime grows its per-domain pool
  // arrays on demand — so a high count is correct, just suspicious. Emit a
  // non-fatal remark (compilation continues) pointing at the likely missing
  // upstream pre-condition, rather than failing the pass.
  if (domains.size() > kDomainCountAdvisoryThreshold) {
    funcOp.emitRemark("hip-pool-allocs: dominance partition produced ")
        << domains.size() << " domains (advisory threshold "
        << kDomainCountAdvisoryThreshold
        << "); this usually means upstream hoisting "
           "(--hip-hoist-alloc-size-arith) is missing or canonicalization left "
           "unhoistable size arithmetic in place. Compilation continues.";
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
    auto dynBuckets = packDynamicAllocs(dynamics);
    NumDynBuckets += dynBuckets.size();
    bool hasDynamic = !dynBuckets.empty();

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

    for (auto &bucket : dynBuckets) {
      SmallVector<Operation *> requiredAfter;
      for (Value dynOp : bucket.dynOperands)
        if (auto *def = dynOp.getDefiningOp())
          if (def->getBlock() == &block)
            requiredAfter.push_back(def);
      auto ip =
          findLatestLegalInsertionPoint(block, requiredAfter, domainAllocOps);
      if (!ip) {
        // Should be impossible after a successful Phase 1.5 partition:
        // partitionByDominanceDomain only forms a domain when its full
        // {requiredAfter, requiredBefore} union admits a strict insertion
        // point, and a single bucket's constraints are a subset of the
        // domain's. Surface anyway in case a future change diverges the
        // two feasibility checks.
        funcOp.emitError(
            "hip-pool-allocs: cannot place bucket size arithmetic at a "
            "legal point in the block (a dyn-operand def is at-or-after "
            "the earliest pooled alloc in its domain)");
        return signalPassFailure();
      }
      builder.setInsertionPoint(&block, *ip);
      emitBucketSize(builder, loc, bucket, align);
    }

    // Pool acquisition insertion point: after every bucket's aligned size
    // SSA value, before every alloc IN THIS DOMAIN.
    {
      SmallVector<Operation *> requiredAfter;
      for (auto &bucket : dynBuckets) {
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
            "(some bucket size arithmetic is at-or-after an alloc in its "
            "domain)");
        return signalPassFailure();
      }
      builder.setInsertionPoint(&block, *ip);
    }

    // Pool size = staticPoolSize + sum_buckets(alignedSize * numBins).
    Value poolSize =
        arith::ConstantIndexOp::create(builder, loc, staticPoolSize);
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

    LLVM_DEBUG(llvm::dbgs()
               << "  Pool[" << domainId << "]: static=" << staticPoolSize
               << " bytes, " << dynBuckets.size() << " dynamic buckets, "
               << domain.allocs.size() << " allocs\n");

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
  //   Emitted only when domain_count > 1 (consumed by the multi-domain
  //   runtime — older runtimes predating the multi-domain ABI silently ignore
  //   the extra attrs and would alias all domains onto a single pool, which is
  //   incorrect; the compiler/runtime ABI bump fixes that):
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
