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
#include "mlir/Interfaces/SideEffectInterfaces.h"

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

  // No allocs: nothing to pool.
  // Previously we also short-circuited at `< 2`, but that left single
  // allocs to flow through `LowerAllocs` -> `hip.alloc` -> `MemoryLowering`
  // which lowers to a global `hip_device_malloc` symbol that the runtime
  // does not export. We pool any allocs we see, even if just one, so the
  // dyn-pool path is the only allocation path the compiled model exercises.
  if (allInfos.empty()) {
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

  Operation *firstPooledAlloc = allInfos.front().allocOp.getOperation();
  for (auto &info : allInfos)
    if (info.allocOp->isBeforeInBlock(firstPooledAlloc))
      firstPooledAlloc = info.allocOp.getOperation();

  // Partition dynamics into "poolable" and "unpoolable" based on whether
  // every transitive dependency of each dyn-size operand is hoistable
  // above `firstPooledAlloc`.
  //
  // An op is hoistable when it is (a) a function argument, (b) already
  // before `firstPooledAlloc`, or (c) side-effect-free AND every operand
  // is itself hoistable. Hoistability is recursive because the
  // byte-size computation we emit below (`staticFactor * dynDim0 * ...`)
  // is inserted at `firstPooledAlloc`'s position and so requires every
  // input to dominate it. Side-effectful ops like `memref.load` cannot
  // be moved without risking aliasing with an intervening store.
  //
  // Concretely: bufferization may build a dyn-alloc size from a
  // `memref.load` of a shape buffer that another op (e.g. `hip.shape`)
  // writes into earlier in the function. Pooling such an alloc would
  // require moving the `memref.load` above the `hip.shape` write —
  // impossible without redoing memory analysis. We sidestep the issue
  // by leaving those allocs unpooled; `LowerAllocs` then converts each
  // to `hip.alloc` + `hip.free`. Statically-sized allocs and dynamic
  // allocs whose sizes derive purely from args/constants are still
  // pooled.
  //
  // We must hoist hoistable defining ops (e.g. `memref.dim %arg, %c0`
  // that bufferization places right next to its consumer alloc) to
  // before `firstPooledAlloc` so the byte-size computation that
  // consumes them dominates `hip.get_pool`.
  SmallVector<AllocInfo> poolableDynamics;
  SmallVector<AllocInfo> unpoolableDynamics;
  {
    DenseMap<Operation *, bool> hoistableCache;
    std::function<bool(Operation *)> isHoistable = [&](Operation *def) -> bool {
      if (!def)
        return true; // block argument
      if (def->getBlock() != &block)
        return true; // outside this block — dominates
      if (def->isBeforeInBlock(firstPooledAlloc))
        return true; // already dominates
      auto it = hoistableCache.find(def);
      if (it != hoistableCache.end())
        return it->second;
      // Mark as visited (assume false to handle any cycles defensively;
      // SSA in a single block shouldn't have cycles but be safe).
      hoistableCache[def] = false;
      if (!isPure(def))
        return false;
      for (Value v : def->getOperands()) {
        if (!isHoistable(v.getDefiningOp()))
          return false;
      }
      hoistableCache[def] = true;
      return true;
    };

    for (AllocInfo &info : dynamics) {
      bool allHoistable = true;
      for (Value dynOperand : info.allocOp.getDynamicSizes()) {
        if (!isHoistable(dynOperand.getDefiningOp())) {
          allHoistable = false;
          break;
        }
      }
      if (allHoistable)
        poolableDynamics.push_back(info);
      else
        unpoolableDynamics.push_back(info);
    }

    // Hoist every cached-as-hoistable op that's still after
    // `firstPooledAlloc` to dominate it. Iterate to a fixed point so
    // dependencies move before users (small worklist in practice).
    bool changed = true;
    while (changed) {
      changed = false;
      for (auto &[def, hoistable] : hoistableCache) {
        if (!hoistable)
          continue;
        if (def->isBeforeInBlock(firstPooledAlloc))
          continue;
        bool allOperandsDominate = true;
        for (Value v : def->getOperands()) {
          Operation *vDef = v.getDefiningOp();
          if (!vDef)
            continue;
          if (vDef->getBlock() != &block)
            continue;
          if (!vDef->isBeforeInBlock(firstPooledAlloc)) {
            allOperandsDominate = false;
            break;
          }
        }
        if (allOperandsDominate) {
          def->moveBefore(firstPooledAlloc);
          changed = true;
        }
      }
    }
  }

  builder.setInsertionPoint(firstPooledAlloc);

  auto dynBuckets = packDynamicAllocs(poolableDynamics, builder, loc, align);
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
  // `buffer_count` is the number of allocs actually pooled (== size of
  // `staticOffsets`), NOT `allInfos.size()`. When some dynamic allocs
  // are left unpooled (their dyn-size operands depend on non-hoistable
  // ops; see Phase 4 partitioning) `buffer_count` < `allInfos.size()`.
  // `GenerateInterface` indexes `buffer_offsets[0..buffer_count)` so the
  // two sizes must agree.
  ModuleOp moduleOp = funcOp->getParentOfType<ModuleOp>();
  MLIRContext *mlirCtx = funcOp.getContext();
  moduleOp->setAttr("hipdnn.pool_size",
                    builder.getI64IntegerAttr(staticPoolSize));
  moduleOp->setAttr("hipdnn.buffer_count",
                    builder.getI64IntegerAttr((int64_t)staticOffsets.size()));
  SmallVector<Attribute> offsetAttrs;
  for (int64_t off : staticOffsets)
    offsetAttrs.push_back(builder.getI64IntegerAttr(off));
  moduleOp->setAttr("hipdnn.buffer_offsets",
                    ArrayAttr::get(mlirCtx, offsetAttrs));
}

} // namespace
} // namespace hip
} // namespace mlir
