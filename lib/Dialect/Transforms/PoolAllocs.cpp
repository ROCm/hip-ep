/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PoolAllocs.cpp - Pack memref.alloc into a single i8 pool -----------===//
//
// Packs all memref.alloc ops in a single-block function into one contiguous
// byte pool (memref<Nxi8> for fully-static, memref<?xi8> when any alloc is
// dynamic), replacing each original alloc with a memref.view at a computed
// byte offset.
//
// Algorithm overview:
//   Phase 1 - Liveness:  assign [defIndex, lastUseIndex] to each alloc,
//             following view-like ops transitively.
//   Phase 2 - Partition:  split allocs into static (compile-time byte size)
//             and dynamic (runtime byte size) groups.
//   Phase 3 - Static packing:  greedy best-fit offset assignment inspired by
//             greedy best-fit strip packing.  Allocs whose
//             lifetimes don't overlap may share the same offset range.
//   Phase 4 - Dynamic packing:  group by structural byte-size key (same
//             static factor + same dynamic SSA operands).  Within each group,
//             bin non-overlapping lifetimes so they share an offset at runtime.
//   Phase 5 - IR emission:  create the pool memref.alloc, compute offsets via
//             arith ops, and replace each original alloc with memref.view.
//             Attach hipdnn.pool_size / hipdnn.buffer_offsets metadata.
//
// All sub-buffer offsets are aligned to kDefaultAlignment (256 bytes) to
// satisfy GPU coalesced-access requirements.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "llvm/Support/Debug.h"

#include <list>

#define DEBUG_TYPE "hip-pool-allocs"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_POOLALLOCSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// GPU-friendly alignment for sub-buffer byte offsets.
static constexpr int64_t kDefaultAlignment = 256;

//===----------------------------------------------------------------------===//
// AllocInfo - per-alloc metadata collected in Phase 1
//===----------------------------------------------------------------------===//

struct AllocInfo {
  memref::AllocOp allocOp;
  unsigned defIndex;       ///< sequential index of the alloc op in the block
  unsigned lastUseIndex;   ///< highest index of any (transitive) user
  int64_t staticByteSize;  ///< >0 for static shapes, 0 for dynamic
};

/// Total byte size for a fully-static memref type, 0 when any dim is dynamic.
static int64_t getStaticByteSize(MemRefType type) {
  if (!type.hasStaticShape())
    return 0;
  return type.getNumElements() * type.getElementTypeBitWidth() / 8;
}

/// True when \p user produces a memref that aliases its memref operand
/// (subview, cast, reshape, select, ...).
static bool isMemRefAlias(Operation* user) {
  return isa<ViewLikeOpInterface, arith::SelectOp>(user);
}

/// Walk all transitive users of \p value (through view-like and select ops)
/// and return the maximum operation index.  This extends an alloc's effective
/// lifetime to cover all derived views.
///
/// memref.dealloc ops are excluded so that lifetimes reflect actual data
/// usage, not administrative cleanup.
static unsigned findLastTransitiveUseIndex(
    Value value, Block& block,
    const DenseMap<Operation*, unsigned>& opIndex, unsigned blockSize) {
  unsigned lastIdx = 0;
  SmallVector<Value> worklist = {value};
  DenseSet<Value> visited;

  while (!worklist.empty()) {
    Value current = worklist.pop_back_val();
    if (!visited.insert(current).second)
      continue;

    for (Operation* user : current.getUsers()) {
      if (isa<memref::DeallocOp>(user))
        continue;

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

      // Follow alias chain so the source buffer stays live.
      if (isMemRefAlias(user)) {
        for (Value result : user->getResults())
          if (isa<MemRefType>(result.getType()))
            worklist.push_back(result);
      }
    }
  }
  return lastIdx;
}

static int64_t alignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
}

/// True when two allocs' [def, lastUse] intervals overlap.
static bool lifetimesOverlap(const AllocInfo& a, const AllocInfo& b) {
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
  AllocInfo* info;
  int64_t offset;
  int64_t size;  ///< aligned byte size
};

/// Assign byte offsets to static allocs using a greedy best-fit strategy.
///
/// For each alloc (processed largest-first), scan the existing reservations
/// whose lifetimes overlap, find the smallest gap that fits, and place it
/// there.  If no gap fits, append after the last overlapping reservation.
static SmallVector<std::pair<AllocInfo*, int64_t>>
packStaticAllocs(MutableArrayRef<AllocInfo> statics, int64_t alignment) {
  SmallVector<std::pair<AllocInfo*, int64_t>> assignments;
  std::list<Reservation> reservations;  // kept sorted by offset

  for (AllocInfo& info : statics) {
    int64_t size = alignUp(info.staticByteSize, alignment);
    int64_t bestOffset = -1;
    int64_t bestFit = INT64_MAX;

    // Walk existing reservations with overlapping lifetimes and look for
    // the smallest gap between them that fits this allocation.
    int64_t currentOffset = 0;
    for (auto& res : reservations) {
      if (!lifetimesOverlap(info, *res.info))
        continue;
      int64_t alignedOffset = alignUp(currentOffset, alignment);
      if (alignedOffset + size <= res.offset &&
          res.offset - alignedOffset < bestFit) {
        bestOffset = alignedOffset;
        bestFit = res.offset - alignedOffset;
      }
      currentOffset = std::max(currentOffset, res.offset + res.size);
    }

    // No gap found - append after all overlapping reservations.
    if (bestOffset < 0)
      bestOffset = alignUp(currentOffset, alignment);

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
//
// Example (dynamic attention model, 6 allocs after optimize-memrefs):
//
//   Alloc    Type                Operands              Key
//   -------  ------------------  --------------------  ----------------------
//   Q        memref<?x?x64xf32>  (%dim, %dim_0)       {256, [%dim, %dim_0]}
//   K        memref<?x?x64xf32>  (%dim, %dim_0)       {256, [%dim, %dim_0]}
//   V        memref<?x?x64xf32>  (%dim, %dim_0)       {256, [%dim, %dim_0]}
//   KT       memref<?x64x?xf32>  (%dim, %dim_0)       {256, [%dim, %dim_0]}
//   scores   memref<?x?x?xf32>   (%dim,%dim_0,%dim_0) {4, [%dim,%dim_0,%dim_0]}
//   scaled   memref<?x?x?xf32>   (%dim,%dim_0,%dim_0) {4, [%dim,%dim_0,%dim_0]}
//
//   Note: ?x?x64xf32 and ?x64x?xf32 have different shapes but the same
//   total byte size (4 * 64 * dim * dim_0 = 256 * dim * dim_0).
//
//   Bucket A  (byte_size = 256 * dim * dim_0):  Q, K, V, KT
//     All lifetimes overlap -> 4 bins, one alloc each.
//   Bucket B  (byte_size = 4 * dim * dim_0^2):  scores, scaled
//     Lifetimes overlap -> 2 bins.
//
//   Pool layout at runtime:
//     |--Q--|--K--|--V--|--KT--|-scores-|-scaled-|
//      bin0   bin1  bin2  bin3    bin0     bin1
//      <- 4 x alignUp(bucketA) -> <- 2 x alignUp(bucketB) ->
//
//   pool_size = 4*alignUp(bucketA) + 2*alignUp(bucketB)

/// A bucket groups dynamic allocs that share the same runtime byte size.
/// Within a bucket, each "bin" holds allocs with non-overlapping lifetimes
/// that can share a single offset at runtime.
struct DynBucket {
  Value byteSizeValue;                        ///< SSA value for the byte size
  SmallVector<SmallVector<AllocInfo*>> bins;  ///< bins of non-overlapping allocs
};

/// Structural key for grouping allocs with identical runtime byte sizes.
///
/// Two allocs match if they have the same compile-time factor (product of
/// element bytes and static dimensions) and the same SSA operands for
/// dynamic dimensions.
///
/// Example: memref<?x?x64xf32> allocated with (%dim, %dim_0)
///   staticFactor = 4 (f32 bytes) * 64 (static dim) = 256
///   dynOperands  = [%dim, %dim_0]
///   runtime byte size = 256 * %dim * %dim_0
struct DynSizeKey {
  int64_t staticFactor;
  SmallVector<Value, 2> dynOperands;

  bool operator==(const DynSizeKey& other) const {
    return staticFactor == other.staticFactor &&
           dynOperands == other.dynOperands;
  }
};

/// Group dynamic allocs into buckets and bins.
///
/// 1. Compute a structural DynSizeKey for each alloc.
/// 2. Deduplicate keys and emit one byte-size SSA value per unique key
///    (arith.constant * arith.muli chain) at the builder's current position.
/// 3. Group allocs by byte-size SSA value.
/// 4. Within each group, first-fit bin allocs with non-overlapping lifetimes.
static SmallVector<DynBucket>
packDynamicAllocs(MutableArrayRef<AllocInfo> dynamics, OpBuilder& builder,
                  Location loc) {
  struct KeyedInfo {
    DynSizeKey key;
    AllocInfo* info;
  };

  // Step 1: build structural keys.
  SmallVector<KeyedInfo> keyed;
  for (AllocInfo& info : dynamics) {
    MemRefType type = info.allocOp.getType();
    auto dynSizes = info.allocOp.getDynamicSizes();
    int64_t elemBytes = type.getElementTypeBitWidth() / 8;
    int64_t staticFactor = elemBytes;
    for (int64_t dim : type.getShape())
      if (!ShapedType::isDynamic(dim))
        staticFactor *= dim;
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
  auto findOrCreateByteSize = [&](const DynSizeKey& key) -> Value {
    for (auto& [k, v] : uniqueKeys)
      if (k == key)
        return v;
    // byte_size = staticFactor * dynDim0 * dynDim1 * ...
    Value byteSize;
    if (key.staticFactor != 1)
      byteSize =
          arith::ConstantIndexOp::create(builder, loc, key.staticFactor);
    for (Value dynDim : key.dynOperands) {
      if (byteSize)
        byteSize = arith::MulIOp::create(builder, loc, byteSize, dynDim);
      else
        byteSize = dynDim;
    }
    if (!byteSize)
      byteSize =
          arith::ConstantIndexOp::create(builder, loc, key.staticFactor);
    uniqueKeys.push_back({key, byteSize});
    return byteSize;
  };

  // Step 3: group by byte-size SSA value.
  llvm::MapVector<Value, SmallVector<AllocInfo*>> bySize;
  for (auto& [key, info] : keyed)
    bySize[findOrCreateByteSize(key)].push_back(info);

  // Step 4: first-fit bin packing within each group.
  SmallVector<DynBucket> buckets;
  for (auto& [sizeVal, infos] : bySize) {
    DynBucket bucket;
    bucket.byteSizeValue = sizeVal;
    for (AllocInfo* info : infos) {
      bool placed = false;
      for (auto& bin : bucket.bins) {
        bool conflicts = false;
        for (AllocInfo* existing : bin) {
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
  void runOnOperation() override;
};

void PoolAllocsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.empty())
    return;
  assert(funcOp.getBody().hasOneBlock() &&
         "hip-pool-allocs requires single-block functions; liveness analysis "
         "uses sequential op indices that do not generalize to control flow");

  Block& block = funcOp.getBody().front();

  // ----- Phase 1: liveness analysis ------------------------------------
  //
  // Assign each op a sequential index and compute [defIndex, lastUseIndex]
  // for every memref.alloc.  lastUseIndex is the maximum index among all
  // *transitive* users: if %buf feeds memref.subview -> op X, then %buf
  // must stay live until X, not just until the subview.

  DenseMap<Operation*, unsigned> opIndex;
  unsigned idx = 0;
  for (Operation& op : block)
    opIndex[&op] = idx++;
  unsigned blockSize = idx;

  SmallVector<AllocInfo> allInfos;
  for (Operation& op : block) {
    auto allocOp = dyn_cast<memref::AllocOp>(op);
    if (!allocOp)
      continue;
    Value result = allocOp.getResult();
    if (result.use_empty())
      continue;
    AllocInfo info;
    info.allocOp = allocOp;
    info.defIndex = opIndex[&op];
    info.lastUseIndex =
        findLastTransitiveUseIndex(result, block, opIndex, blockSize);
    info.staticByteSize = getStaticByteSize(allocOp.getType());
    allInfos.push_back(info);
  }

  // Need at least 2 allocs to pool.
  if (allInfos.size() < 2)
    return;

  // ----- Phase 2: partition into static / dynamic ----------------------
  //
  // Static allocs (all dims known) go through the greedy offset assignment
  // in Phase 3.  Dynamic allocs go through the bucket/bin packing in Phase 4.

  SmallVector<AllocInfo> statics, dynamics;
  for (auto& info : allInfos) {
    if (info.staticByteSize > 0)
      statics.push_back(info);
    else
      dynamics.push_back(info);
  }

  // Largest-first ordering improves packing density:  big allocs are harder
  // to fit into gaps, so placing them first gives smaller allocs more
  // opportunities to fill the remaining holes.
  llvm::sort(statics, [](const AllocInfo& a, const AllocInfo& b) {
    return a.staticByteSize > b.staticByteSize;
  });

  // ----- Phase 3: pack statics ----------------------------------------

  auto staticAssignments = packStaticAllocs(statics, kDefaultAlignment);

  int64_t staticPoolSize = 0;
  for (auto& [info, offset] : staticAssignments) {
    int64_t end = offset + alignUp(info->staticByteSize, kDefaultAlignment);
    staticPoolSize = std::max(staticPoolSize, end);
  }

  // ----- Phase 4: pack dynamics ----------------------------------------

  Location loc = funcOp.getLoc();
  OpBuilder builder(funcOp.getContext());

  // Insert all pool IR right before the earliest alloc being pooled so
  // that downstream passes (e.g. hip-lower-allocs) see the pool after
  // any prerequisite ops like hip.create_handle but before any consumer
  // of the pooled buffers.
  Operation* firstPooledAlloc = allInfos.front().allocOp.getOperation();
  for (auto& info : allInfos)
    if (info.allocOp->isBeforeInBlock(firstPooledAlloc))
      firstPooledAlloc = info.allocOp.getOperation();
  builder.setInsertionPoint(firstPooledAlloc);

  auto dynBuckets = packDynamicAllocs(dynamics, builder, loc);

  // ----- Phase 5: emit pool + views -----------------------------------
  //
  // Build a single pool memref and replace every original alloc with a
  // memref.view into it.
  //
  // Emitted IR for a fully-static case (4 allocs totalling 131072 bytes):
  //
  //   %pool = memref.alloc() : memref<131072xi8>
  //   %off0 = arith.constant 0     : index
  //   %v0   = memref.view %pool[%off0][] : memref<131072xi8> to memref<...>
  //   %off1 = arith.constant 32768 : index
  //   %v1   = memref.view %pool[%off1][] : memref<131072xi8> to memref<...>
  //   ...
  //
  // For a mixed static+dynamic case the pool becomes memref<?xi8>:
  //
  //   %static_part = arith.constant 32768 : index
  //   %aligned     = <alignUp(%dynBucketSize, 256)>  // arith add/div/mul
  //   %pool_size   = arith.addi %static_part, %aligned : index
  //   %pool        = memref.alloc(%pool_size) : memref<?xi8>
  //   ...

  bool hasDynamic = !dynBuckets.empty();

  // 5a. Compute total pool size as an SSA value.
  Value poolSize;
  if (hasDynamic) {
    poolSize = arith::ConstantIndexOp::create(builder, loc, staticPoolSize);
    Value alignConst =
        arith::ConstantIndexOp::create(builder, loc, kDefaultAlignment);

    for (auto& bucket : dynBuckets) {
      int64_t numBins = bucket.bins.size();
      // alignUp(bucketSize, alignment) via arith ops.
      Value alignM1 = arith::ConstantIndexOp::create(
          builder, loc, kDefaultAlignment - 1);
      Value sum =
          arith::AddIOp::create(builder, loc, bucket.byteSizeValue, alignM1);
      Value divided = arith::DivUIOp::create(builder, loc, sum, alignConst);
      Value alignedBucketSize =
          arith::MulIOp::create(builder, loc, divided, alignConst);

      if (numBins > 1) {
        Value numBinsVal =
            arith::ConstantIndexOp::create(builder, loc, numBins);
        Value contribution =
            arith::MulIOp::create(builder, loc, alignedBucketSize, numBinsVal);
        poolSize = arith::AddIOp::create(builder, loc, poolSize, contribution);
      } else {
        poolSize =
            arith::AddIOp::create(builder, loc, poolSize, alignedBucketSize);
      }
    }
  } else {
    if (staticPoolSize == 0)
      return;
  }

  // 5b. Create the single pool allocation.
  Value pool;
  if (hasDynamic) {
    auto poolType =
        MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(8));
    pool =
        memref::AllocOp::create(builder, loc, poolType, ValueRange{poolSize});
  } else {
    auto poolType =
        MemRefType::get({staticPoolSize}, builder.getIntegerType(8));
    pool = memref::AllocOp::create(builder, loc, poolType);
  }

  // 5c. Compute byte offsets for every alloc and store in allocToOffset.
  DenseMap<Operation*, Value> allocToOffset;

  for (auto& [info, offset] : staticAssignments) {
    Value offsetVal = arith::ConstantIndexOp::create(builder, loc, offset);
    allocToOffset[info->allocOp.getOperation()] = offsetVal;
  }

  if (hasDynamic) {
    Value currentBase =
        arith::ConstantIndexOp::create(builder, loc, staticPoolSize);
    Value alignConst2 =
        arith::ConstantIndexOp::create(builder, loc, kDefaultAlignment);
    Value alignM1 =
        arith::ConstantIndexOp::create(builder, loc, kDefaultAlignment - 1);

    for (auto& bucket : dynBuckets) {
      Value sum = arith::AddIOp::create(builder, loc, bucket.byteSizeValue,
                                        alignM1);
      Value divided = arith::DivUIOp::create(builder, loc, sum, alignConst2);
      Value alignedBucketSize =
          arith::MulIOp::create(builder, loc, divided, alignConst2);

      for (int64_t binIdx = 0;
           binIdx < static_cast<int64_t>(bucket.bins.size()); ++binIdx) {
        // offset = currentBase + binIdx * alignedBucketSize
        Value binOffset;
        if (binIdx == 0) {
          binOffset = currentBase;
        } else {
          Value binIdxVal =
              arith::ConstantIndexOp::create(builder, loc, binIdx);
          Value binContrib =
              arith::MulIOp::create(builder, loc, alignedBucketSize, binIdxVal);
          binOffset =
              arith::AddIOp::create(builder, loc, currentBase, binContrib);
        }
        for (AllocInfo* info : bucket.bins[binIdx])
          allocToOffset[info->allocOp.getOperation()] = binOffset;
      }

      // Advance base past this entire bucket.
      int64_t numBins = bucket.bins.size();
      if (numBins > 1) {
        Value numBinsVal =
            arith::ConstantIndexOp::create(builder, loc, numBins);
        Value bucketTotal =
            arith::MulIOp::create(builder, loc, alignedBucketSize, numBinsVal);
        currentBase =
            arith::AddIOp::create(builder, loc, currentBase, bucketTotal);
      } else {
        currentBase = arith::AddIOp::create(builder, loc, currentBase,
                                            alignedBucketSize);
      }
    }
  }

  // 5d. Replace each original alloc with a memref.view into the pool.
  bool hadDeallocs = false;
  SmallVector<int64_t> staticOffsets;
  for (auto& info : allInfos) {
    auto it = allocToOffset.find(info.allocOp.getOperation());
    if (it == allocToOffset.end())
      continue;

    Value offset = it->second;
    builder.setInsertionPoint(info.allocOp);

    MemRefType viewType = info.allocOp.getType();
    auto dynSizes = info.allocOp.getDynamicSizes();
    auto view = memref::ViewOp::create(builder, info.allocOp.getLoc(), viewType,
                                       pool, offset,
                                       SmallVector<Value>(dynSizes));

    info.allocOp.replaceAllUsesWith(view.getResult());
    info.allocOp.erase();

    // Record offsets for the hipdnn.buffer_offsets attribute.
    // Dynamic offsets are represented as -1 since the actual value
    // is only known at runtime.
    if (auto constOp = offset.getDefiningOp<arith::ConstantIndexOp>())
      staticOffsets.push_back(constOp.value());
    else
      staticOffsets.push_back(-1);
  }

  // 5d'. Erase deallocs that now target views (invalid to free a view)
  // and insert a single dealloc for the pool buffer.  After RAUW above,
  // any `memref.dealloc %alloc_X` became `memref.dealloc %view_X`.
  SmallVector<memref::DeallocOp> orphanedDeallocs;
  funcOp.walk([&](memref::DeallocOp op) {
    if (op.getMemref().getDefiningOp<memref::ViewOp>())
      orphanedDeallocs.push_back(op);
  });
  hadDeallocs = !orphanedDeallocs.empty();
  for (auto op : orphanedDeallocs)
    op.erase();

  if (hadDeallocs) {
    builder.setInsertionPoint(block.getTerminator());
    memref::DeallocOp::create(builder, loc, pool);
  }

  // 5e. Attach pool metadata to the function for runtime/deployment use.
  MLIRContext* ctx = funcOp.getContext();
  if (!hasDynamic)
    funcOp->setAttr("hipdnn.pool_size",
                    builder.getI64IntegerAttr(staticPoolSize));
  SmallVector<Attribute> offsetAttrs;
  for (int64_t off : staticOffsets)
    offsetAttrs.push_back(builder.getI64IntegerAttr(off));
  funcOp->setAttr("hipdnn.buffer_offsets", ArrayAttr::get(ctx, offsetAttrs));
}

}  // namespace
}  // namespace hip
}  // namespace mlir
