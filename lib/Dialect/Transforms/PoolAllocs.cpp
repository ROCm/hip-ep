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
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

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
  // memref.cast preserves dim values (verifier requires shape compatibility);
  // forward the query to the source so type-erasing casts don't terminate the
  // chain prematurely. Common pattern: a `memref.view` with known shape gets
  // cast to `memref<?x?x?xf16>` before being fed into a `memref.reshape` or
  // similar — without this passthrough we'd emit `memref.dim` on the cast,
  // and PoolAllocs's hoist walker leaves the cast in place since it's not in
  // the hoistable set, producing a dominance failure.
  if (auto castOp = dyn_cast_or_null<memref::CastOp>(def))
    return resolveDimAtSource(b, loc, castOp.getSource(), i);
  // memref.view dim_i is either the matching static result-type dim or the
  // corresponding entry in the view's dynamic-size operand list. Same shape
  // as the AllocOp case below.
  if (auto viewOp = dyn_cast_or_null<memref::ViewOp>(def)) {
    auto vt = cast<MemRefType>(viewOp.getResult().getType());
    if (i < 0 || i >= vt.getRank())
      return Value();
    if (!vt.isDynamicDim(i))
      return arith::ConstantIndexOp::create(b, loc, vt.getDimSize(i));
    int64_t dynIdx = 0;
    for (int64_t j : llvm::seq<int64_t>(0, i))
      if (vt.isDynamicDim(j))
        ++dynIdx;
    auto sizes = viewOp.getSizes();
    if (dynIdx < static_cast<int64_t>(sizes.size()))
      return sizes[dynIdx];
  }
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
  // memref.dim(memref.cast(src), i) -> memref.dim(src, i). Casts only change
  // the layout / dynamic-dim annotations; the runtime dim values are
  // unchanged. Recurse so subview/expand/etc. underneath get resolved too.
  if (auto castOp = dyn_cast_or_null<memref::CastOp>(def))
    return resolveDimAtSource(b, loc, castOp.getSource(), i);

  // memref.dim(memref.subview(parent, ...), i) -> i-th `size` operand of the
  // subview. Static sizes become constants; dynamic sizes are the SSA values
  // the subview was already sized by, which are hoistable through the rest of
  // the pool prelude (typically themselves memref.dim / arith.* / index_cast).
  //
  // This avoids the dominance trap of emitting a fresh `memref.dim` on the
  // subview itself, because the subview op is created in the middle of the
  // function body and isn't reachable through Phase-4 hoisting.
  //
  // Before / After (canonical case: packed-QKV split via subview):
  //   Before:
  //     %qkv  = memref.alloc(%d0, %d1) : memref<?x?x8192xf16>
  //     %k    = memref.subview %qkv[0,0,2048] [%d0,%d1,2048] [1,1,1]
  //               : memref<?x?x8192xf16>
  //               to   memref<?x?x2048xf16, strided<[?, 8192, 1], offset:
  //               2048>>
  //     %dim0 = memref.dim %k, %c0   // !! dim-of-subview, not hoistable
  //   After:
  //     %qkv  = memref.alloc(%d0, %d1) : memref<?x?x8192xf16>
  //     %k    = memref.subview ...                // unchanged (dead-arith DCE
  //                                               // may remove it later)
  //     %dim0 = %d0                              // direct use of operand
  if (auto subview = dyn_cast_or_null<memref::SubViewOp>(def)) {
    auto resultType = cast<MemRefType>(src.getType());
    if (i < 0 || i >= resultType.getRank())
      return Value();
    if (!resultType.isDynamicDim(i))
      return arith::ConstantIndexOp::create(b, loc, resultType.getDimSize(i));
    auto mixedSizes = subview.getMixedSizes();
    // memref.subview's `sizes` are 1:1 with the SOURCE rank (not result),
    // when rank-reduction is allowed. SubViewOp::getMixedSizes() returns
    // entries in source-rank order; the result rank is built by skipping
    // size-1 dims dropped by rank reduction.  Walk source dims and skip the
    // dropped ones to find the source dim that maps to result dim `i`.
    auto droppedDims = subview.getDroppedDims();
    int64_t resultDim = -1;
    for (int64_t srcDim = 0, e = static_cast<int64_t>(mixedSizes.size());
         srcDim < e; ++srcDim) {
      if (droppedDims.test(srcDim))
        continue;
      ++resultDim;
      if (resultDim != i)
        continue;
      OpFoldResult size = mixedSizes[srcDim];
      if (auto attr = dyn_cast<Attribute>(size)) {
        if (auto intAttr = dyn_cast<IntegerAttr>(attr))
          return arith::ConstantIndexOp::create(b, loc, intAttr.getInt());
        return Value();
      }
      return cast<Value>(size);
    }
    return Value();
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
  } else if (auto reshape = dyn_cast_or_null<memref::ReshapeOp>(def)) {
    // memref.reshape carries a runtime shape operand (memref<Nxi64>), but its
    // result type captures every dim that's statically known. For a static
    // output dim, return the constant directly. For the single dynamic output
    // dim (the common case for ONNX Reshape with one -1 entry), recover it
    // from the source memref's total element count divided by the static
    // product of the other output dims — same trick as the expand_shape
    // dynamic-dim case above, but generalised across arbitrary input ranks.
    //
    // Required because the tensor.reshape fallback in ReshapeConversion
    // bufferizes to memref.reshape, which is not in PoolAllocs's hoistable
    // set. Without this fold, `memref.dim(%reshape, i)` would be hoisted
    // (DimOp is hoistable) above its source (reshape isn't) → SSA dominance
    // failure in Phase 4.
    //
    // Before:
    //   %r = memref.reshape %src(%shape) : (memref<?x?x?xf16>, memref<4xi64>)
    //                                       -> memref<1x?x16x72xf16>
    //   %d = memref.dim %r, %c1 : memref<1x?x16x72xf16>
    //
    // After (input has 3 dyn dims d0,d1,d2; output static product = 1152):
    //   %d0 = memref.dim %src, %c0
    //   %d1 = memref.dim %src, %c1
    //   %d2 = memref.dim %src, %c2
    //   %t  = arith.muli %d0, %d1
    //   %t2 = arith.muli %t, %d2
    //   %k  = arith.constant 1152 : index
    //   %d  = arith.divui %t2, %k : index
    auto resultType = cast<MemRefType>(reshape.getResult().getType());
    if (i < 0 || i >= resultType.getRank())
      return Value();
    if (!resultType.isDynamicDim(i))
      return arith::ConstantIndexOp::create(b, loc, resultType.getDimSize(i));
    int64_t staticProduct = 1;
    int64_t dynCount = 0;
    for (int64_t j : llvm::seq<int64_t>(0, resultType.getRank())) {
      if (resultType.isDynamicDim(j))
        ++dynCount;
      else
        staticProduct *= resultType.getDimSize(j);
    }
    // Only the single-dynamic-dim case is structurally derivable; multi-dyn
    // outputs would need the runtime shape buffer (not hoistable).
    if (dynCount == 1) {
      Value srcMem = reshape.getSource();
      auto srcType = cast<MemRefType>(srcMem.getType());
      Value total;
      bool ok = true;
      for (int64_t j : llvm::seq<int64_t>(0, srcType.getRank())) {
        Value d = resolveDimAtSource(b, loc, srcMem, j);
        if (!d) {
          ok = false;
          break;
        }
        total = total ? b.createOrFold<arith::MulIOp>(loc, total, d) : d;
      }
      if (ok && total) {
        if (staticProduct == 1)
          return total;
        Value div = arith::ConstantIndexOp::create(b, loc, staticProduct);
        return b.createOrFold<arith::DivUIOp>(loc, total, div);
      }
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

/// Return true when `op` writes to its `outs` operand `m` and the
/// computation is a pure elementwise binary op on scalar memrefs that we can
/// promote to `arith` SSA. Sets `lhs`/`rhs`/`kind` on success.
enum class ScalarBinaryKind { None, Mul, Add, Sub, Div };
static ScalarBinaryKind classifyScalarBinaryWriter(Operation *op, Value m,
                                                   Value &lhs, Value &rhs) {
  if (auto mul = dyn_cast<MulOp>(op)) {
    if (mul.getOutput() != m)
      return ScalarBinaryKind::None;
    lhs = mul.getLhs();
    rhs = mul.getRhs();
    return ScalarBinaryKind::Mul;
  }
  if (auto add = dyn_cast<AddOp>(op)) {
    if (add.getOutput() != m)
      return ScalarBinaryKind::None;
    lhs = add.getLhs();
    rhs = add.getRhs();
    return ScalarBinaryKind::Add;
  }
  if (auto sub = dyn_cast<SubOp>(op)) {
    if (sub.getOutput() != m)
      return ScalarBinaryKind::None;
    lhs = sub.getLhs();
    rhs = sub.getRhs();
    return ScalarBinaryKind::Sub;
  }
  if (auto div = dyn_cast<DivOp>(op)) {
    if (div.getOutput() != m)
      return ScalarBinaryKind::None;
    lhs = div.getLhs();
    rhs = div.getRhs();
    return ScalarBinaryKind::Div;
  }
  return ScalarBinaryKind::None;
}

/// True if `m` is a scalar-shaped memref: rank-0, or rank-1 with static
/// size 1.
static bool isScalarShapeMemref(Value m) {
  auto ty = dyn_cast<MemRefType>(m.getType());
  if (!ty)
    return false;
  if (ty.getRank() == 0)
    return true;
  if (ty.getRank() == 1 && !ty.isDynamicDim(0) && ty.getDimSize(0) == 1)
    return true;
  return false;
}

/// True if `indices` are all constant zero (scalar slot access).
static bool indicesAreZero(ValueRange indices) {
  for (Value idx : indices) {
    auto cst = getConstantIntValue(idx);
    if (!cst || *cst != 0)
      return false;
  }
  return true;
}

/// True if `op` writes to a memref that may alias `target` (intervening
/// store-or-write between a forward-fold candidate's writer and the load).
/// Uses BufferOriginAnalysis to detect aliases through view chains
/// (memref.view / reinterpret_cast / cast / subview / expand_shape /
/// collapse_shape).
static bool opMayWriteToAlias(Operation *op, Value target,
                              BufferOriginAnalysis &origin) {
  // Cheap pre-filter: ops with no memref operands cannot write to memrefs.
  bool hasMemrefOperand = false;
  for (Value operand : op->getOperands())
    if (isa<MemRefType>(operand.getType())) {
      hasMemrefOperand = true;
      break;
    }
  if (!hasMemrefOperand)
    return false;

  // Helper: a memref operand may write to target if it might be the same
  // allocation (returns nullopt or true).
  auto mayAlias = [&](Value v) {
    std::optional<bool> same = origin.isSameAllocation(v, target);
    // Conservative: unknown (nullopt) -> treat as aliasing.
    return !same.has_value() || *same;
  };

  // memref.store writes to its memref operand.  Use STRICT SSA-equality
  // (not BufferOriginAnalysis::isSameAllocation) here: in the canonical
  // host-scratch pattern produced by `--hip-materialize-host-scalars` —
  //
  //   %0 = hip.get_host_scratch(...) : memref<?xi8>
  //   %view   = memref.view %0[%c0]   : memref<?xi8> to memref<i64>  ;; slot 0
  //   (batch) memref.store %batch, %view[] %view_2 = memref.view %0[%c64]  :
  //   memref<?xi8> to memref<i64>  ;; slot 1 (seq) memref.store %seq, %view_2[]
  //   %view_3 = memref.view %0[%c128] : memref<?xi8> to memref<i64>  ;; slot 2
  //   (mul) hip.mul ins(%view, %view_2) outs(%view_3)
  //
  // BufferOriginAnalysis says all three views share the same root
  // allocation (%0), so under `mayAlias` semantics a `store %seq, %view_2`
  // would be classified as "may write to alias of %view" — and the latest
  // such writer (the seq-store at offset 64) would shadow the batch-store
  // at offset 0.  `materializeScalarFromMemref(%view)` then folds %view's
  // scalar to %seq instead of %batch, and `hip.mul`'s output materializes
  // as `arith.muli %seq, %seq` instead of `arith.muli %batch, %seq` —
  // silently corrupting the dynamic Range/Reshape alloc size on every
  // asymmetric (batch != seq) shape.
  //
  // Two views with different offsets are distinct scalar slots even though
  // they share a root buffer; SSA-equality on the store target is the
  // correct discrimination (canonicalize/CSE keeps `memref.view %0[%c64]`
  // unique per offset).
  if (auto store = dyn_cast<memref::StoreOp>(op))
    return store.getMemRef() == target;

  // hip dialect DPS ops write to their `output` operand (last memref operand
  // by convention; matches the Hip_DpsOp layout used across the dialect).
  // Reads (ins) don't matter for forward-fold correctness.
  if (op->getDialect() && op->getDialect()->getNamespace() == "hip") {
    Value last;
    for (Value operand : op->getOperands())
      if (isa<MemRefType>(operand.getType()))
        last = operand;
    if (last)
      return mayAlias(last);
    return false;
  }

  // memref.copy / dealloc are conservatively treated as writes.
  if (isa<memref::CopyOp, memref::DeallocOp>(op)) {
    for (Value operand : op->getOperands())
      if (isa<MemRefType>(operand.getType()) && mayAlias(operand))
        return true;
    return false;
  }

  // memref.load / memref.dim / view-likes only read.
  if (isa<memref::LoadOp, memref::DimOp, memref::ViewOp,
          memref::ReinterpretCastOp, memref::CastOp, memref::CollapseShapeOp,
          memref::ExpandShapeOp, memref::SubViewOp>(op))
    return false;

  // Conservatively treat unknown ops with memref operands as potential writers.
  return true;
}

/// Walk the block from start to (but excluding) `useOp` and return the most
/// recent op that writes to a memref aliased with `target`.  This is the
/// classical "immediately preceding store/write" pattern but generalised
/// across aliasing view-like ops (memref.view / reinterpret_cast / cast /
/// subview / collapse_shape / expand_shape) via BufferOriginAnalysis.
///
/// Multiple-writer scratch slots are routine in this pipeline: a single
/// scratch byte slot (e.g. host-scratch offset 64 emitted by
/// `--hip-materialize-host-scalars`) is overwritten once per Range / Expand /
/// etc.  The relevant writer for a given load is the one immediately before
/// it in topological (block) order.
static Operation *
findImmediatePrecedingAliasingWriter(Operation *useOp, Value target,
                                     BufferOriginAnalysis &origin) {
  Block *block = useOp->getBlock();
  Operation *latest = nullptr;
  for (Operation &op : *block) {
    if (&op == useOp)
      break;
    if (opMayWriteToAlias(&op, target, origin))
      latest = &op;
  }
  return latest;
}

/// Recursive scalar materializer.  Walks back from `m` (a scalar-shape
/// memref<T>) through the immediately-preceding writer to produce a pure
/// SSA scalar value that, when bufferization unwinds, equals the value
/// stored at index 0 of `m` at `useOp`'s position.  Returns `Value()` when
/// traceback fails (no preceding writer, non-promotable writer, or
/// recursive call fails).
///
/// Handled writers:
///   * `memref.store %v, %m[%c0]`            -> %v
///   * `hip.mul ins(%a, %b) outs(%m)`        -> arith.muli(scalar(%a),
///                                                         scalar(%b))
///   * `hip.add/sub/div` similar             -> arith.{addi, subi, divsi}
///
/// "Immediately preceding" semantics matter because the canonical IR has
/// many writers to the same scratch slot (e.g. one `hip.mul` per Range op
/// overwriting the same offset-64 view of host scratch).  Each `memref.load`
/// reads what was written by the most-recent preceding hip op, and that
/// chain is exactly what hoisting needs to ascend through.
static Value materializeScalarFromMemref(Value m, Operation *useOp,
                                         BufferOriginAnalysis &origin,
                                         OpBuilder &builder, Location loc) {
  if (!isScalarShapeMemref(m))
    return Value();

  Operation *writer = findImmediatePrecedingAliasingWriter(useOp, m, origin);
  if (!writer)
    return Value();

  if (auto store = dyn_cast<memref::StoreOp>(writer)) {
    if (!indicesAreZero(store.getIndices()))
      return Value();
    return store.getValueToStore();
  }

  Value lhsMem, rhsMem;
  ScalarBinaryKind kind = classifyScalarBinaryWriter(writer, m, lhsMem, rhsMem);
  // The writer might be classified but with a different output operand (e.g.
  // it writes to an alias of m, not m itself).  classifyScalarBinaryWriter
  // requires output == m exactly.  When the writer aliases m via
  // reinterpret_cast / subview chains, we conservatively bail — the simpler
  // case (same SSA value) is what we care about here.
  if (kind == ScalarBinaryKind::None)
    return Value();
  Value lhsScalar =
      materializeScalarFromMemref(lhsMem, writer, origin, builder, loc);
  if (!lhsScalar)
    return Value();
  Value rhsScalar =
      materializeScalarFromMemref(rhsMem, writer, origin, builder, loc);
  if (!rhsScalar)
    return Value();
  if (lhsScalar.getType() != rhsScalar.getType())
    return Value();
  if (!isa<IntegerType>(lhsScalar.getType()))
    return Value(); // Only promote integer-typed scalar arith.
  switch (kind) {
  case ScalarBinaryKind::Mul:
    return arith::MulIOp::create(builder, loc, lhsScalar, rhsScalar)
        .getResult();
  case ScalarBinaryKind::Add:
    return arith::AddIOp::create(builder, loc, lhsScalar, rhsScalar)
        .getResult();
  case ScalarBinaryKind::Sub:
    return arith::SubIOp::create(builder, loc, lhsScalar, rhsScalar)
        .getResult();
  case ScalarBinaryKind::Div:
    return arith::DivSIOp::create(builder, loc, lhsScalar, rhsScalar)
        .getResult();
  case ScalarBinaryKind::None:
    return Value();
  }
  return Value();
}

/// Forward-fold `memref.load %m[%c0]` (where `%m` is a scalar-shape memref
/// fed by a single `memref.store` or hip elementwise DPS op) into pure SSA
/// arith on the underlying scalars, so `hip-pool-allocs` can hoist dynamic
/// alloc sizes through the count chain.
///
/// Motivating IR (after `--hip-materialize-host-scalars` redirects scalar
/// `tensor.from_elements` allocations to a host-mapped scratch buffer):
///
///   Before:
///     %712 = arith.index_cast %dim_275 : index to i64
///     memref.store %712, %reinterpret_cast[%c0]
///     %713 = arith.index_cast %dim_276 : index to i64
///     memref.store %713, %view[%c0]
///     hip.mul ins(%reinterpret_cast, %view) outs(%view_278)
///     %715 = memref.load %view_278[%c0]   // <-- not hoistable
///     ...
///     %733 = arith.index_cast %732 : i64 to index
///     %alloc = memref.alloc(%733) : memref<?xi64>
///
///   After (this pass):
///     %712 = arith.index_cast %dim_275 : index to i64
///     %713 = arith.index_cast %dim_276 : index to i64
///     %715 = arith.muli %712, %713 : i64  // <-- promoted, hoistable
///     ...
///     %733 = arith.index_cast %732 : i64 to index
///     %alloc = memref.alloc(%733) : memref<?xi64>
///
/// Dead `memref.store` / `hip.mul` ops remain in the IR — they have memref
/// side effects so we don't DCE them here; downstream passes (canonicalize)
/// remove them once nothing reads the scratch slot.  Their cost is one
/// scratch write per inference, negligible relative to GPU work.
static void foldScalarMemrefArith(func::FuncOp funcOp,
                                  BufferOriginAnalysis &origin) {
  bool changed = true;
  while (changed) {
    changed = false;
    SmallVector<memref::LoadOp> loads;
    funcOp.walk([&](memref::LoadOp op) {
      if (!isScalarShapeMemref(op.getMemRef()))
        return;
      if (!indicesAreZero(op.getIndices()))
        return;
      loads.push_back(op);
    });
    for (memref::LoadOp load : loads) {
      OpBuilder builder(load);
      Value scalar = materializeScalarFromMemref(load.getMemRef(), load, origin,
                                                 builder, load.getLoc());
      if (!scalar)
        continue;
      if (scalar.getType() != load.getType())
        continue;
      load.replaceAllUsesWith(scalar);
      load.erase();
      changed = true;
    }
  }
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
    // Worklist mirrors the producers handled in resolveDimAtSource. Anything
    // that can be folded to either a hoistable scalar (constant / arith) or
    // to a dim-of-root must be listed here, otherwise the unhandled
    // dim-of-non-alloc survives into Phase-4 hoisting and triggers
    // "operand #0 does not dominate this use" when the producer is later in
    // the function body than the hoist target (typical of memref.subview /
    // memref.cast chains created by packed-QKV split or strided-output IR).
    if (def &&
        (isa<memref::AllocOp>(def) || isa<memref::CollapseShapeOp>(def) ||
         isa<memref::ExpandShapeOp>(def) || isa<memref::ReshapeOp>(def) ||
         isa<memref::CastOp>(def) || isa<memref::ViewOp>(def) ||
         isa<memref::SubViewOp>(def)))
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

  // TODO: Generalize to multi-block functions using MLIR's Liveness analysis
  // instead of sequential op indices.
  if (!funcOp.getBody().hasOneBlock()) {
    funcOp.emitError("hip-pool-allocs requires single-block functions; "
                     "liveness analysis uses sequential op indices that do "
                     "not generalize to control flow");
    return signalPassFailure();
  }

  // Pre-pass: forward-fold scalar `memref.load`s into pure SSA arith so the
  // hoist worklist below ascends through `arith.muli/cmpi/select/ceildivsi`
  // instead of stopping at `memref.load %scratch[%c0]` (which would leave
  // the dynamic-size alloc referencing values defined later than the pool
  // prelude — "operand #0 does not dominate this use" in Phase 4).
  // Order matters: this runs BEFORE foldDimOfReshape because some shape
  // arithmetic chains feed through both reshape ops AND scratch slot loads.
  // BufferOriginAnalysis is used to detect aliasing writes through view /
  // reinterpret_cast / cast chains (multiple SSA reinterpret_casts of one
  // rank-0 scratch alloc would otherwise be treated as independent slots).
  {
    BufferOriginAnalysis origin(funcOp);
    foldScalarMemrefArith(funcOp, origin);
  }

  BufferViewFlowAnalysis aliasAnalysis(funcOp);

  // Pre-pass: simplify `memref.dim` of `memref.collapse_shape`/
  // `memref.expand_shape` so the hoist worklist below can ascend through to
  // the original source memref. Without this, a surviving dim-of-collapse
  // breaks Phase 4's SSA dominance for any same-rank dynamic Reshape
  // decomposed into expand_shape + collapse_shape.
  foldDimOfReshape(funcOp);

  Block &block = funcOp.getBody().front();

  // ----- Phase 1: liveness analysis ------------------------------------

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
  // Hoistable: `memref.dim` (special-cased — we want to chase dim queries up
  // to a hoistable root) plus any side-effect-free op in the `arith` dialect.
  // Restricting to `arith` keeps the whitelist tight (no `hip.*` / `memref.*`
  // ops with hidden ordering constraints) while covering every shape-arithmetic
  // pattern the bufferizer is allowed to emit (index/integer math, clamps,
  // boolean combinators, FP shape math from Cast→Floor→Ceil chains, etc.).
  // The MemoryEffectFree check is the same purity guarantee we relied on for
  // the original narrow whitelist — broadened so we don't need to chase every
  // new arith op kind that ONNX lowerings happen to produce.
  //
  // Also explicitly allow `memref.get_global` (pure symbol lookup) and
  // `memref.load` of a session-immutable global (`memref.global "..." constant`
  // or one carrying the `hip.external_data` attr). These two are not in the
  // arith dialect and would not pass the MemoryEffectFree check on their own
  // (LoadOp has a Read effect) — kept as a defensive allowance for converters
  // that legitimately need to read a 1-element external constant at compile
  // time (e.g. ConstantOfShape's dynamic path on a shape tensor that survived
  // conversion). Range itself materialises its start/limit/delta as pure-SSA
  // `arith.constant` via the hybrid two-phase constant externalisation in
  // ConvertOnnxToHipPass, so this branch doesn't fire for it.
  auto isHoistable = [](Operation *op) {
    if (isa<memref::DimOp>(op))
      return true;
    if (op->getDialect() && op->getDialect()->getNamespace() ==
                                arith::ArithDialect::getDialectNamespace())
      return isMemoryEffectFree(op);
    if (isa<memref::GetGlobalOp>(op))
      return true;
    if (auto load = dyn_cast<memref::LoadOp>(op)) {
      auto getGlobal = load.getMemRef().getDefiningOp<memref::GetGlobalOp>();
      if (!getGlobal)
        return false;
      auto symbol = SymbolTable::lookupNearestSymbolFrom(
          getGlobal, getGlobal.getNameAttr());
      auto globalOp = dyn_cast_or_null<memref::GlobalOp>(symbol);
      if (!globalOp)
        return false;
      return globalOp.getConstant() || globalOp->hasAttr("hip.external_data");
    }
    return false;
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
  // Transitive hoistability: an op is safe to hoist only if (a) it is locally
  // hoistable AND (b) every same-block operand def is either already before
  // firstPooledAlloc (no move needed) or itself transitively hoistable. If
  // any operand traces back to a non-hoistable op in the same block (e.g. a
  // `memref.load` or `hip.*` whose result feeds shape arithmetic), hoisting
  // the consumer would leave the producer behind and break SSA dominance.
  //
  // Without this filter, a broader `isHoistable` whitelist (e.g. all pure
  // `arith` ops) pulls in chains like `arith.cmpf(memref.load(...), ...)`
  // through reachable shape arithmetic — the cmpf is locally hoistable but
  // its load operand is not, so a naive hoist creates a use-before-def.
  Block *funcBlock = firstPooledAlloc->getBlock();
  llvm::DenseMap<Operation *, bool> canHoistMemo;
  std::function<bool(Operation *)> canHoist = [&](Operation *op) -> bool {
    if (!op)
      return true; // block arg / no def
    if (op->getBlock() != funcBlock)
      return true; // dominates from outside the block
    if (op->isBeforeInBlock(firstPooledAlloc))
      return true; // already in the prelude region, no move needed
    auto it = canHoistMemo.find(op);
    if (it != canHoistMemo.end())
      return it->second;
    // Mark in-progress as false to break cycles conservatively.
    canHoistMemo[op] = false;
    if (!isHoistable(op))
      return false;
    for (Value operand : op->getOperands())
      if (!canHoist(operand.getDefiningOp()))
        return false;
    canHoistMemo[op] = true;
    return true;
  };

  // Filter out dynamic allocs whose dyn_size chains include same-block ops
  // that can't be hoisted (e.g. shape arithmetic that depends on a runtime
  // `memref.load` of a value computed by a `hip.*` op earlier in the block).
  // Such allocs cannot be pooled: even after the hoist, Phase 5's pool-offset
  // arithmetic would reference SSA values defined after `firstPooledAlloc`,
  // violating dominance. Leave them as plain `memref.alloc` so the subsequent
  // `hip-lower-allocs` pass turns them into individual `hip.alloc`/`hip.free`
  // calls — slower than pooling but functionally correct.
  SmallVector<AllocInfo> poolableDynamics;
  SmallVector<memref::AllocOp> droppedAllocs;
  for (auto &info : dynamics) {
    bool allOk = true;
    for (Value dyn : info.allocOp.getDynamicSizes())
      if (Operation *defOp = dyn.getDefiningOp())
        if (!canHoist(defOp)) {
          allOk = false;
          break;
        }
    if (allOk)
      poolableDynamics.push_back(info);
    else
      droppedAllocs.push_back(info.allocOp);
  }
  dynamics = std::move(poolableDynamics);
  // Rebuild allInfos to drop the un-poolable allocs so Phase 5's `for (auto &
  // info : allInfos)` loop doesn't try to replace them with views.
  if (!droppedAllocs.empty()) {
    llvm::DenseSet<Operation *> dropped;
    for (auto a : droppedAllocs)
      dropped.insert(a.getOperation());
    SmallVector<AllocInfo> kept;
    for (auto &info : allInfos)
      if (!dropped.count(info.allocOp.getOperation()))
        kept.push_back(info);
    allInfos = std::move(kept);
    // Recompute firstPooledAlloc — the original may have been a dropped alloc.
    if (!allInfos.empty()) {
      firstPooledAlloc = allInfos.front().allocOp.getOperation();
      for (auto &info : allInfos)
        if (info.allocOp->isBeforeInBlock(firstPooledAlloc))
          firstPooledAlloc = info.allocOp.getOperation();
    }
  }

  llvm::DenseSet<Operation *> hoistSet;
  for (Operation *op : hoistWorklist)
    if (canHoist(op))
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
