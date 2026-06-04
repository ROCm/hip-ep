/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MaterializeHostScalars.cpp - host-mapped scratch for tiny scalars --===//
//
// Walks `memref.alloc` ops in the function and redirects ones that look like
// host-fed scalar staging buffers -- small static-shape, integer-or-index
// memrefs that have at least one host I/O user (memref.store / load) --
// away from the GPU pool.  Each candidate is replaced by a `memref.view`
// over a single per-function `hip.get_host_scratch(%ctx, %total)` buffer
// (host-mapped via `hipHostMalloc(hipHostMallocMapped)`, GPU-readable at
// the same VA on UMA targets, runtime-owned, grow-on-demand).
//
// Why this pass exists
// --------------------
// On targets where the GPU pool is real device memory, the bufferized
// `tensor.from_elements` lowering emits `memref.alloc + memref.store`
// which, once absorbed by `hip-pool-allocs`, becomes a host store into
// device memory and SEGVs.  Other targets silently worked because hipMalloc
// returned UMA-mapped host memory there, masking the bug.  This pass runs
// BEFORE `hip-pool-allocs` so candidates never enter the GPU pool.
//
// Example IR (canonical seqlens_k pattern, edited for brevity)
// ------------------------------------------------------------
// Before (single-element host-fed scalar consumed by a hip op):
//
//   %c0_i64 = arith.constant 0 : i64
//   %alloc  = memref.alloc() : memref<i64>           // <-- candidate
//   memref.store %c0_i64, %alloc[] : memref<i64>     // host store
//   %seqlens_k = hip.cast %alloc                     // hip consumer
//                : memref<i64> to memref<1xi32>
//
// After (alloc replaced by a view into the per-function host scratch buffer):
//
//   %total   = arith.constant 8 : index              // 1 elem * 8 bytes (i64)
//   %scratch = hip.get_host_scratch(%ctx, %total) : memref<?xi8>
//   %c0      = arith.constant 0 : index
//   %view    = memref.view %scratch[%c0][] : memref<?xi8> to memref<i64>
//   %c0_i64  = arith.constant 0 : i64
//   memref.store %c0_i64, %view[] : memref<i64>      // host store, into
//                                                    // host-mapped memory
//   %seqlens_k = hip.cast %view                      // GPU still reads at
//                : memref<i64> to memref<1xi32>      // the same VA (UMA)
//
// Multiple candidates in one function share ONE `hip.get_host_scratch`
// emitted at the entry block; each candidate gets its own 64-byte-aligned
// offset.  See test/lit/Dialect/hip-materialize-host-scalars.mlir for the
// full set of accepted/rejected shapes.
//
// Peeking through view ops (host scalar reached via a descriptor edit)
// --------------------------------------------------------------------
// Bufferization + CSE often fuse two `tensor.from_elements` shape-arith
// buffers into one alloc and `memref.reinterpret_cast` it for the second
// (smaller) use, so the host store reaches its hip consumer through a view:
//
//   %alloc = memref.alloc() : memref<3xi64>          // <-- candidate
//   memref.store %d0, %alloc[%c0] : memref<3xi64>    // host store (direct)
//   hip.expand ins(%expand, %alloc) ...              // hip consumer (direct)
//   %rc = memref.reinterpret_cast %alloc to          // view of the alloc
//           offset: [0], sizes: [1], strides: [1]
//           : memref<3xi64> to memref<1xi64>
//   memref.store %n, %rc[%c0] : memref<1xi64>        // host store (via view)
//   hip.slice ins(..., %rc) ...                      // hip consumer (via view)
//
// `classifyHostScalarUsers` recurses through the reinterpret_cast -- which
// touches no memory of its own -- finds only host-I/O and hip users at the
// leaves, and accepts the alloc.  Before this peek-through, the lone
// reinterpret_cast user rejected the alloc, so it stayed in the GPU pool and
// the host store SEGV'd on targets where the pool is real device memory.
//
// Hip-dialect users are accepted, not rejected
// --------------------------------------------
// `hipHostMalloc(hipHostMallocMapped)` returns a host pointer that is also
// GPU-accessible at the same virtual address on UMA targets, so the bare-ptr
// ABI used by `--convert-hip-to-llvm` consumes the same buffer regardless of
// whether it was hipMalloc'd or hipHostMalloc'd.  The canonical
// `tensor.from_elements -> reduce_sum + sub + cast -> seqlens_k` GQA pattern
// produces exactly an alloc with a host `memref.store` followed by a
// `hip.cast` that reads it; an earlier design rejected such hip consumers and
// left the host store crashing inside the GPU pool.
//
// Non-goals
// ---------
//   - Memrefs larger than 16 elements: the size cap keeps host scratch
//     bounded and matches the seqlens_k / shape-arith patterns observed
//     in practice.  Larger buffers stay in the GPU pool where they belong.
//   - Floating-point element types: rare on the host-fed scalar path,
//     almost always GPU-consumed in flight, where the GPU pool is the
//     right home.
//   - Functions whose arg 0 is not `!hip.context`: silently skipped.
//     Utility functions and pre-context-arg passes don't have access to
//     the runtime scratch handle; the pass is a best-effort mitigation,
//     not a correctness requirement on every function.
//   - Cross-function scratch coalescing: each function gets its own
//     `hip.get_host_scratch` allocation.  The runtime pool is
//     grow-on-demand and amortizes over the model's lifetime;
//     cross-function pooling would need a different mechanism.
//   - Pipeline placement after `hip-pool-allocs`: this pass MUST run
//     before `hip-pool-allocs`, otherwise candidates have already been
//     rewritten as views into the GPU pool and the host-mapping rewrite
//     is too late to help.  Ordering is enforced in `Pipelines.cpp` and
//     locked down by a LIT regression test.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <iterator>

#define DEBUG_TYPE "hip-materialize-host-scalars"

STATISTIC(NumAllocsMaterialized,
          "Number of memref.alloc redirected to host-mapped scratch");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_MATERIALIZEHOSTSCALARSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Round \p x up to a multiple of \p align (align must be a power of two).
static int64_t roundUp(int64_t x, int64_t align) {
  return (x + align - 1) & ~(align - 1);
}

/// Classify every transitive user of \p memrefVal and decide whether they are
/// all compatible with the backing buffer living in host-mapped scratch (see
/// the file header for the rationale behind each category). Sets \p sawHostIO
/// when a `memref.store`/`load` is found anywhere in the chain. Returns false
/// at the first user we cannot vouch for -- an op of an unknown dialect, or a
/// view whose own users escape.
///
/// Pure view/descriptor ops (reinterpret_cast, expand/collapse_shape, subview,
/// view, cast) touch no memory themselves, so the backing buffer is still a
/// host-staged scalar; we recurse into the view's result rather than bail.
static bool classifyHostScalarUsers(Value memrefVal, bool &sawHostIO) {
  for (Operation *user : memrefVal.getUsers()) {
    if (isa<memref::StoreOp, memref::LoadOp>(user)) {
      sawHostIO = true;
      continue;
    }
    if (isa<memref::DimOp, memref::DeallocOp>(user))
      continue;
    if (user->getDialect() && user->getDialect()->getNamespace() == "hip")
      continue;
    if (isa<memref::ReinterpretCastOp, memref::ExpandShapeOp,
            memref::CollapseShapeOp, memref::SubViewOp, memref::ViewOp,
            memref::CastOp>(user)) {
      if (!classifyHostScalarUsers(user->getResult(0), sawHostIO))
        return false;
      continue;
    }
    return false;
  }
  return true;
}

/// Direct `memref.store` ops whose memref operand is exactly \p allocResult,
/// in program order. Used to detect (and split) the scalar-reuse hazard.
static SmallVector<memref::StoreOp> directStores(Value allocResult) {
  SmallVector<memref::StoreOp> stores;
  for (Operation *user : allocResult.getUsers())
    if (auto st = dyn_cast<memref::StoreOp>(user))
      if (st.getMemref() == allocResult)
        stores.push_back(st);
  llvm::sort(stores, [](memref::StoreOp a, memref::StoreOp b) {
    return a->isBeforeInBlock(b);
  });
  return stores;
}

/// Number of distinct host-scratch slots a candidate needs.
///
/// The default is 1 (the canonical single-store seqlens_k pattern). A
/// single-element scalar that is written by MORE THAN ONE store is a buffer
/// reused across logically-independent scalars (an upstream pass coalesced two
/// `tensor.from_elements` buffers into one). If an earlier store's value is
/// consumed by an ASYNC `hip.*` op and a later store overwrites the same slot
/// before that op executes on the stream, the GPU reads the wrong value. To
/// make every store collision-free we hand each store its own slot. Restricted
/// to single-element buffers whose users all live in the alloc's own block, so
/// the program-order epoch walk in the rewrite is well-defined (multi-element
/// array-fill buffers — e.g. a 3xi64 shape vector filled element by element —
/// are NOT reuse and keep one slot).
static int64_t numScratchSlotsFor(memref::AllocOp allocOp) {
  if (allocOp.getType().getNumElements() != 1)
    return 1;
  Value res = allocOp.getResult();
  Block *block = allocOp->getBlock();
  for (Operation *user : res.getUsers())
    if (user->getBlock() != block)
      return 1; // cross-block use: program-order epochs ill-defined → 1 slot
  int64_t numStores = static_cast<int64_t>(directStores(res).size());
  return numStores > 1 ? numStores : 1;
}

/// True if \p allocOp is a tiny host-fed scalar staging buffer: a static,
/// small (<= 16 elements), integer-or-index memref with at least one host-I/O
/// user (possibly reached through view ops) whose entire transitive user set
/// is host I/O, metadata, or hip consumers. See classifyHostScalarUsers and
/// the file header for why each constraint exists.
static bool isHostScalarCandidate(memref::AllocOp allocOp) {
  MemRefType type = allocOp.getType();
  if (!type.hasStaticShape())
    return false;
  if (type.getNumElements() > 16)
    return false;
  Type elem = type.getElementType();
  if (!elem.isIntOrIndex())
    return false;

  bool hasHostIO = false;
  if (!classifyHostScalarUsers(allocOp.getResult(), hasHostIO))
    return false;
  return hasHostIO;
}

struct MaterializeHostScalarsPass
    : public impl::MaterializeHostScalarsPassBase<MaterializeHostScalarsPass> {

  void runOnOperation() override;
};

void MaterializeHostScalarsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  // Need !hip.context as arg 0 to call hip.get_host_scratch. If absent (e.g.
  // utility funcs or pre-context-arg passes), silently skip — we're a
  // best-effort mitigation, not a correctness requirement on every function.
  if (funcOp.getNumArguments() == 0 ||
      !isa<ContextType>(funcOp.getArgument(0).getType()))
    return;
  Value ctx = funcOp.getArgument(0);

  // Collect candidates.  The walk order here determines the per-candidate
  // scratch offsets computed below; any refactor that changes the traversal
  // (sorting, reverse-walk, parallel collection) changes the scratch layout
  // and therefore the LIT IR snapshots.  Keep `funcOp.walk` (post-order,
  // deterministic).
  SmallVector<memref::AllocOp> candidates;
  funcOp.walk([&](memref::AllocOp op) {
    if (isHostScalarCandidate(op))
      candidates.push_back(op);
  });
  if (candidates.empty())
    return;

  // Compute byte size per candidate and aligned base offsets. 64-byte
  // alignment matches the runtime pool and is comfortably above the largest
  // scalar alignment; gives every candidate (and every per-store slot) its own
  // cache line. A candidate may need MORE THAN ONE slot — see
  // numScratchSlotsFor for the scalar-reuse-hazard split.
  constexpr int64_t kAlign = 64;
  SmallVector<int64_t> baseOffsets; // first slot offset per candidate
  SmallVector<int64_t> slotStrides; // bytes between consecutive slots
  SmallVector<int64_t> slotCounts;  // number of slots per candidate
  baseOffsets.reserve(candidates.size());
  slotStrides.reserve(candidates.size());
  slotCounts.reserve(candidates.size());
  int64_t total = 0;
  for (memref::AllocOp allocOp : candidates) {
    MemRefType ty = allocOp.getType();
    Type elemTy = ty.getElementType();
    unsigned bits = elemTy.isIndex() ? 64 : elemTy.getIntOrFloatBitWidth();
    int64_t elemBytes = static_cast<int64_t>((bits + 7) / 8);
    int64_t bytes = ty.getNumElements() * elemBytes;
    if (bytes == 0)
      bytes = 1; // rank-0 with i1 etc. — never zero-sized
    int64_t nSlots = numScratchSlotsFor(allocOp);
    int64_t stride = roundUp(bytes, kAlign); // each slot on its own cache line
    int64_t base = roundUp(total, kAlign);
    baseOffsets.push_back(base);
    slotStrides.push_back(stride);
    slotCounts.push_back(nSlots);
    total = base + nSlots * stride;
  }
  total = roundUp(total, kAlign);

  // Emit hip.get_host_scratch(%ctx, %total) at function entry.
  Block &entry = funcOp.getBody().front();
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&entry);
  Location loc = funcOp.getLoc();
  Value totalSize = arith::ConstantIndexOp::create(builder, loc, total);
  auto scratchType =
      MemRefType::get({ShapedType::kDynamic}, builder.getIntegerType(8));
  Value scratch =
      GetHostScratchOp::create(builder, loc, scratchType, ctx, totalSize);

  // Replace each candidate with memref.view(s) over the scratch buffer.
  // Deallocs of the candidate are erased — the scratch buffer is owned by
  // the runtime (released in hipdnn_ep_state_cleanup), like the pool.
  for (auto [allocOp, base, stride, nSlots] :
       llvm::zip(candidates, baseOffsets, slotStrides, slotCounts)) {
    builder.setInsertionPoint(allocOp);

    // Materialize one view per slot at the alloc site (so each view dominates
    // every use that follows the alloc in the block).
    SmallVector<Value> views;
    views.reserve(nSlots);
    for (int64_t s : llvm::seq<int64_t>(nSlots)) {
      Value offsetVal = arith::ConstantIndexOp::create(
          builder, allocOp.getLoc(), base + s * stride);
      views.push_back(memref::ViewOp::create(builder, allocOp.getLoc(),
                                             allocOp.getType(), scratch,
                                             offsetVal,
                                             /*sizes=*/ValueRange{})
                          .getResult());
    }

    // Erase deallocs first (they reference the alloc's result).
    SmallVector<memref::DeallocOp> deallocs;
    for (Operation *user : allocOp->getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        deallocs.push_back(d);
    for (auto d : deallocs)
      d.erase();

    Value allocResult = allocOp.getResult();
    if (nSlots == 1) {
      // Common case: one slot, byte-identical to the original behaviour.
      allocResult.replaceAllUsesWith(views[0]);
    } else {
      // Scalar-reuse split: each `memref.store` opens a new epoch bound to its
      // own slot; every other use binds to the most recent epoch's slot. The
      // alloc's block is straight-line (cross-block users were excluded by
      // numScratchSlotsFor), so program order == epoch order and a later host
      // store can no longer clobber a slot still pending an async hip read.
      //
      // Before (one slot, %s reused — async hip.cast clobbered by 2nd store):
      //   %v  = view %scr[0]
      //   store %a, %v ; hip.cast %v -> ... ; store %b, %v ; readback %v
      // After (one slot per store):
      //   %v0 = view %scr[0] ; %v1 = view %scr[64]
      //   store %a, %v0 ; hip.cast %v0 -> ... ; store %b, %v1 ; readback %v1
      int64_t epoch = -1;
      for (Operation &op : llvm::make_early_inc_range(
               llvm::make_range(std::next(allocOp->getIterator()),
                                allocOp->getBlock()->end()))) {
        if (auto st = dyn_cast<memref::StoreOp>(&op);
            st && st.getMemref() == allocResult) {
          ++epoch;
          st.getMemrefMutable().assign(views[epoch]);
          continue;
        }
        if (llvm::is_contained(op.getOperands(), allocResult))
          op.replaceUsesOfWith(allocResult, views[std::max<int64_t>(epoch, 0)]);
      }
      assert(allocResult.use_empty() && "alloc still used after slot split");
    }
    allocOp.erase();
    NumAllocsMaterialized += nSlots;
    LLVM_DEBUG(llvm::dbgs()
               << "  Materialized " << nSlots << " slot(s) at base " << base
               << " (stride " << stride << ")\n");
  }
}

} // namespace
} // namespace hip
} // namespace mlir
