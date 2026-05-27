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
// Reverse direction: GPU producer -> host load needs a stream sync
// ----------------------------------------------------------------
// The mirror pattern also occurs: a hip op writes a tiny scalar to a
// host-scratch view, and the host then reads it back to do shape arith
// (e.g. ONNX `Range(start, limit, delta)` where `start` arrives via
// `hip.cast(i64) -> f32` and the trip count is computed on the host as
// `(limit - start) / delta`). On `hipHostMallocMapped` memory the GPU
// kernel is async with respect to the host; without a stream sync the
// host load returns stale bytes, the trip count comes out 0, the
// downstream `hip.alloc(0)` returns NULL, and the model SEGVs.
//
// After this pass: for any `memref.load` whose source memref aliases the
// runtime host-scratch AND for which an unsynced `hip.*` op precedes the
// load in the same block, a `hip.host_sync(%ctx)` is inserted ahead of
// the load.
//
//   hip.cast(%ctx) ins(%x) outs(%scratch_view)
//   hip.host_sync(%ctx)                           // <-- inserted
//   %val = memref.load %scratch_view[]
//
// Multiple candidates in one function share ONE `hip.get_host_scratch`
// emitted at the entry block; each candidate gets its own 64-byte-aligned
// offset.  See test/lit/Dialect/hip-materialize-host-scalars.mlir for the
// full set of accepted/rejected shapes.
//
// Hip-dialect users are accepted, not rejected
// --------------------------------------------
// `hipHostMalloc(hipHostMallocMapped)` returns a host pointer that is also
// GPU-accessible at the same virtual address on UMA targets, so the bare-ptr
// ABI used by `--convert-hip-to-llvm` consumes the same buffer regardless of
// whether it was hipMalloc'd or hipHostMalloc'd.  This is the correctness fix
// vs the early design that rejected hip consumers: the canonical
// `tensor.from_elements -> reduce_sum + sub + cast -> seqlens_k` GQA
// pattern produces exactly an alloc with a host `memref.store` followed
// by a `hip.cast` reading it; rejecting it left the host store crashing
// inside the GPU pool.  `isHostScalarCandidate` allows `hip.*` users for
// this reason -- see the inline comment on the user-classification loop.
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
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-materialize-host-scalars"

STATISTIC(NumAllocsMaterialized,
          "Number of memref.alloc redirected to host-mapped scratch");
STATISTIC(NumHostSyncsInserted,
          "Number of hip.host_sync inserted before host loads of scratch");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_MATERIALIZEHOSTSCALARSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Round \p x up to a multiple of \p align (align must be a power of two).
static int64_t roundUp(int64_t x, int64_t align) {
  return (x + align - 1) & ~(align - 1);
}

/// True if \p allocOp looks like a tiny host-fed scalar staging buffer:
///   - static shape, small (<=16 elements)
///   - integer or index element type (no float -> these are bigger and almost
///     always GPU-consumed in flight)
///   - has at least one host I/O user (memref.store or memref.load): this is
///     the SEGV trigger on targets where the GPU pool is real device memory
///     — host accessing a GPU-pool address
///   - every user is in {memref.store, load, dim, dealloc} OR is a hip dialect
///     op. Hip consumers are fine: hipHostMalloc(hipHostMallocMapped) returns
///     a host pointer that is also GPU-accessible at the same VA on UMA
///     targets, so the bare-ptr ABI used by --convert-hip-to-llvm works
///     whether the backing memory is hipMalloc'd or hipHostMalloc'd.
///
/// Critically, we DO accept allocs with hip op users (writers and readers).
/// The original implementation rejected those — but the canonical
/// `tensor.from_elements` -> `reduce_sum + sub + cast -> seqlens_k` pattern
/// produces exactly such an alloc: `memref.store` from host then `hip.cast`
/// reads it. Rejecting it left the host store crashing inside the GPU pool.
static bool isHostScalarCandidate(memref::AllocOp allocOp) {
  MemRefType type = allocOp.getType();
  if (!type.hasStaticShape())
    return false;
  if (type.getNumElements() > 16)
    return false;
  Type elem = type.getElementType();
  // Integer / index types are the canonical host-scalar shape (loop counters,
  // dim arithmetic, GQA `seqlens_k`, etc.). Float types also appear in vision
  // encoder graphs: e.g. an ONNX `Range(start_f32, limit_f32, delta_f32)`
  // computes its output element count from the scalar operands via
  // `arith.divf + arith.ceildivsi + ...` host arithmetic, and the operands
  // arrive as `memref<f32>` from a preceding `hip.cast(i64) -> f32`. Without
  // host-mapped backing, the host-side arith.divf dereferences a GPU pointer
  // → access violation (same bug class as the documented i64 SEGV).
  //
  // Index-typed scalars are also accepted — they reach this point via
  // bufferized `tensor.from_elements` of an `index` value, and behave
  // identically to integer scalars.
  if (!elem.isIntOrIndex() && !mlir::isa<mlir::FloatType>(elem))
    return false;

  // Walk the alloc's transitive view-like users to discover host I/O. The
  // canonical regression pattern is
  //   %a = memref.alloc() : memref<i64>
  //   %rc = memref.reinterpret_cast %a ... to memref<1xi64>
  //   hip.gather(...) outs(%a : memref<i64>)        // GPU writer
  //   %v = memref.load %rc[%c0] : memref<1xi64>     // HOST reader via alias
  // The host load is reachable through `memref.reinterpret_cast` (and its
  // siblings), not directly on the alloc, so a flat user check misses it
  // and leaves the alloc in the GPU pool — racing the next host load. The
  // walk treats view-like ops as transparent: their users count for both
  // hostIO detection and user-kind filtering. Bails on truly unknown
  // dialects (the original safety net).
  bool hasHostIO = false;
  llvm::SmallVector<Operation *, 8> worklist;
  llvm::SmallPtrSet<Operation *, 8> seen;
  for (Operation *u : allocOp->getUsers())
    if (seen.insert(u).second)
      worklist.push_back(u);
  while (!worklist.empty()) {
    Operation *user = worklist.pop_back_val();
    if (isa<memref::StoreOp, memref::LoadOp>(user)) {
      hasHostIO = true;
      continue;
    }
    // memref.copy is a host-side memcpy on the bufferized IR. It indicates
    // that the bytes participate in host-side memory motion (e.g. building
    // a small `tensor.from_elements` shape vector), and it requires both
    // operands to be host-accessible. Treat it as host I/O.
    if (isa<memref::CopyOp>(user)) {
      hasHostIO = true;
      continue;
    }
    if (isa<memref::DimOp, memref::DeallocOp>(user))
      continue;
    // Hip dialect users (e.g. hip.cast that consumes a host-stored scalar to
    // produce a GPU i32 for GQA) are fine — see comment above.
    if (user->getDialect() && user->getDialect()->getNamespace() == "hip")
      continue;
    // View-like memref ops are transparent — enqueue their users. memref.reshape
    // is also accepted (the runtime-shape fallback for tensor.reshape).
    if (isa<memref::ViewOp, memref::SubViewOp, memref::CastOp,
            memref::ReinterpretCastOp, memref::ExpandShapeOp,
            memref::CollapseShapeOp, memref::ReshapeOp>(user)) {
      for (Operation *u2 : user->getUsers())
        if (seen.insert(u2).second)
          worklist.push_back(u2);
      continue;
    }
    // Truly unknown dialect / op — bail.
    return false;
  }
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

  // Compute byte size per candidate and aligned offsets. 64-byte alignment
  // matches the runtime pool and is comfortably above the largest scalar
  // alignment; gives every candidate its own cache line.
  constexpr int64_t kAlign = 64;
  SmallVector<int64_t> offsets;
  offsets.reserve(candidates.size());
  int64_t total = 0;
  for (memref::AllocOp allocOp : candidates) {
    MemRefType ty = allocOp.getType();
    Type elemTy = ty.getElementType();
    unsigned bits = elemTy.isIndex() ? 64 : elemTy.getIntOrFloatBitWidth();
    int64_t elemBytes = static_cast<int64_t>((bits + 7) / 8);
    int64_t bytes = ty.getNumElements() * elemBytes;
    if (bytes == 0)
      bytes = 1; // rank-0 with i1 etc. — never zero-sized
    int64_t off = roundUp(total, kAlign);
    offsets.push_back(off);
    total = off + bytes;
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

  // Replace each candidate with a memref.view over the scratch buffer.
  // Deallocs of the candidate are erased — the scratch buffer is owned by
  // the runtime (released in hipdnn_ep_state_cleanup), like the pool.
  for (auto [allocOp, offset] : llvm::zip(candidates, offsets)) {
    builder.setInsertionPoint(allocOp);
    Value offsetVal =
        arith::ConstantIndexOp::create(builder, allocOp.getLoc(), offset);
    auto view = memref::ViewOp::create(builder, allocOp.getLoc(),
                                       allocOp.getType(), scratch, offsetVal,
                                       /*sizes=*/ValueRange{});

    // Erase deallocs first (they reference the alloc's result).
    SmallVector<memref::DeallocOp> deallocs;
    for (Operation *user : allocOp->getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        deallocs.push_back(d);
    for (auto d : deallocs)
      d.erase();

    allocOp.replaceAllUsesWith(view.getResult());
    allocOp.erase();
    ++NumAllocsMaterialized;
    LLVM_DEBUG(llvm::dbgs()
               << "  Materialized " << view << " at offset " << offset << "\n");
  }

  // ---------------------------------------------------------------------
  // Insert hip.host_sync before any memref.load whose source memref aliases
  // the runtime host-scratch AND for which an unsynced hip.* op precedes the
  // load in the same block. The scratch is hipHostMallocMapped — the host
  // CAN read it, but GPU writes are async and not visible until the stream
  // is synced.
  //
  // The alias check walks back through memref view-likes (view, subview,
  // cast, reinterpret_cast, expand_shape, collapse_shape) until either the
  // scratch base (`hip.get_host_scratch` result, here just `scratch`) is
  // reached or a non-view producer is hit. Conservative: any hip-dialect op
  // in source order before the load marks the block "dirty"; the next
  // dirty-block load gets a sync inserted ahead of it, after which the
  // block becomes clean again until the next hip op.
  //
  // Before:
  //   hip.cast(%ctx) ins(%x) outs(%view : memref<f32>)
  //   %v = memref.load %view[] : memref<f32>
  // After:
  //   hip.cast(%ctx) ins(%x) outs(%view : memref<f32>)
  //   hip.host_sync(%ctx)
  //   %v = memref.load %view[] : memref<f32>
  auto aliasesScratch = [scratch](Value v) -> bool {
    llvm::SmallPtrSet<Operation *, 8> seen;
    while (v) {
      if (v == scratch)
        return true;
      Operation *def = v.getDefiningOp();
      if (!def || !seen.insert(def).second)
        return false;
      if (isa<memref::ViewOp, memref::SubViewOp, memref::CastOp,
              memref::ReinterpretCastOp, memref::ExpandShapeOp,
              memref::CollapseShapeOp>(def)) {
        v = def->getOperand(0);
        continue;
      }
      return false;
    }
    return false;
  };

  for (Block &block : funcOp.getBody()) {
    bool dirty = false;
    for (Operation &op : llvm::make_early_inc_range(block)) {
      if (auto loadOp = dyn_cast<memref::LoadOp>(&op)) {
        if (dirty && aliasesScratch(loadOp.getMemRef())) {
          OpBuilder b(&op);
          HostSyncOp::create(b, op.getLoc(), ctx);
          dirty = false;
          ++NumHostSyncsInserted;
        }
        continue;
      }
      // memref.copy is a host-side memcpy; the source is read on the host
      // just like memref.load. Treat it the same way for sync insertion.
      if (auto copyOp = dyn_cast<memref::CopyOp>(&op)) {
        if (dirty && (aliasesScratch(copyOp.getSource()) ||
                      aliasesScratch(copyOp.getTarget()))) {
          OpBuilder b(&op);
          HostSyncOp::create(b, op.getLoc(), ctx);
          dirty = false;
          ++NumHostSyncsInserted;
        }
        continue;
      }
      // Any hip op (other than the sync we just inserted) marks the block
      // dirty. Be conservative — even hip ops that don't touch scratch may
      // be followed by ones that do via an aliasing chain we don't see.
      if (op.getDialect() &&
          op.getDialect()->getNamespace() == "hip" &&
          !isa<HostSyncOp, GetHostScratchOp, GetPoolOp, GetConstantOp>(&op)) {
        dirty = true;
      }
    }
  }
}

} // namespace
} // namespace hip
} // namespace mlir
