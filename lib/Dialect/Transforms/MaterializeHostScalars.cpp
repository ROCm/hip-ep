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
// On gfx1151 the GPU pool is real device memory; the bufferized
// `tensor.from_elements` lowering emits `memref.alloc + memref.store`
// which, once absorbed by `hip-pool-allocs`, becomes a host store into
// device memory and SEGVs.  gfx1150 silently worked because hipMalloc
// returned UMA-mapped host memory there.  This pass runs BEFORE
// `hip-pool-allocs` so candidates never enter the GPU pool.
//
// Hip-dialect users are accepted, not rejected
// --------------------------------------------
// `hipHostMalloc(hipHostMallocMapped)` returns a host pointer that is also
// GPU-accessible at the same virtual address on UMA targets (gfx1100,
// gfx1101, gfx1150, gfx1151 verified), so the bare-ptr ABI used by
// `--convert-hip-to-llvm` consumes the same buffer regardless of whether
// it was hipMalloc'd or hipHostMalloc'd.  This is the correctness fix vs
// the early design that rejected hip consumers: the canonical
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

#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

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

/// True if \p allocOp looks like a tiny host-fed scalar staging buffer:
///   - static shape, small (<=16 elements)
///   - integer or index element type (no float -> these are bigger and almost
///     always GPU-consumed in flight)
///   - has at least one host I/O user (memref.store or memref.load): this is
///     the SEGV trigger on gfx1151 — host accessing a GPU-pool address
///   - every user is in {memref.store, load, dim, dealloc} OR is a hip dialect
///     op. Hip consumers are fine: hipHostMalloc(hipHostMallocMapped) returns
///     a host pointer that is also GPU-accessible at the same VA on UMA
///     (gfx1150/gfx1151), so the bare-ptr ABI used by --convert-hip-to-llvm
///     works whether the backing memory is hipMalloc'd or hipHostMalloc'd.
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
  if (!elem.isIntOrIndex())
    return false;

  bool hasHostIO = false;
  for (Operation *user : allocOp->getUsers()) {
    if (isa<memref::StoreOp, memref::LoadOp>(user)) {
      hasHostIO = true;
      continue;
    }
    if (isa<memref::DimOp, memref::DeallocOp>(user))
      continue;
    // Hip dialect users (e.g. hip.cast that consumes a host-stored scalar to
    // produce a GPU i32 for GQA) are fine — see comment above.
    if (user->getDialect() && user->getDialect()->getNamespace() == "hip")
      continue;
    // Anything else (view-likes, casts that escape the function, unknown
    // dialects) — bail out: we can't reason about its memory expectations.
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
}

} // namespace
} // namespace hip
} // namespace mlir
