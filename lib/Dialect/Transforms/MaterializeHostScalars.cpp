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
// View ops are matched generically through `ViewLikeOpInterface` rather than a
// fixed `isa<>` list, so `memref.reshape` (whose result aliases its DATA
// operand, never its shape operand) and any future view op are covered without
// editing this pass.  `memref.copy` is accepted as a terminal host-mapping-safe
// use: some shape-staging chains end in a copy into another buffer, and the
// host-mapped scratch is a valid copy source/destination.
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
//   - Allocs/allocas carrying an explicit `#hip.mem<>` memory space: these
//     carry a deliberate space attribute (e.g. a `#hip.mem<host>` destination
//     for a cross-space copy) and are otherwise indistinguishable from a
//     host-staged scalar. Grabbing one here would build a `memref.view` whose
//     host-space result mismatches the space-less scratch base, so the
//     explicit-space filter leaves them alone.
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
#include "mlir/Interfaces/ViewLikeInterface.h"

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

/// Classify every transitive user of \p memrefVal and decide whether they are
/// all compatible with the backing buffer living in host-mapped scratch (see
/// the file header for the rationale behind each category). Sets \p sawHostIO
/// when a `memref.store`/`load` is found anywhere in the chain. Returns false
/// at the first user we cannot vouch for -- an op of an unknown dialect, or a
/// view whose own users escape.
///
/// View/descriptor ops are recognized generically via `ViewLikeOpInterface`
/// (reinterpret_cast, expand/collapse_shape, subview, view, cast, AND reshape)
/// instead of a hand-maintained `isa<>` list. They touch no memory themselves,
/// so the backing buffer is still a host-staged scalar; we recurse into the
/// view's result. The recursion is gated on `getViewSource() == memrefVal`:
/// this matters for `memref.reshape`, whose viewed source is the DATA operand,
/// not the shape operand -- when our buffer is the shape operand the reshape
/// result aliases the data (not us), so it is a terminal accept, not a recurse.
/// `memref.copy` is not view-like (it carries Read/Write memory effects); it is
/// a terminal host-mapping-safe use and does not itself flag host I/O.
static bool classifyHostScalarUsers(Value memrefVal, bool &sawHostIO) {
  for (Operation *user : memrefVal.getUsers()) {
    // Host I/O — the SEGV trigger we're staging away from the GPU pool.
    if (isa<memref::StoreOp, memref::LoadOp>(user)) {
      sawHostIO = true;
      continue;
    }
    // Metadata-only / lifetime users: harmless.
    if (isa<memref::DimOp, memref::DeallocOp>(user))
      continue;
    // hip.* consumers are host-mapping-safe (hipHostMallocMapped is
    // GPU-readable at the same VA on UMA targets).
    if (user->getDialect() && user->getDialect()->getNamespace() == "hip")
      continue;
    // View/alias ops: recurse ONLY into the result that actually aliases this
    // buffer. getViewSource() pins which operand is the viewed one — critical
    // for memref.reshape, where our buffer may be the SHAPE operand (then the
    // result aliases the DATA, not us → terminal accept, not a recurse).
    if (auto view = dyn_cast<ViewLikeOpInterface>(user)) {
      if (view.getViewSource() == memrefVal) {
        if (!classifyHostScalarUsers(view->getResult(0), sawHostIO))
          return false;
      }
      continue;
    }
    // memref.copy is not view-like; it's a terminal host-mapping-safe use
    // (Read src / Write dst) and does NOT itself flag host I/O.
    if (isa<memref::CopyOp>(user))
      continue;

    return false; // genuinely unknown user → reject candidate
  }
  return true;
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
  // Allocs carrying an explicit #hip.mem<> space are not candidates for this
  // pinned host_scratch pool. In particular a small #hip.mem<host> integer
  // buffer (e.g. a cross-space copy destination) otherwise looks exactly like a
  // host-staged scalar; grabbing it here would emit a memref.view whose
  // #hip.mem<host> result mismatches the space-less scratch base (memref<?xi8>)
  // and fail the memref.view verifier. Leave it alone.
  if (dyn_cast_or_null<MemorySpaceAttr>(type.getMemorySpace()))
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
