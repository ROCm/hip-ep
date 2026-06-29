/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PoolHostTransfers.cpp - pageable host pool for #hip.mem<host> ------===//
//
// Packs every `memref.alloc` whose result type carries the `#hip.mem<host>`
// memory space into a single per-function `hip.get_host_mem(%ctx, %total)`
// PAGEABLE host buffer (plain malloc/realloc, runtime-owned, grow-on-demand).
// Each candidate becomes a `memref.view` at its own 64-byte-aligned offset and
// the original `memref.dealloc`s are erased.
//
// Why this pass exists
// --------------------
// Bufferizing `hip.transfer %x to host` (HipTransferBufferizableModel) emits a
// `memref.alloc` in `#hip.mem<host>` as the device->host copy destination
// (today's only producer: hip.pad's pads/axes). Such an alloc must NEVER reach
// `MemRefAllocOpLowering`, which would lower it to the undefined
// `hip_device_malloc`. This pass consumes them first, routing them to a
// dedicated pageable host pool that is fully SEPARATE from the pinned
// `get_host_scratch` pool used by `hip-materialize-host-scalars` (so a pageable
// `host` buffer is never mislabelled as `pinned`). It runs adjacent to
// `hip-materialize-host-scalars` (after `hip-promote-strided-hip-operands`,
// before `hip-pool-allocs`).
//
// Example IR
// ----------
// Before:
//   %a = memref.alloc() : memref<8xi64, #hip.mem<host>>     // <-- candidate
//   hip.memcpy_d2h_async(%ctx, %a, %dev : ..., ...)
//   memref.dealloc %a : memref<8xi64, #hip.mem<host>>
//
// After (alloc replaced by a view into the per-function pageable host pool):
//   %total = arith.constant 64 : index
//   %hm    = hip.get_host_mem(%ctx, %total) : memref<?xi8, #hip.mem<host>>
//   %c0    = arith.constant 0 : index
//   %a     = memref.view %hm[%c0][] : memref<?xi8, #hip.mem<host>>
//                 to memref<8xi64, #hip.mem<host>>
//   hip.memcpy_d2h_async(%ctx, %a, %dev : ..., ...)
//
// Multiple candidates in one function (pad's pads AND axes live simultaneously)
// share ONE `hip.get_host_mem` at distinct aligned offsets.
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

#define DEBUG_TYPE "hip-pool-host-transfers"

STATISTIC(NumHostAllocsPooled,
          "Number of #hip.mem<host> memref.alloc packed into the host pool");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_POOLHOSTTRANSFERSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Round \p x up to a multiple of \p align (align must be a power of two).
static int64_t roundUp(int64_t x, int64_t align) {
  return (x + align - 1) & ~(align - 1);
}

// True if \p type is a memref in the #hip.mem<host> memory space.
static bool isHostSpaceMemRef(MemRefType type) {
  auto sp = dyn_cast_or_null<MemorySpaceAttr>(type.getMemorySpace());
  return sp && sp.getKind() == MemorySpaceKind::Host;
}

struct PoolHostTransfersPass
    : public impl::PoolHostTransfersPassBase<PoolHostTransfersPass> {
  void runOnOperation() override;
};

void PoolHostTransfersPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  // Need !hip.context as arg 0 to call hip.get_host_mem. If absent, skip.
  if (funcOp.getNumArguments() == 0 ||
      !isa<ContextType>(funcOp.getArgument(0).getType()))
    return;
  Value ctx = funcOp.getArgument(0);

  // Collect every static-shape #hip.mem<host> alloc. Post-order walk is
  // deterministic; the traversal order fixes the per-candidate offsets (and
  // therefore the LIT IR snapshots), so don't reorder.
  SmallVector<memref::AllocOp> candidates;
  funcOp.walk([&](memref::AllocOp op) {
    MemRefType ty = op.getType();
    if (ty.hasStaticShape() && isHostSpaceMemRef(ty))
      candidates.push_back(op);
  });
  if (candidates.empty())
    return;

  // Compute byte size per candidate and aligned offsets. 64-byte alignment
  // matches the runtime pool and gives every candidate its own cache line.
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
      bytes = 1;
    int64_t off = roundUp(total, kAlign);
    offsets.push_back(off);
    total = off + bytes;
  }
  total = roundUp(total, kAlign);

  // Emit hip.get_host_mem(%ctx, %total) at function entry. The pool memref is
  // rank-1 i8 in the host memory space; memref.view requires source and result
  // to share the memory space, so candidates (also host space) view into it.
  Block &entry = funcOp.getBody().front();
  OpBuilder builder(funcOp.getContext());
  builder.setInsertionPointToStart(&entry);
  Location loc = funcOp.getLoc();
  Value totalSize = arith::ConstantIndexOp::create(builder, loc, total);
  auto hostSpace =
      MemorySpaceAttr::get(funcOp.getContext(), MemorySpaceKind::Host);
  auto poolType = MemRefType::get({ShapedType::kDynamic},
                                  builder.getIntegerType(8),
                                  /*layout=*/MemRefLayoutAttrInterface{},
                                  /*memorySpace=*/hostSpace);
  Value pool = GetHostMemOp::create(builder, loc, poolType, ctx, totalSize);

  // Replace each candidate with a memref.view over the pool buffer. Deallocs
  // are erased — the pool is runtime-owned (released in cleanup).
  for (auto [allocOp, offset] : llvm::zip(candidates, offsets)) {
    builder.setInsertionPoint(allocOp);
    Value offsetVal =
        arith::ConstantIndexOp::create(builder, allocOp.getLoc(), offset);
    auto view = memref::ViewOp::create(builder, allocOp.getLoc(),
                                       allocOp.getType(), pool, offsetVal,
                                       /*sizes=*/ValueRange{});

    SmallVector<memref::DeallocOp> deallocs;
    for (Operation *user : allocOp->getUsers())
      if (auto d = dyn_cast<memref::DeallocOp>(user))
        deallocs.push_back(d);
    for (auto d : deallocs)
      d.erase();

    allocOp.replaceAllUsesWith(view.getResult());
    allocOp.erase();
    ++NumHostAllocsPooled;
    LLVM_DEBUG(llvm::dbgs()
               << "  Pooled host transfer " << view << " at offset " << offset
               << "\n");
  }
}

} // namespace
} // namespace hip
} // namespace mlir
