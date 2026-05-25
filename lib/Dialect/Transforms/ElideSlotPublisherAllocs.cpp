/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ElideSlotPublisherAllocs.cpp - Shrink Cat-C DPS-init allocs --------===//
//
// Phase 1 of the slot-buffer-coalescing initiative. See
// docs/design/slot-buffer-coalesce.md.
//
// Cat-C publishers (NonZero, Range Cat-C, ConstantOfShape Cat-C) allocate
// and publish their own exact-size GPU buffer at runtime through the
// hipdnn_ep_state_dyn_pool_alloc + hipdnn_ep_state_publish_buffer pair.
// The DPS-init memref.alloc that bufferize materialised for them holds
// an UPPER BOUND that the wrapper never reads or writes. PoolAllocs
// would otherwise reserve those upper-bound bytes in the static pool.
//
// This pass shrinks each such alloc to a 0-byte placeholder by replacing
// every dynamic-size operand with arith.constant 0 : index. The alloc's
// MemRefType (and therefore the SSA edges to the publisher and to any
// downstream consumer) stays unchanged; only the runtime size is
// collapsed. PoolAllocs then packs a 0-byte dynamic alloc whose bucket
// contributes 0 bytes to the pool footprint.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-elide-slot-publisher-allocs"

STATISTIC(NumPublishersElided,
          "Number of Cat-C publisher DPS-init allocs shrunk to 0 bytes");
STATISTIC(NumPublishersSkipped,
          "Number of Cat-C publisher DPS-init allocs left untouched (unsafe)");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_ELIDESLOTPUBLISHERALLOCSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// Walk back from `v` through any sequence of view-like memref ops
// (memref.cast, memref.view, memref.subview, memref.reinterpret_cast,
// memref.expand_shape, memref.collapse_shape) until we find the
// underlying memref.alloc. Returns nullptr if `v` is not rooted at a
// memref.alloc (e.g. it's a func-arg or a non-alloc allocation op).
memref::AllocOp findAllocBehind(Value v) {
  Value cursor = v;
  for (int depth = 0; depth < 32 && cursor; ++depth) {
    Operation *def = cursor.getDefiningOp();
    if (!def)
      return nullptr;
    if (auto alloc = dyn_cast<memref::AllocOp>(def))
      return alloc;
    // View-like ops: chase operand 0. Restrict to memref dialect to keep
    // the walk tight.
    if (def->getDialect() !=
        def->getContext()->getLoadedDialect<memref::MemRefDialect>())
      return nullptr;
    if (def->getNumOperands() == 0)
      return nullptr;
    cursor = def->getOperand(0);
  }
  return nullptr;
}

// Return true iff every non-dealloc use of `allocResult` is "safe": the
// publisher itself, another hip-dialect op, a memref.dim query, or a
// view-like memref op. Any other use causes us to skip (be safe).
bool everyUseSafeToShrink(Value allocResult, Operation *publisher) {
  auto *hipDialect = allocResult.getContext()->getLoadedDialect<HipDialect>();
  auto *memrefDialect =
      allocResult.getContext()->getLoadedDialect<memref::MemRefDialect>();
  for (Operation *user : allocResult.getUsers()) {
    if (user == publisher)
      continue;
    if (isa<memref::DeallocOp>(user))
      continue;
    if (isa<memref::DimOp>(user))
      continue;
    // View-like memref reshaping is OK -- the descriptor is the only
    // thing being recomputed; downstream consumers either (a) honor
    // hipdnn.input_slot_buffers (rewired pointer) or (b) read via
    // hipdnn.input_dim_slots (rewired dim).
    if (user->getDialect() == memrefDialect)
      continue;
    if (user->getDialect() == hipDialect)
      continue;
    // Conservative: anything else (e.g. tensor-dialect leftover, linalg
    // op) is treated as opaque and we abandon the elision for this
    // publisher.
    LLVM_DEBUG(llvm::dbgs()
               << "  skip elision: unexpected user of publisher alloc: "
               << *user << "\n");
    return false;
  }
  return true;
}

struct ElideSlotPublisherAllocsPass
    : public impl::ElideSlotPublisherAllocsPassBase<
          ElideSlotPublisherAllocsPass> {

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = &getContext();
    auto *hipDialect = ctx->getLoadedDialect<HipDialect>();

    // Collect first, mutate second: walking the IR while shrinking allocs
    // is fine for individual ops but cleaner to do in two phases.
    SmallVector<std::pair<Operation *, memref::AllocOp>> work;
    SmallPtrSet<Operation *, 8> seenAllocs;
    funcOp.walk([&](Operation *op) {
      if (op->getDialect() != hipDialect)
        return;
      if (!op->hasAttr("hipdnn.elide_dps_init"))
        return;
      auto dpsOp = dyn_cast<DestinationStyleOpInterface>(op);
      if (!dpsOp)
        return;
      auto inits = dpsOp.getDpsInits();
      for (Value init : inits) {
        // After bufferize, the init operand is a memref produced (possibly
        // via a chain of view-like ops) by a memref.alloc with dynamic
        // upper-bound size operands. We want THAT alloc.
        memref::AllocOp alloc = findAllocBehind(init);
        if (!alloc) {
          LLVM_DEBUG(llvm::dbgs()
                     << "  no alloc backing init operand of publisher: " << *op
                     << "\n");
          continue;
        }
        // Skip if the alloc has no dynamic operands -- the type is
        // already fully static and there's nothing to shrink. (Could
        // happen for a Cat-C op whose upper bound is a compile-time
        // constant -- still wasteful but no operands to rewrite.)
        if (alloc.getDynamicSizes().empty())
          continue;
        // De-duplicate: a single alloc may back multiple publisher inits
        // if (unlikely) the same buffer was wired into two ops.
        if (!seenAllocs.insert(alloc.getOperation()).second)
          continue;
        if (!everyUseSafeToShrink(alloc.getResult(), op)) {
          ++NumPublishersSkipped;
          continue;
        }
        work.emplace_back(op, alloc);
      }
    });

    if (work.empty())
      return;

    OpBuilder builder(ctx);
    for (auto [publisher, alloc] : work) {
      // Replace every dynamic-size operand with `arith.constant 0 : index`.
      // The MemRefType stays the same (dim is still dynamic in the type),
      // but the runtime extent collapses to zero. PoolAllocs treats this
      // as a 0-byte dynamic bucket whose contribution to the pool size
      // is `aligned(0) * num_bins = 0`.
      builder.setInsertionPoint(alloc);
      Value zero = arith::ConstantIndexOp::create(builder, alloc.getLoc(), 0);
      // memref.alloc's operand layout is `dynamicSizes ++ symbolOperands`.
      // We rewrite exactly the first `dynamicSizes.size()` operands so
      // any symbolOperands (none today, but be defensive) are untouched.
      const unsigned numDyn = alloc.getDynamicSizes().size();
      for (unsigned i = 0; i < numDyn; ++i)
        alloc->setOperand(i, zero);
      ++NumPublishersElided;
      LLVM_DEBUG(llvm::dbgs() << "  elided DPS-init of publisher " << *publisher
                              << " (alloc " << alloc << ")\n");
    }
  }
};

} // namespace
} // namespace hip
} // namespace mlir
