/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- DedupDPSInits.cpp - liveness-safe tensor.empty DPS dedup -----------===//
//
// Merges `tensor.empty` ops that feed DPS `outs` operands when their
// consumers have non-overlapping live ranges in the tensor domain.
//
// Each HIP DPS op (`hip.gemm`, `hip.matmul`, …) takes a `tensor.empty`
// as its destination-passing-style init. `tensor.empty` is side-effect-free,
// so a stock CSE merges ALL same-typed empties — after bufferize, the DPS
// ops share one buffer and clobber each other when simultaneously live
// (Whisper encoder: cosine ~0.6). Removing CSE entirely is safe for
// correctness but prevents buffer reuse across transformer layers,
// inflating the GPU pool ~5× on deep models (gemma3-4b: 7 → 35 GB).
//
// This pass replaces the unsafe CSE: it groups empties by result type and
// only merges those whose live intervals `[empty_index, last_consumer_index]`
// do not overlap, so simultaneously-live empties stay distinct.
//
// Before (two non-overlapping DPS consumers share one empty):
//   %e = tensor.empty() : tensor<2x8xf16>
//   %a = hip.matmul ... outs(%e)       // consumer at index 10
//   ... %a consumed and dead ...
//   %b = hip.matmul ... outs(%e)       // consumer at index 30 (reuses %e)
//
// Before (two simultaneously-live DPS consumers keep separate empties):
//   %e0 = tensor.empty() : tensor<2x8xf16>
//   %e1 = tensor.empty() : tensor<2x8xf16>
//   %a  = hip.matmul ... outs(%e0)     // consumer at index 10
//   %b  = hip.matmul ... outs(%e1)     // consumer at index 12
//   ... both %a and %b read later ...  // overlapping lifetimes → no merge
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

namespace mlir {
namespace hip {
#define GEN_PASS_DEF_DEDUPDPSINITSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"
} // namespace hip
} // namespace mlir

using namespace mlir;

namespace {

struct EmptyInfo {
  tensor::EmptyOp emptyOp;
  unsigned defIndex;
  unsigned lastConsumerIndex;
};

struct DedupDPSInitsPass
    : public hip::impl::DedupDPSInitsPassBase<DedupDPSInitsPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();

    Block *block = &funcOp.getBody().front();
    if (block->empty())
      return;

    // Build op → sequential-index map.
    DenseMap<Operation *, unsigned> opIndex;
    unsigned idx = 0;
    for (Operation &op : *block)
      opIndex[&op] = idx++;
    unsigned blockSize = idx;

    // Collect tensor.empty ops that feed at least one DPS init.
    SmallVector<EmptyInfo> empties;
    for (Operation &op : *block) {
      auto emptyOp = dyn_cast<tensor::EmptyOp>(&op);
      if (!emptyOp)
        continue;

      unsigned lastConsumer = opIndex[&op];
      bool feedsDPS = false;

      for (OpOperand &use : emptyOp.getResult().getUses()) {
        Operation *user = use.getOwner();
        auto dstOp = dyn_cast<DestinationStyleOpInterface>(user);
        if (!dstOp)
          continue;
        // Check if this use is a DPS init (outs) operand.
        if (!dstOp.isDpsInit(&use))
          continue;
        feedsDPS = true;

        // The last consumer of this empty is the latest DPS op using it
        // as init, plus any user of the DPS result (they read the buffer
        // the empty will become). In the tensor domain, the DPS result
        // is a NEW SSA value, so we trace its users too.
        OpResult tiedResult = dstOp.getTiedOpResult(&use);
        for (Operation *resultUser : tiedResult.getUsers()) {
          Operation *resolved = resultUser;
          if (resolved->getBlock() != block)
            resolved = block->findAncestorOpInBlock(*resolved);
          if (resolved) {
            auto it = opIndex.find(resolved);
            if (it != opIndex.end())
              lastConsumer = std::max(lastConsumer, it->second);
          } else {
            lastConsumer = blockSize - 1;
          }
        }
        // The DPS op itself is also a consumer.
        auto it = opIndex.find(user);
        if (it != opIndex.end())
          lastConsumer = std::max(lastConsumer, it->second);
      }

      if (!feedsDPS)
        continue;

      empties.push_back({emptyOp, opIndex[&op], lastConsumer});
    }

    if (empties.empty())
      return;

    // Group by result type.
    DenseMap<Type, SmallVector<unsigned>> typeGroups;
    for (auto [i, info] : llvm::enumerate(empties))
      typeGroups[info.emptyOp.getType()].push_back(i);

    // Greedy merge within each type group: for each empty (in textual
    // order), find the earliest existing "representative" whose interval
    // does not overlap. If found, replace; otherwise this empty becomes
    // a new representative.
    unsigned mergeCount = 0;
    for (auto &[type, indices] : typeGroups) {
      // Representatives: empties we keep, each with their (possibly
      // extended) live interval.
      SmallVector<unsigned> reps; // indices into `empties`

      for (unsigned idx : indices) {
        EmptyInfo &info = empties[idx];
        bool merged = false;

        for (unsigned repIdx : reps) {
          EmptyInfo &rep = empties[repIdx];
          // Non-overlapping: one ends before the other starts.
          bool overlap = !(info.lastConsumerIndex < rep.defIndex ||
                           rep.lastConsumerIndex < info.defIndex);
          if (!overlap) {
            // Merge: replace this empty's uses with the representative's.
            info.emptyOp.getResult().replaceAllUsesWith(
                rep.emptyOp.getResult());
            info.emptyOp->erase();
            // Extend the representative's interval to cover both.
            rep.lastConsumerIndex =
                std::max(rep.lastConsumerIndex, info.lastConsumerIndex);
            rep.defIndex = std::min(rep.defIndex, info.defIndex);
            merged = true;
            ++mergeCount;
            break;
          }
        }

        if (!merged)
          reps.push_back(idx);
      }
    }

    if (mergeCount > 0)
      llvm::errs() << "[DedupDPSInits] merged " << mergeCount
                   << " tensor.empty ops (liveness-safe)\n";
  }
};

} // namespace
