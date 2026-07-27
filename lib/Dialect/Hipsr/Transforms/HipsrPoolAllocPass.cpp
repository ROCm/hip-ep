/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipsrPoolAllocPass.cpp - Lifetime-based pool allocation -----------===//
//
// Pools the memref.alloc buffers inside each hipsr.pool_domain: allocations
// with disjoint lifetimes share one hipsr.get_pool byte buffer, each alloc
// becoming a memref.view at offset 0. Lifetime start = first write (DPS outs),
// end = last use. This pass implements the single-group case; more than one
// non-overlapping group errors out (left to a follow-up).
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrGetPoolOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstdint>
#include <limits>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRPOOLALLOCPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Sub-buffer alignment (bytes): every pool group starts on this boundary to
// satisfy GPU coalesced-access requirements.
constexpr int64_t kPoolAlignment = 256;

struct Lifetime {
  size_t start; // op index of first write (DPS outs)
  size_t end;   // op index of last use
};

// [start, end] intervals overlap iff NOT (a.end < b.start OR b.end < a.start).
bool overlaps(const Lifetime &a, const Lifetime &b) {
  return !(a.end < b.start || b.end < a.start);
}

// first write = earliest DPS op using the buffer as an init/out; end = last
// use. Returns std::nullopt for a buffer never written (dead), which the
// caller skips.
std::optional<Lifetime>
computeLifetime(Value buffer,
                const llvm::DenseMap<Operation *, size_t> &opIndices) {
  size_t start = std::numeric_limits<size_t>::max();
  size_t end = 0;
  for (Operation *user : buffer.getUsers()) {
    size_t userIdx = opIndices.lookup(user);
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user)) {
      for (Value init : dpsOp.getDpsInits()) {
        if (init == buffer) {
          start = std::min(start, userIdx);
          break;
        }
      }
    }
    end = std::max(end, userIdx);
  }
  if (start == std::numeric_limits<size_t>::max()) {
    return std::nullopt;
  }
  return Lifetime{start, end};
}

// Greedy grouping: allocations sorted by lifetime start are placed into the
// first existing group with no overlap, else open a new group.
llvm::SmallVector<llvm::SmallVector<Value>>
greedyGroup(llvm::ArrayRef<Value> allocs,
            const llvm::DenseMap<Value, Lifetime> &lifetimes) {
  llvm::SmallVector<Value> ordered(allocs.begin(), allocs.end());
  llvm::sort(ordered, [&](Value a, Value b) {
    return lifetimes.lookup(a).start < lifetimes.lookup(b).start;
  });

  llvm::SmallVector<llvm::SmallVector<Value>> groups;
  for (Value alloc : ordered) {
    Lifetime lt = lifetimes.lookup(alloc);
    bool placed = false;
    for (auto &group : groups) {
      bool conflicts = false;
      for (Value member : group) {
        if (overlaps(lifetimes.lookup(member), lt)) {
          conflicts = true;
          break;
        }
      }
      if (!conflicts) {
        group.push_back(alloc);
        placed = true;
        break;
      }
    }
    if (!placed) {
      groups.push_back({alloc});
    }
  }
  return groups;
}

// Byte size of an alloc = elemBytes * product(static dims) * product(dynamic
// size operands). Built from the alloc's own dynamic-size SSA operands (defined
// above the alloc), never from the buffer value itself, so it stays valid after
// the alloc is replaced and erased.
Value emitAllocByteSize(OpBuilder &builder, Location loc,
                        memref::AllocOp allocOp) {
  auto memTy = cast<MemRefType>(allocOp.getType());
  int64_t staticFactor = memTy.getElementTypeBitWidth() / 8;
  for (int64_t dim : memTy.getShape()) {
    if (!ShapedType::isDynamic(dim)) {
      staticFactor *= dim;
    }
  }
  Value size = arith::ConstantIndexOp::create(builder, loc, staticFactor);
  for (Value dynDim : allocOp.getDynamicSizes()) {
    size = arith::MulIOp::create(builder, loc, size, dynDim);
  }
  return size;
}

// alignUp(size, alignment) = ((size + alignment - 1) / alignment) * alignment.
Value emitAlignUp(OpBuilder &builder, Location loc, Value size,
                  int64_t alignment) {
  Value alignVal = arith::ConstantIndexOp::create(builder, loc, alignment);
  Value alignMinus1 =
      arith::ConstantIndexOp::create(builder, loc, alignment - 1);
  Value numerator = arith::AddIOp::create(builder, loc, size, alignMinus1);
  Value divided = arith::DivUIOp::create(builder, loc, numerator, alignVal);
  return arith::MulIOp::create(builder, loc, divided, alignVal);
}

struct HipsrPoolAllocPass : impl::HipsrPoolAllocPassBase<HipsrPoolAllocPass> {
  using impl::HipsrPoolAllocPassBase<
      HipsrPoolAllocPass>::HipsrPoolAllocPassBase;

  void runOnOperation() override {
    getOperation().walk([&](PoolDomainOp domain) { processDomain(domain); });
  }

  void processDomain(PoolDomainOp domain) {
    Block &body = domain.getBody().front();

    llvm::DenseMap<Operation *, size_t> opIndices;
    size_t idx = 0;
    for (Operation &op : body) {
      opIndices[&op] = idx++;
    }

    llvm::SmallVector<Value> allocs;
    llvm::DenseMap<Value, Lifetime> lifetimes;
    for (Operation &op : body) {
      auto allocOp = dyn_cast<memref::AllocOp>(&op);
      if (!allocOp) {
        continue;
      }
      Value buffer = allocOp.getResult();
      if (std::optional<Lifetime> lt = computeLifetime(buffer, opIndices)) {
        allocs.push_back(buffer);
        lifetimes[buffer] = *lt;
      }
    }

    // No live memref.alloc (e.g. a tensor-mode domain): leave the IR untouched.
    if (allocs.empty()) {
      return;
    }

    auto groups = greedyGroup(allocs, lifetimes);
    if (groups.size() > 1) {
      domain.emitError("hipsr-pool-alloc: multi-group pooling not yet "
                       "supported (")
          << groups.size() << " non-overlapping groups)";
      signalPassFailure();
      return;
    }

    Value ctx;
    for (BlockArgument arg : body.getArguments()) {
      if (isa<ContextType>(arg.getType())) {
        ctx = arg;
        break;
      }
    }
    // get_pool requires an !hipsr.context operand; without one (an
    // unpartitioned fixture) there is nothing this pass can anchor to, so skip.
    if (!ctx) {
      return;
    }

    Operation *lastAlloc = nullptr;
    for (Operation &op : body) {
      if (isa<memref::AllocOp>(&op)) {
        lastAlloc = &op;
      }
    }

    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(lastAlloc);
    Location loc = domain.getLoc();

    llvm::ArrayRef<Value> group = groups.front();
    Value groupSize = emitAllocByteSize(
        builder, loc, group.front().getDefiningOp<memref::AllocOp>());
    for (Value alloc : group.drop_front()) {
      Value size = emitAllocByteSize(builder, loc,
                                     alloc.getDefiningOp<memref::AllocOp>());
      groupSize = arith::MaxUIOp::create(builder, loc, groupSize, size);
    }
    groupSize = emitAlignUp(builder, loc, groupSize, kPoolAlignment);

    auto deviceSpace =
        MemorySpaceAttr::get(&getContext(), MemorySpaceKind::Device);
    auto poolType = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                                    /*layout=*/MemRefLayoutAttrInterface{},
                                    /*memorySpace=*/deviceSpace);
    Value pool = GetPoolOp::create(builder, loc, poolType, ctx, groupSize);

    Value offset = arith::ConstantIndexOp::create(builder, loc, 0);

    // View emission stays at the builder point (after the pool) so each view
    // dominates the data ops that consume it; emitting at the alloc's original
    // slot would place the view above the pool it references.
    for (Value alloc : group) {
      auto allocOp = alloc.getDefiningOp<memref::AllocOp>();
      auto view = memref::ViewOp::create(
          builder, allocOp.getLoc(), allocOp.getType(), pool, offset,
          llvm::SmallVector<Value>(allocOp.getDynamicSizes()));
      allocOp.replaceAllUsesWith(view.getResult());
      allocOp.erase();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
