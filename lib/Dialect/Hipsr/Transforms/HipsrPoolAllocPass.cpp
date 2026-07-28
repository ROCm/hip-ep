/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

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

constexpr int64_t kPoolAlignment = 256;

struct Lifetime {
  size_t start;
  size_t end;
};

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

void collectAllocLifetimes(Block &body, llvm::SmallVectorImpl<Value> &allocs,
                           llvm::DenseMap<Value, Lifetime> &lifetimes) {
  llvm::DenseMap<Operation *, size_t> opIndices;
  size_t idx = 0;
  for (Operation &op : body) {
    opIndices[&op] = idx++;
  }
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
}

llvm::SmallVector<llvm::SmallVector<Value>>
greedyGroup(llvm::ArrayRef<Value> allocs,
            const llvm::DenseMap<Value, Lifetime> &lifetimes) {
  auto overlaps = [](const Lifetime &a, const Lifetime &b) {
    return !(a.end < b.start || b.end < a.start);
  };

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

Value findContext(Block &body) {
  for (BlockArgument arg : body.getArguments()) {
    if (isa<ContextType>(arg.getType())) {
      return arg;
    }
  }
  return {};
}

Operation *findLastAlloc(Block &body) {
  Operation *lastAlloc = nullptr;
  for (Operation &op : body) {
    if (isa<memref::AllocOp>(&op)) {
      lastAlloc = &op;
    }
  }
  return lastAlloc;
}

Value emitGroupSize(OpBuilder &builder, Location loc,
                    llvm::ArrayRef<Value> group, int64_t alignment) {
  llvm::SmallVector<Value> sizes;
  for (Value alloc : group) {
    auto allocOp = alloc.getDefiningOp<memref::AllocOp>();
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
    sizes.push_back(size);
  }
  Value groupSize = sizes.front();
  for (Value size : llvm::ArrayRef<Value>(sizes).drop_front()) {
    groupSize = arith::MaxUIOp::create(builder, loc, groupSize, size);
  }
  Value alignVal = arith::ConstantIndexOp::create(builder, loc, alignment);
  Value alignMinus1 =
      arith::ConstantIndexOp::create(builder, loc, alignment - 1);
  Value numerator = arith::AddIOp::create(builder, loc, groupSize, alignMinus1);
  Value divided = arith::DivUIOp::create(builder, loc, numerator, alignVal);
  return arith::MulIOp::create(builder, loc, divided, alignVal);
}

Value emitPool(OpBuilder &builder, Location loc, Value ctx, Value groupSize) {
  auto deviceSpace =
      MemorySpaceAttr::get(builder.getContext(), MemorySpaceKind::Device);
  auto poolType = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                                  MemRefLayoutAttrInterface{}, deviceSpace);
  return GetPoolOp::create(builder, loc, poolType, ctx, groupSize);
}

void replaceAllocsWithViews(OpBuilder &builder, llvm::ArrayRef<Value> group,
                            Value pool, Value offset) {
  for (Value alloc : group) {
    auto allocOp = alloc.getDefiningOp<memref::AllocOp>();
    auto view = memref::ViewOp::create(
        builder, allocOp.getLoc(), allocOp.getType(), pool, offset,
        llvm::SmallVector<Value>(allocOp.getDynamicSizes()));
    allocOp.replaceAllUsesWith(view.getResult());
    allocOp.erase();
  }
}

struct HipsrPoolAllocPass : impl::HipsrPoolAllocPassBase<HipsrPoolAllocPass> {
  using impl::HipsrPoolAllocPassBase<
      HipsrPoolAllocPass>::HipsrPoolAllocPassBase;

  void runOnOperation() override {
    getOperation().walk([&](PoolDomainOp domain) { processDomain(domain); });
  }

  void processDomain(PoolDomainOp domain) {
    Block &body = domain.getBody().front();

    llvm::SmallVector<Value> allocs;
    llvm::DenseMap<Value, Lifetime> lifetimes;
    collectAllocLifetimes(body, allocs, lifetimes);
    if (allocs.empty()) {
      return;
    }

    Value ctx = findContext(body);
    if (!ctx) {
      domain.emitError("hipsr-pool-alloc: pool_domain has poolable allocs but "
                       "no !hipsr.context operand");
      signalPassFailure();
      return;
    }

    auto groups = greedyGroup(allocs, lifetimes);

    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(findLastAlloc(body));
    Location loc = domain.getLoc();

    llvm::SmallVector<Value> groupSizes;
    for (auto &group : groups) {
      groupSizes.push_back(emitGroupSize(builder, loc, group, kPoolAlignment));
    }

    Value poolSize = groupSizes.front();
    for (Value gs : llvm::ArrayRef<Value>(groupSizes).drop_front()) {
      poolSize = arith::AddIOp::create(builder, loc, poolSize, gs);
    }

    Value pool = emitPool(builder, loc, ctx, poolSize);

    llvm::SmallVector<Value> offsets;
    offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
    for (size_t i = 1; i < groups.size(); ++i) {
      offsets.push_back(arith::AddIOp::create(builder, loc, offsets[i - 1],
                                              groupSizes[i - 1]));
    }

    for (size_t i = 0; i < groups.size(); ++i) {
      replaceAllocsWithViews(builder, groups[i], pool, offsets[i]);
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
