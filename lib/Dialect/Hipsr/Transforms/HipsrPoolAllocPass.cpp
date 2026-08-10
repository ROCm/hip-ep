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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRPOOLALLOCPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct Lifetime {
  size_t start;
  size_t end;
};

struct AllocationSize {
  int64_t staticBytes;
  llvm::SmallVector<Value> dynamicDims;
};

constexpr int64_t kPoolAlignment = 256;

int64_t staticByteFactor(MemRefType memTy) {
  int64_t factor = memTy.getElementTypeBitWidth() / 8;
  for (int64_t dim : memTy.getShape()) {
    if (!ShapedType::isDynamic(dim)) {
      factor *= dim;
    }
  }
  return factor;
}

AllocationSize computeAllocationSize(memref::AllocOp allocOp) {
  return {staticByteFactor(allocOp.getType()),
          llvm::to_vector(allocOp.getDynamicSizes())};
}

llvm::DenseMap<Value, Lifetime> computeLiveness(Block &body) {
  llvm::DenseMap<Operation *, size_t> opIndices;
  size_t idx = 0;
  for (Operation &op : body) {
    opIndices[&op] = idx++;
  }

  llvm::DenseMap<Value, Lifetime> lifetimes;
  for (Operation &op : body) {
    auto allocOp = dyn_cast<memref::AllocOp>(&op);
    if (!allocOp) {
      continue;
    }
    Value buffer = allocOp.getResult();
    std::optional<size_t> start;
    std::optional<size_t> end;
    for (Operation *user : buffer.getUsers()) {
      Operation *blockLevelUser = body.findAncestorOpInBlock(*user);
      assert(blockLevelUser && "alloc escapes its IsolatedFromAbove domain");
      size_t userIdx = opIndices.lookup(blockLevelUser);
      auto dpsUser = dyn_cast<DestinationStyleOpInterface>(user);
      if (dpsUser && llvm::is_contained(dpsUser.getDpsInits(), buffer) &&
          (!start || userIdx < *start)) {
        start = userIdx;
      }
      if (!end || userIdx > *end) {
        end = userIdx;
      }
    }
    if (start) {
      lifetimes[buffer] = Lifetime{*start, *end};
    }
  }
  return lifetimes;
}

llvm::SmallVector<llvm::SmallVector<Value>>
greedyGrouping(Block &body, const llvm::DenseMap<Value, Lifetime> &lifetimes) {
  llvm::SmallVector<Value> allocs;
  for (Operation &op : body) {
    auto allocOp = dyn_cast<memref::AllocOp>(&op);
    if (!allocOp) {
      continue;
    }
    if (lifetimes.contains(allocOp.getResult())) {
      allocs.push_back(allocOp.getResult());
    }
  }
  llvm::stable_sort(allocs, [&](Value a, Value b) {
    return lifetimes.lookup(a).start < lifetimes.lookup(b).start;
  });

  auto overlaps = [](const Lifetime &a, const Lifetime &b) {
    return !(a.end < b.start || b.end < a.start);
  };

  llvm::SmallVector<llvm::SmallVector<Value>> groups;
  for (Value alloc : allocs) {
    Lifetime lifetime = lifetimes.lookup(alloc);
    bool placed = false;
    for (llvm::SmallVector<Value> &group : groups) {
      if (llvm::any_of(group, [&](Value member) {
            return overlaps(lifetimes.lookup(member), lifetime);
          })) {
        continue;
      }
      group.push_back(alloc);
      placed = true;
      break;
    }
    if (!placed) {
      groups.push_back({alloc});
    }
  }
  return groups;
}

Operation *
findInsertionPoint(Block &body,
                   const llvm::DenseMap<Value, Lifetime> &lifetimes) {
  Operation *point = nullptr;
  for (Operation &op : body) {
    auto allocOp = dyn_cast<memref::AllocOp>(&op);
    if (allocOp && lifetimes.contains(allocOp.getResult())) {
      point = &op;
    }
  }
  return point;
}

void emitPoolingReport(Block &body,
                       const llvm::DenseMap<Value, Lifetime> &lifetimes,
                       llvm::ArrayRef<llvm::SmallVector<Value>> groups,
                       Operation *insertionPoint) {
  size_t insertionIdx = 0;
  for (Operation &op : body) {
    if (&op == insertionPoint) {
      break;
    }
    ++insertionIdx;
  }
  body.getParentOp()->emitRemark()
      << "hipsr-pool-alloc: insertion point after op " << insertionIdx;

  llvm::DenseMap<Value, size_t> groupIndices;
  for (auto [groupIdx, group] : llvm::enumerate(groups)) {
    for (Value member : group) {
      groupIndices[member] = groupIdx;
    }
  }

  for (Operation &op : body) {
    auto allocOp = dyn_cast<memref::AllocOp>(&op);
    if (!allocOp) {
      continue;
    }
    auto it = lifetimes.find(allocOp.getResult());
    if (it == lifetimes.end()) {
      continue;
    }
    AllocationSize size = computeAllocationSize(allocOp);
    InFlightDiagnostic remark = allocOp.emitRemark();
    remark << "hipsr-pool-alloc: lifetime [" << it->second.start << ","
           << it->second.end << "] group "
           << groupIndices.lookup(allocOp.getResult()) << " size "
           << size.staticBytes;
    if (!size.dynamicDims.empty()) {
      remark << " x " << size.dynamicDims.size() << " dyn";
    }
  }
}

Value emitAllocationSize(OpBuilder &builder, Location loc,
                         memref::AllocOp allocOp) {
  AllocationSize size = computeAllocationSize(allocOp);
  Value bytes = arith::ConstantIndexOp::create(builder, loc, size.staticBytes);
  for (Value dyn : size.dynamicDims) {
    bytes = arith::MulIOp::create(builder, loc, bytes, dyn);
  }
  return bytes;
}

Value emitGroupSize(OpBuilder &builder, Location loc,
                    llvm::ArrayRef<Value> group, int64_t alignment) {
  llvm::SmallVector<Value> sizes;
  for (Value alloc : group) {
    sizes.push_back(emitAllocationSize(builder, loc,
                                       alloc.getDefiningOp<memref::AllocOp>()));
  }
  Value maxSize = sizes.front();
  for (Value size : llvm::ArrayRef<Value>(sizes).drop_front()) {
    maxSize = arith::MaxUIOp::create(builder, loc, maxSize, size);
  }
  Value alignVal = arith::ConstantIndexOp::create(builder, loc, alignment);
  Value alignMinus1 =
      arith::ConstantIndexOp::create(builder, loc, alignment - 1);
  Value numerator = arith::AddIOp::create(builder, loc, maxSize, alignMinus1);
  Value divided = arith::DivUIOp::create(builder, loc, numerator, alignVal);
  return arith::MulIOp::create(builder, loc, divided, alignVal);
}

Value findContext(Block &body) {
  for (BlockArgument arg : body.getArguments()) {
    if (isa<ContextType>(arg.getType())) {
      return arg;
    }
  }
  return nullptr;
}

Value emitPool(OpBuilder &builder, Location loc, Value ctx,
               llvm::ArrayRef<Value> groupSizes, uint64_t domainId) {
  Value poolSize = groupSizes.front();
  for (Value groupSize : groupSizes.drop_front()) {
    poolSize = arith::AddIOp::create(builder, loc, poolSize, groupSize);
  }
  auto deviceSpace =
      MemorySpaceAttr::get(builder.getContext(), MemorySpaceKind::Device);
  auto poolType = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                                  MemRefLayoutAttrInterface{}, deviceSpace);
  return GetPoolOp::create(builder, loc, poolType, ctx, poolSize, domainId);
}

llvm::SmallVector<Value> emitOffsets(OpBuilder &builder, Location loc,
                                     llvm::ArrayRef<Value> groupSizes) {
  llvm::SmallVector<Value> offsets;
  offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
  for (Value groupSize : groupSizes.drop_back()) {
    offsets.push_back(
        arith::AddIOp::create(builder, loc, offsets.back(), groupSize));
  }
  return offsets;
}

struct HipsrPoolAllocPass : impl::HipsrPoolAllocPassBase<HipsrPoolAllocPass> {
  using impl::HipsrPoolAllocPassBase<
      HipsrPoolAllocPass>::HipsrPoolAllocPassBase;

  void runOnOperation() override {
    getOperation().walk([&](PoolDomainOp domain) {
      Block &body = domain.getBody().front();
      llvm::DenseMap<Value, Lifetime> lifetimes = computeLiveness(body);
      Operation *insertionPoint = findInsertionPoint(body, lifetimes);
      if (!insertionPoint) {
        domain.emitError(
            "hipsr-pool-alloc: pool_domain has no poolable allocation");
        signalPassFailure();
        return;
      }
      Value ctx = findContext(body);
      if (!ctx) {
        domain.emitError("hipsr-pool-alloc: pool_domain has no context");
        signalPassFailure();
        return;
      }
      llvm::SmallVector<llvm::SmallVector<Value>> groups =
          greedyGrouping(body, lifetimes);
      if (emitPoolReport) {
        emitPoolingReport(body, lifetimes, groups, insertionPoint);
      }

      OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(insertionPoint);
      llvm::SmallVector<Value> groupSizes;
      for (llvm::ArrayRef<Value> group : groups) {
        groupSizes.push_back(
            emitGroupSize(builder, domain.getLoc(), group, kPoolAlignment));
      }
      emitPool(builder, domain.getLoc(), ctx, groupSizes, domain.getDomainId());
      emitOffsets(builder, domain.getLoc(), groupSizes);
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
