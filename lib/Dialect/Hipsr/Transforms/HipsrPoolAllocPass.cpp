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

void emitPoolingReport(Block &body,
                       const llvm::DenseMap<Value, Lifetime> &lifetimes,
                       llvm::ArrayRef<llvm::SmallVector<Value>> groups) {
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

struct HipsrPoolAllocPass : impl::HipsrPoolAllocPassBase<HipsrPoolAllocPass> {
  using impl::HipsrPoolAllocPassBase<
      HipsrPoolAllocPass>::HipsrPoolAllocPassBase;

  void runOnOperation() override {
    getOperation().walk([&](PoolDomainOp domain) {
      Block &body = domain.getBody().front();
      llvm::DenseMap<Value, Lifetime> lifetimes = computeLiveness(body);
      llvm::SmallVector<llvm::SmallVector<Value>> groups =
          greedyGrouping(body, lifetimes);
      if (emitPoolReport) {
        emitPoolingReport(body, lifetimes, groups);
      }
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
