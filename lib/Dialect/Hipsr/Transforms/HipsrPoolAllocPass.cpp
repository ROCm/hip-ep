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

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <utility>

#define DEBUG_TYPE "hipsr-pool-alloc"
#define DBGS() (llvm::dbgs() << "[" DEBUG_TYPE "] ")

STATISTIC(NumDomainsProcessed, "Number of pool domains processed");
STATISTIC(NumAllocationsPooled, "Number of allocations backed by a pool view");
STATISTIC(NumGroupsCreated, "Number of pool groups created");
STATISTIC(NumAllocationsReused,
          "Number of pool slots saved by reusing a group");

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

struct MeasuredSizes {
  int64_t maxBytes;
  int64_t sumBytes;
  size_t numDynamicDims;
};

std::optional<MeasuredSizes> measureSizes(llvm::ArrayRef<Value> allocs) {
  AllocationSize first =
      computeAllocationSize(allocs.front().getDefiningOp<memref::AllocOp>());
  MeasuredSizes measured{first.staticBytes, first.staticBytes,
                         first.dynamicDims.size()};
  for (Value alloc : allocs.drop_front()) {
    AllocationSize size =
        computeAllocationSize(alloc.getDefiningOp<memref::AllocOp>());
    if (size.dynamicDims != first.dynamicDims) {
      return std::nullopt;
    }
    measured.maxBytes = std::max(measured.maxBytes, size.staticBytes);
    measured.sumBytes += size.staticBytes;
  }
  return measured;
}

int64_t percentOf(int64_t part, int64_t whole) {
  return whole == 0 ? 0 : part * 100 / whole;
}

llvm::DenseMap<Value, size_t> mapAllocIndices(Block &body) {
  llvm::DenseMap<Value, size_t> indices;
  for (auto [opIdx, op] : llvm::enumerate(body)) {
    if (auto allocOp = dyn_cast<memref::AllocOp>(&op)) {
      indices[allocOp.getResult()] = opIdx;
    }
  }
  return indices;
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
    Operation *firstWrite = nullptr;
    Operation *lastUse = nullptr;
    for (Operation *user : buffer.getUsers()) {
      Operation *blockLevelUser = body.findAncestorOpInBlock(*user);
      assert(blockLevelUser && "alloc escapes its IsolatedFromAbove domain");
      size_t userIdx = opIndices.lookup(blockLevelUser);
      if (llvm::is_contained(getHipsrDestinationOperands(user), buffer) &&
          (!start || userIdx < *start)) {
        start = userIdx;
        firstWrite = blockLevelUser;
      }
      if (!end || userIdx > *end) {
        end = userIdx;
        lastUse = blockLevelUser;
      }
    }
    if (start) {
      lifetimes[buffer] = Lifetime{*start, *end};
      LLVM_DEBUG(DBGS() << "alloc #" << opIndices.lookup(&op) << " lifetime ["
                        << *start << "," << *end << "]: first write "
                        << firstWrite->getName() << ", last use "
                        << lastUse->getName() << "\n");
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

  llvm::DenseMap<Value, size_t> allocIndices = mapAllocIndices(body);
  llvm::SmallVector<llvm::SmallVector<Value>> groups;
  for (Value alloc : allocs) {
    Lifetime lifetime = lifetimes.lookup(alloc);
    bool placed = false;
    for (auto [groupIdx, group] : llvm::enumerate(groups)) {
      Value *conflict = llvm::find_if(group, [&](Value member) {
        return overlaps(lifetimes.lookup(member), lifetime);
      });
      if (conflict != group.end()) {
        LLVM_DEBUG({
          Lifetime other = lifetimes.lookup(*conflict);
          DBGS() << "alloc #" << allocIndices.lookup(alloc) << " ["
                 << lifetime.start << "," << lifetime.end
                 << "] conflicts with alloc #" << allocIndices.lookup(*conflict)
                 << " [" << other.start << "," << other.end << "] in group "
                 << groupIdx << "\n";
        });
        continue;
      }
      group.push_back(alloc);
      placed = true;
      break;
    }
    if (!placed) {
      LLVM_DEBUG(DBGS() << "alloc #" << allocIndices.lookup(alloc)
                        << " opens group " << groups.size() << "\n");
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

void logGroupSizes(Block &body,
                   llvm::ArrayRef<llvm::SmallVector<Value>> groups) {
  llvm::DenseMap<Value, size_t> allocIndices = mapAllocIndices(body);
  for (auto [groupIdx, group] : llvm::enumerate(groups)) {
    std::optional<MeasuredSizes> measured = measureSizes(group);
    if (!measured) {
      DBGS() << "group " << groupIdx
             << " max not comparable (mixed dynamic extents)\n";
      for (Value member : group) {
        AllocationSize size =
            computeAllocationSize(member.getDefiningOp<memref::AllocOp>());
        DBGS() << "  alloc #" << allocIndices.lookup(member) << " "
               << size.staticBytes << " bytes x " << size.dynamicDims.size()
               << " dyn\n";
      }
      continue;
    }
    DBGS() << "group " << groupIdx << " max " << measured->maxBytes << " bytes";
    if (measured->numDynamicDims != 0) {
      llvm::dbgs() << " x " << measured->numDynamicDims << " dyn";
    }
    llvm::dbgs() << "\n";
    for (Value member : group) {
      int64_t bytes =
          computeAllocationSize(member.getDefiningOp<memref::AllocOp>())
              .staticBytes;
      DBGS() << "  alloc #" << allocIndices.lookup(member) << " " << bytes
             << " bytes, " << (measured->maxBytes - bytes) << " unused\n";
    }
  }
}

void emitPoolingReport(PoolDomainOp domain,
                       llvm::ArrayRef<llvm::SmallVector<Value>> groups) {
  llvm::SmallVector<Value> allocs;
  for (llvm::ArrayRef<Value> group : groups) {
    allocs.append(group.begin(), group.end());
  }

  auto remark = [&] {
    InFlightDiagnostic diag = domain.emitRemark();
    diag << "hipsr-pool-alloc: domain " << domain.getDomainId() << ": ";
    return diag;
  };

  int64_t numAllocs = allocs.size();
  int64_t numGroups = groups.size();
  int64_t saved = numAllocs - numGroups;
  remark() << numAllocs << " allocs in " << numGroups << " groups, " << saved
           << " saved (" << percentOf(saved, numAllocs) << "%)";

  std::optional<MeasuredSizes> domainSizes = measureSizes(allocs);
  {
    InFlightDiagnostic diag = remark();
    diag << "before vs after: ";
    if (!domainSizes) {
      diag << "not comparable (mixed dynamic extents)";
    } else {
      int64_t after = 0;
      for (llvm::ArrayRef<Value> group : groups) {
        after += measureSizes(group)->maxBytes;
      }
      int64_t before = domainSizes->sumBytes;
      if (domainSizes->numDynamicDims == 0) {
        diag << before << " bytes vs " << after << " bytes, ";
      }
      diag << percentOf(before - after, before) << "% saved";
    }
  }

  for (auto [groupIdx, group] : llvm::enumerate(groups)) {
    InFlightDiagnostic diag = remark();
    diag << "group " << groupIdx << ": " << group.size() << " allocs, ";
    std::optional<MeasuredSizes> measured = measureSizes(group);
    if (!measured) {
      diag << "not comparable (mixed dynamic extents)";
      continue;
    }
    if (measured->numDynamicDims == 0) {
      diag << "max " << measured->maxBytes << " bytes, ";
    }
    int64_t capacity = measured->maxBytes * group.size();
    diag << percentOf(capacity - measured->sumBytes, capacity)
         << "% avg unused";
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

Value emitPool(OpBuilder &builder, Location loc, Value ctx, Value poolSize,
               uint64_t domainId) {
  auto deviceSpace =
      MemorySpaceAttr::get(builder.getContext(), MemorySpaceKind::Device);
  auto poolType = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                                  MemRefLayoutAttrInterface{}, deviceSpace);
  return GetPoolOp::create(builder, loc, poolType, ctx, poolSize, domainId);
}

std::pair<llvm::SmallVector<Value>, Value>
emitPoolLayout(OpBuilder &builder, Location loc,
               llvm::ArrayRef<Value> groupSizes) {
  llvm::SmallVector<Value> offsets;
  offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
  Value poolSize = groupSizes.front();
  for (Value groupSize : groupSizes.drop_front()) {
    offsets.push_back(poolSize);
    poolSize = arith::AddIOp::create(builder, loc, poolSize, groupSize);
  }
  return {offsets, poolSize};
}

void replaceAllocsWithViews(OpBuilder &builder, llvm::ArrayRef<Value> group,
                            Value pool, Value offset) {
  for (Value alloc : group) {
    auto allocOp = alloc.getDefiningOp<memref::AllocOp>();
    auto view =
        memref::ViewOp::create(builder, allocOp.getLoc(), allocOp.getType(),
                               pool, offset, allocOp.getDynamicSizes());
    allocOp.getResult().replaceAllUsesWith(view.getResult());
    allocOp.erase();
  }
}

struct HipsrPoolAllocPass : impl::HipsrPoolAllocPassBase<HipsrPoolAllocPass> {
  using impl::HipsrPoolAllocPassBase<
      HipsrPoolAllocPass>::HipsrPoolAllocPassBase;

  void runOnOperation() override {
    size_t domainOrdinal = 0;
    getOperation().walk([&](PoolDomainOp domain) {
      LLVM_DEBUG(DBGS() << "pool_domain #" << domainOrdinal << " (domain_id "
                        << domain.getDomainId() << ")\n");
      ++domainOrdinal;
      Block &body = domain.getBody().front();
      llvm::DenseMap<Value, Lifetime> lifetimes = computeLiveness(body);
      Operation *insertionPoint = findInsertionPoint(body, lifetimes);
      if (!insertionPoint) {
        domain.emitError(
            "hipsr-pool-alloc: pool_domain has no poolable allocation");
        signalPassFailure();
        return;
      }
      LLVM_DEBUG({
        size_t insertionIdx = 0;
        for (Operation &op : body) {
          if (&op == insertionPoint) {
            break;
          }
          ++insertionIdx;
        }
        DBGS() << "insertion point after op #" << insertionIdx << "\n";
      });
      Value ctx = findContext(body);
      if (!ctx) {
        domain.emitError("hipsr-pool-alloc: pool_domain has no context");
        signalPassFailure();
        return;
      }
      llvm::SmallVector<llvm::SmallVector<Value>> groups =
          greedyGrouping(body, lifetimes);
      LLVM_DEBUG(logGroupSizes(body, groups));
      ++NumDomainsProcessed;
      NumAllocationsPooled += lifetimes.size();
      NumGroupsCreated += groups.size();
      NumAllocationsReused += lifetimes.size() - groups.size();
      if (emitPoolReport) {
        emitPoolingReport(domain, groups);
      }

      OpBuilder builder(&getContext());
      builder.setInsertionPointAfter(insertionPoint);
      llvm::SmallVector<Value> groupSizes;
      for (llvm::ArrayRef<Value> group : groups) {
        groupSizes.push_back(
            emitGroupSize(builder, domain.getLoc(), group, kPoolAlignment));
      }
      auto [offsets, poolSize] =
          emitPoolLayout(builder, domain.getLoc(), groupSizes);
      Value pool = emitPool(builder, domain.getLoc(), ctx, poolSize,
                            domain.getDomainId());
      for (auto [group, offset] : llvm::zip_equal(groups, offsets)) {
        replaceAllocsWithViews(builder, group, pool, offset);
      }
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
