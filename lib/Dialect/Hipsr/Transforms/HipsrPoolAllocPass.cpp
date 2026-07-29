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
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#include <cstdint>
#include <limits>

// Distinct from the legacy hip-pool-allocs pass so --debug-only can select one.
#define DEBUG_TYPE "hipsr-pool-allocs"

STATISTIC(NumDomainsProcessed, "Number of pool domains processed");
STATISTIC(NumAllocationsPooled, "Number of allocations pooled into a domain");
STATISTIC(NumGroupsCreated, "Number of lifetime groups created");
STATISTIC(NumAllocationsReused,
          "Number of allocations sharing another allocation's pool space");

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRPOOLALLOCPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

constexpr int64_t kPoolAlignment = 256;

struct Lifetime {
  size_t start;
  size_t end;
  // Ops the endpoints resolve to; read only by the LLVM_DEBUG trace.
  Operation *startOp = nullptr;
  Operation *endOp = nullptr;
};

std::optional<Lifetime>
computeLifetime(Value buffer,
                const llvm::DenseMap<Operation *, size_t> &opIndices) {
  Lifetime lt{std::numeric_limits<size_t>::max(), 0};
  for (Operation *user : buffer.getUsers()) {
    size_t userIdx = opIndices.lookup(user);
    if (auto dpsOp = dyn_cast<DestinationStyleOpInterface>(user)) {
      for (Value init : dpsOp.getDpsInits()) {
        if (init == buffer) {
          if (userIdx < lt.start) {
            lt.start = userIdx;
            lt.startOp = user;
          }
          break;
        }
      }
    }
    if (!lt.endOp || userIdx > lt.end) {
      lt.end = userIdx;
      lt.endOp = user;
    }
  }
  if (lt.start == std::numeric_limits<size_t>::max()) {
    return std::nullopt;
  }
  return lt;
}

// Element bytes times every static dimension: the whole byte size when the
// shape is static, and the compile-time factor of the runtime size otherwise.
int64_t staticByteFactor(MemRefType memTy) {
  int64_t factor = memTy.getElementTypeBitWidth() / 8;
  for (int64_t dim : memTy.getShape()) {
    if (!ShapedType::isDynamic(dim)) {
      factor *= dim;
    }
  }
  return factor;
}

int64_t alignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) / alignment * alignment;
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
      continue;
    }
    // Without a DPS `outs` write there is no live range to place, which points
    // at an upstream pass leaving a dead alloc behind rather than at an input
    // this pass is expected to see.
    allocOp.emitWarning("hipsr-pool-alloc: allocation has no first-write or no "
                        "users, not pooled");
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
    for (size_t groupIdx = 0; groupIdx < groups.size(); ++groupIdx) {
      Value conflict;
      for (Value member : groups[groupIdx]) {
        if (overlaps(lifetimes.lookup(member), lt)) {
          conflict = member;
          break;
        }
      }
      if (!conflict) {
        groups[groupIdx].push_back(alloc);
        placed = true;
        break;
      }
      LLVM_DEBUG({
        Lifetime other = lifetimes.lookup(conflict);
        llvm::dbgs() << "  alloc " << alloc << " [" << lt.start << "," << lt.end
                     << "] conflicts with " << conflict << " [" << other.start
                     << "," << other.end << "] in group " << groupIdx << "\n";
      });
    }
    if (!placed) {
      groups.push_back({alloc});
    }
  }
  return groups;
}

Value findContext(Block &body) {
  Value ctx;
  for (BlockArgument arg : body.getArguments()) {
    if (isa<ContextType>(arg.getType())) {
      ctx = arg;
    }
  }
  return ctx;
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

llvm::SmallVector<Value>
emitGroupSizes(OpBuilder &builder, Location loc,
               llvm::ArrayRef<llvm::SmallVector<Value>> groups,
               int64_t alignment) {
  llvm::SmallVector<Value> groupSizes;
  for (llvm::ArrayRef<Value> group : groups) {
    llvm::SmallVector<Value> sizes;
    for (Value alloc : group) {
      auto allocOp = alloc.getDefiningOp<memref::AllocOp>();
      Value size = arith::ConstantIndexOp::create(
          builder, loc, staticByteFactor(cast<MemRefType>(allocOp.getType())));
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
    Value numerator =
        arith::AddIOp::create(builder, loc, groupSize, alignMinus1);
    Value divided = arith::DivUIOp::create(builder, loc, numerator, alignVal);
    groupSizes.push_back(
        arith::MulIOp::create(builder, loc, divided, alignVal));
  }
  return groupSizes;
}

Value emitPool(OpBuilder &builder, Location loc, Value ctx,
               llvm::ArrayRef<Value> groupSizes) {
  Value poolSize = groupSizes.front();
  for (Value gs : groupSizes.drop_front()) {
    poolSize = arith::AddIOp::create(builder, loc, poolSize, gs);
  }
  auto deviceSpace =
      MemorySpaceAttr::get(builder.getContext(), MemorySpaceKind::Device);
  auto poolType = MemRefType::get({ShapedType::kDynamic}, builder.getI8Type(),
                                  MemRefLayoutAttrInterface{}, deviceSpace);
  return GetPoolOp::create(builder, loc, poolType, ctx, poolSize);
}

// Byte totals are only meaningful when every size is a compile-time constant;
// a domain with a dynamic extent reports the structural counts alone rather
// than a symbolic expression.
void emitPoolingReport(PoolDomainOp domain, int64_t domainId,
                       llvm::ArrayRef<Value> allocs,
                       llvm::ArrayRef<llvm::SmallVector<Value>> groups,
                       int64_t alignment) {
  InFlightDiagnostic diag = domain.emitRemark();
  diag << "hipsr-pool-alloc: domain " << domainId << " pooled " << allocs.size()
       << " allocs into " << groups.size() << " groups (reused "
       << (allocs.size() - groups.size()) << ")";

  bool allStatic = llvm::all_of(allocs, [](Value alloc) {
    return cast<MemRefType>(alloc.getType()).hasStaticShape();
  });
  if (!allStatic) {
    return;
  }

  int64_t poolBytes = 0;
  int64_t naiveBytes = 0;
  int64_t slackBytes = 0;
  for (llvm::ArrayRef<Value> group : groups) {
    int64_t groupBytes = 0;
    for (Value member : group) {
      groupBytes = std::max(
          groupBytes, staticByteFactor(cast<MemRefType>(member.getType())));
    }
    poolBytes += alignUp(groupBytes, alignment);
    for (Value member : group) {
      int64_t memberBytes =
          staticByteFactor(cast<MemRefType>(member.getType()));
      // Naive baseline gives every alloc its own aligned buffer, so the ratio
      // compares like with like and can never make pooling look like a loss.
      naiveBytes += alignUp(memberBytes, alignment);
      slackBytes += groupBytes - memberBytes;
    }
  }
  int64_t savedPct =
      naiveBytes ? (naiveBytes - poolBytes) * 100 / naiveBytes : 0;
  diag << "; pool " << poolBytes << "/" << naiveBytes << " B (saved "
       << savedPct << "%), slack " << slackBytes << " B";
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
    int64_t domainId = 0;
    getOperation().walk(
        [&](PoolDomainOp domain) { processDomain(domain, domainId++); });
  }

  void processDomain(PoolDomainOp domain, int64_t domainId) {
    Block &body = domain.getBody().front();

    llvm::SmallVector<Value> allocs;
    llvm::DenseMap<Value, Lifetime> lifetimes;
    collectAllocLifetimes(body, allocs, lifetimes);
    if (allocs.empty()) {
      if (emitPoolReport) {
        domain.emitRemark() << "hipsr-pool-alloc: domain " << domainId
                            << " has no poolable allocations";
      }
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

    ++NumDomainsProcessed;
    NumAllocationsPooled += allocs.size();
    NumGroupsCreated += groups.size();
    NumAllocationsReused += allocs.size() - groups.size();

    LLVM_DEBUG({
      llvm::dbgs() << "domain " << domainId << ": " << allocs.size()
                   << " allocs, " << groups.size() << " groups\n";
      for (Value alloc : allocs) {
        Lifetime lt = lifetimes.lookup(alloc);
        llvm::dbgs() << "  " << alloc << " [" << lt.start << "," << lt.end
                     << "] first-write " << lt.startOp->getName()
                     << " last-use " << lt.endOp->getName() << "\n";
      }
      for (auto [groupIdx, group] : llvm::enumerate(groups)) {
        llvm::dbgs() << "  group " << groupIdx << ":";
        for (Value member : group) {
          llvm::dbgs() << " " << member;
        }
        llvm::dbgs() << "\n";
      }
    });

    // Emitted before the allocs are replaced, while their types are still live.
    if (emitPoolReport) {
      emitPoolingReport(domain, domainId, allocs, groups, kPoolAlignment);
    }

    OpBuilder builder(&getContext());
    builder.setInsertionPointAfter(findLastAlloc(body));
    Location loc = domain.getLoc();

    llvm::SmallVector<Value> groupSizes =
        emitGroupSizes(builder, loc, groups, kPoolAlignment);
    Value pool = emitPool(builder, loc, ctx, groupSizes);

    llvm::SmallVector<Value> offsets;
    offsets.push_back(arith::ConstantIndexOp::create(builder, loc, 0));
    for (size_t i = 1; i < groups.size(); ++i) {
      offsets.push_back(arith::AddIOp::create(builder, loc, offsets[i - 1],
                                              groupSizes[i - 1]));
    }

    for (auto [group, offset] : llvm::zip(groups, offsets)) {
      replaceAllocsWithViews(builder, group, pool, offset);
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
