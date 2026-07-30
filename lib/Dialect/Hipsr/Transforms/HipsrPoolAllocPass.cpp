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

#include <cstddef>
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

void emitPoolingReport(Block &body,
                       const llvm::DenseMap<Value, Lifetime> &lifetimes) {
  for (Operation &op : body) {
    auto allocOp = dyn_cast<memref::AllocOp>(&op);
    if (!allocOp) {
      continue;
    }
    auto it = lifetimes.find(allocOp.getResult());
    if (it == lifetimes.end()) {
      continue;
    }
    allocOp.emitRemark() << "hipsr-pool-alloc: lifetime [" << it->second.start
                         << "," << it->second.end << "]";
  }
}

struct HipsrPoolAllocPass : impl::HipsrPoolAllocPassBase<HipsrPoolAllocPass> {
  using impl::HipsrPoolAllocPassBase<
      HipsrPoolAllocPass>::HipsrPoolAllocPassBase;

  void runOnOperation() override {
    getOperation().walk([&](PoolDomainOp domain) {
      Block &body = domain.getBody().front();
      llvm::DenseMap<Value, Lifetime> lifetimes = computeLiveness(body);
      if (emitPoolReport) {
        emitPoolingReport(body, lifetimes);
      }
    });
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
