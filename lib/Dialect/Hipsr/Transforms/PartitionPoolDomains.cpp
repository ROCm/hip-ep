/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PartitionPoolDomains.cpp - Outline hipsr pool domains --------------===//
//
// Phase one wraps the non-terminator ops of a barrier-free, single-block
// function in one standard IsolatedFromAbove domain.
//
// Before:
//   func.func @graph(%arg: tensor<?xf32>) -> tensor<?xf32> {
//     %0 = "example.compute"(%arg) : (tensor<?xf32>) -> tensor<?xf32>
//     return %0 : tensor<?xf32>
//   }
//
// After:
//   func.func @graph(%arg: tensor<?xf32>) -> tensor<?xf32> {
//     %0 = hipsr.pool_domain(%arg : tensor<?xf32>) {
//     ^bb0(%domain_arg: tensor<?xf32>):
//       %1 = "example.compute"(%domain_arg)
//           : (tensor<?xf32>) -> tensor<?xf32>
//       hipsr.pool_domain_yield %1 : tensor<?xf32>
//     } -> tensor<?xf32>
//     return %0 : tensor<?xf32>
//   }
//
// Compute every domain's operands and results before cloning so rewrites cannot
// change their order.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrStartBarrierInterface.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_PARTITIONPOOLDOMAINSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct Domain {
  SmallVector<Operation *> operations;
  llvm::SetVector<Value> operands;
  SmallVector<Value> results;
};

static Operation *findAncestorInBlock(Operation *operation, Block &block) {
  if (operation->getBlock() == &block) {
    return operation;
  }
  return block.findAncestorOpInBlock(*operation);
}

static bool isOperationInsideDomain(
    Operation *operation, Block &entryBlock,
    const llvm::SmallPtrSetImpl<Operation *> &domainOperations) {
  Operation *ancestor = findAncestorInBlock(operation, entryBlock);
  return ancestor && domainOperations.contains(ancestor);
}

static bool isValueDefinedInsideDomain(
    Value value, Block &entryBlock,
    const llvm::SmallPtrSetImpl<Operation *> &domainOperations) {
  if (Operation *definingOp = value.getDefiningOp()) {
    return isOperationInsideDomain(definingOp, entryBlock, domainOperations);
  }

  auto blockArgument = dyn_cast<BlockArgument>(value);
  if (!blockArgument || blockArgument.getOwner() == &entryBlock) {
    return false;
  }

  Operation *parentOp = blockArgument.getOwner()->getParentOp();
  return parentOp &&
         isOperationInsideDomain(parentOp, entryBlock, domainOperations);
}

static void computeDomainBoundary(Domain &domain, Block &parentBlock) {
  llvm::SmallPtrSet<Operation *, 16> domainOperations(domain.operations.begin(),
                                                      domain.operations.end());
  auto recordExternalOperand = [&](Value value) {
    if (!isValueDefinedInsideDomain(value, parentBlock, domainOperations))
      domain.operands.insert(value);
  };

  for (Operation *operation : domain.operations) {
    for (Value operand : operation->getOperands())
      recordExternalOperand(operand);

    // Values captured only by nested regions may not appear among the parent
    // op's operands, but isolation still requires them at the domain boundary.
    if (operation->getNumRegions() != 0) {
      llvm::SetVector<Value> nestedCaptures;
      getUsedValuesDefinedAbove(operation->getRegions(), nestedCaptures);
      for (Value capture : nestedCaptures)
        recordExternalOperand(capture);
    }
  }

  for (Operation *operation : domain.operations) {
    for (Value result : operation->getResults()) {
      bool escapes = llvm::any_of(result.getUses(), [&](OpOperand &use) {
        return !isOperationInsideDomain(use.getOwner(), parentBlock,
                                        domainOperations);
      });
      if (escapes)
        domain.results.push_back(result);
    }
  }
}

static FailureOr<SmallVector<Domain>> partitionIntoDomains(Block &block) {
  SmallVector<Domain> domains;
  Domain currentDomain;

  for (Operation &operation : block) {
    if (operation.hasTrait<OpTrait::IsTerminator>()) {
      continue;
    }

    if (isa<PoolDomainOp>(operation)) {
      operation.emitError(
          "hipsr-partition-pool-domains does not support existing pool "
          "domains");
      return failure();
    }

    // TODO: Replace this phase-one rejection with boundary splitting once
    // barrier operations have defined domain semantics.
    if (auto startBarrier = dyn_cast<StartBarrierInterface>(operation);
        startBarrier && startBarrier.isStartBarrier()) {
      operation.emitError(
          "hipsr-partition-pool-domains: active start barriers are not "
          "supported in phase 1");
      return failure();
    }
    if (auto endBarrier = dyn_cast<EndBarrierInterface>(operation);
        endBarrier && endBarrier.isEndBarrier()) {
      operation.emitError(
          "hipsr-partition-pool-domains: active end barriers are not "
          "supported in phase 1");
      return failure();
    }

    currentDomain.operations.push_back(&operation);
  }

  if (!currentDomain.operations.empty())
    domains.push_back(std::move(currentDomain));

  for (Domain &domain : domains)
    computeDomainBoundary(domain, block);

  return domains;
}

static void materializeDomains(ArrayRef<Domain> domains) {
  if (domains.empty()) {
    return;
  }

  IRRewriter rewriter(domains.front().operations.front()->getContext());
  // Keep originals alive while every plan is cloned because a later domain
  // may consume a result replaced by an earlier domain.
  IRMapping globalMapping;
  SmallVector<std::pair<Value, Value>> resultReplacements;

  for (const Domain &domain : domains) {
    Operation *firstOperation = domain.operations.front();
    rewriter.setInsertionPoint(firstOperation);

    SmallVector<Value> resolvedOperands;
    resolvedOperands.reserve(domain.operands.size());
    for (Value operand : domain.operands)
      resolvedOperands.push_back(globalMapping.lookupOrDefault(operand));

    SmallVector<Type> resultTypes;
    resultTypes.reserve(domain.results.size());
    for (Value result : domain.results)
      resultTypes.push_back(result.getType());

    auto domainOp = rewriter.create<PoolDomainOp>(
        firstOperation->getLoc(), resultTypes, resolvedOperands);
    Block *body = rewriter.createBlock(&domainOp.getBody());

    IRMapping localMapping;
    for (Value operand : domain.operands) {
      BlockArgument argument =
          body->addArgument(operand.getType(), operand.getLoc());
      localMapping.map(operand, argument);
    }

    rewriter.setInsertionPointToStart(body);
    for (Operation *operation : domain.operations)
      rewriter.clone(*operation, localMapping);

    SmallVector<Value> yieldedValues;
    yieldedValues.reserve(domain.results.size());
    for (Value result : domain.results)
      yieldedValues.push_back(localMapping.lookup(result));
    rewriter.create<PoolDomainYieldOp>(firstOperation->getLoc(), yieldedValues);

    for (auto [original, replacement] :
         llvm::zip_equal(domain.results, domainOp.getResults())) {
      globalMapping.map(original, replacement);
      resultReplacements.emplace_back(original, replacement);
    }
  }

  for (auto [original, replacement] : resultReplacements)
    rewriter.replaceAllUsesWith(original, replacement);

  for (const Domain &domain : llvm::reverse(domains))
    for (Operation *operation : llvm::reverse(domain.operations))
      rewriter.eraseOp(operation);
}

struct PartitionPoolDomainsPass
    : impl::PartitionPoolDomainsPassBase<PartitionPoolDomainsPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();

    if (funcOp.isDeclaration()) {
      return;
    }

    if (!funcOp.getBody().hasOneBlock()) {
      funcOp.emitError(
          "hipsr-partition-pool-domains only supports single-block functions "
          "in phase 1");
      signalPassFailure();
      return;
    }

    Block &entryBlock = funcOp.front();
    FailureOr<SmallVector<Domain>> domains = partitionIntoDomains(entryBlock);
    if (failed(domains)) {
      signalPassFailure();
      return;
    }

    materializeDomains(*domains);
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
