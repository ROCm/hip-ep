/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PartitionPoolDomains.cpp - Outline hipsr pool domains --------------===//
//
// Greedily partitions the non-terminator ops of a single-block function into
// standard IsolatedFromAbove domains. An active start barrier begins a domain;
// the first operation using an active end-barrier result begins the next.
// Top-level placeholders are validated before planning and excluded from
// boundary analysis.
//
// In the example, example.end_barrier implements EndBarrierInterface, and
// hipsr.example_start_barrier implements both StartBarrierInterface and
// DestinationStyleOpInterface with %init as its DPS init operand.
//
// Before:
//   func.func @graph(%arg: tensor<?xf32>) -> tensor<?xf32> {
//     %init = hipsr.placeholder : tensor<?xf32>
//     %0 = "example.end_barrier"(%arg)
//         : (tensor<?xf32>) -> tensor<?xf32>
//     %1 = "example.compute"(%0) : (tensor<?xf32>) -> tensor<?xf32>
//     %2 = "hipsr.example_start_barrier"(%1, %init)
//         : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
//     return %2 : tensor<?xf32>
//   }
//
// After:
//   func.func @graph(%arg: tensor<?xf32>) -> tensor<?xf32> {
//     %0 = hipsr.pool_domain(%arg : tensor<?xf32>) {
//     ^bb0(%domain_arg: tensor<?xf32>):
//       %3 = "example.end_barrier"(%domain_arg)
//           : (tensor<?xf32>) -> tensor<?xf32>
//       hipsr.pool_domain_yield %3 : tensor<?xf32>
//     } -> tensor<?xf32>
//     %1 = hipsr.pool_domain(%0 : tensor<?xf32>) {
//     ^bb0(%domain_arg: tensor<?xf32>):
//       %3 = "example.compute"(%domain_arg)
//           : (tensor<?xf32>) -> tensor<?xf32>
//       hipsr.pool_domain_yield %3 : tensor<?xf32>
//     } -> tensor<?xf32>
//     %2 = hipsr.pool_domain(%1 : tensor<?xf32>) {
//     ^bb0(%domain_arg: tensor<?xf32>):
//       %init = hipsr.placeholder : tensor<?xf32>
//       %3 = "hipsr.example_start_barrier"(%domain_arg, %init)
//           : (tensor<?xf32>, tensor<?xf32>) -> tensor<?xf32>
//       hipsr.pool_domain_yield %3 : tensor<?xf32>
//     } -> tensor<?xf32>
//     return %2 : tensor<?xf32>
//   }
//
// A read-only preflight validates each placeholder. Domain planning then
// considers only non-placeholder operations. Materialization moves those
// operations into their domains and derives each consumer's placeholders from
// its DPS init operands. It recreates the placeholders immediately before the
// consumer and erases the originals. MLIR's region-isolation utility then
// derives the operands and entry-block arguments.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.h"
#include "hip/Dialect/Hipsr/IR/HipsrPlaceholderOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrPoolDomainYieldOp.h"
#include "hip/Dialect/Hipsr/IR/HipsrStartBarrierInterface.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseSet.h"
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
  Operation *insertionPoint = nullptr;
  SmallVector<Operation *> operations;
  SmallVector<Value> results;
};

static bool isActiveStartBarrier(Operation &operation) {
  auto barrier = dyn_cast<StartBarrierInterface>(operation);
  return barrier && barrier.isStartBarrier();
}

static bool isActiveEndBarrier(Operation &operation) {
  auto barrier = dyn_cast<EndBarrierInterface>(operation);
  return barrier && barrier.isEndBarrier();
}

static bool usesAnyEndBarrierResult(
    Operation &operation,
    const llvm::SmallDenseSet<Value, 8> &endBarrierResults) {
  if (llvm::any_of(operation.getOperands(), [&](Value operand) {
        return endBarrierResults.contains(operand);
      })) {
    return true;
  }

  // A region-bearing op can consume a value through an implicit nested capture
  // even when that value is absent from the parent op's operand list.
  bool hasNestedUse = false;
  visitUsedValuesDefinedAbove(operation.getRegions(), [&](OpOperand *operand) {
    hasNestedUse |= endBarrierResults.contains(operand->get());
  });
  return hasNestedUse;
}

static bool isHipsrDpsOp(Operation &operation) {
  return operation.getName().getDialectNamespace() ==
             HipsrDialect::getDialectNamespace() &&
         isa<DestinationStyleOpInterface>(operation);
}

static LogicalResult verifyNoNestedPlaceholders(Operation &operation) {
  WalkResult result = operation.walk([&](PlaceholderOp placeholder) {
    placeholder.emitOpError("must be top-level when partitioning pool domains");
    return WalkResult::interrupt();
  });
  return result.wasInterrupted() ? failure() : success();
}

static LogicalResult verifyDedicatedPlaceholder(PlaceholderOp placeholder,
                                                Block &block) {
  if (placeholder->getNumResults() == 0) {
    placeholder.emitOpError("must produce at least one tensor DPS init");
    return failure();
  }

  Operation *consumer = nullptr;
  for (Value result : placeholder->getResults()) {
    if (!result.hasOneUse()) {
      placeholder.emitOpError("requires each result to have exactly one use");
      return failure();
    }

    OpOperand &use = *result.getUses().begin();
    auto dpsOp = dyn_cast<DestinationStyleOpInterface>(use.getOwner());
    if (!dpsOp || !isHipsrDpsOp(*use.getOwner()) || !dpsOp.isDpsInit(&use) ||
        use.getOwner()->getBlock() != &block ||
        use.getOwner() == block.getTerminator()) {
      placeholder.emitOpError(
          "requires every result to be a DPS init of one top-level hipsr "
          "operation");
      return failure();
    }
    if (consumer && consumer != use.getOwner()) {
      placeholder.emitOpError(
          "requires all results to initialize the same hipsr operation");
      return failure();
    }
    consumer = use.getOwner();
  }

  return success();
}

static LogicalResult verifyDpsInitPlaceholders(Operation &operation,
                                               Block &block) {
  if (!isHipsrDpsOp(operation)) {
    return success();
  }

  auto dpsOp = cast<DestinationStyleOpInterface>(operation);
  for (Value init : dpsOp.getDpsInits()) {
    if (!isa<TensorType>(init.getType())) {
      continue;
    }
    auto placeholder = init.getDefiningOp<PlaceholderOp>();
    if (!placeholder || placeholder->getBlock() != &block) {
      operation.emitOpError(
          "requires each tensor DPS init to be produced by a top-level "
          "hipsr.placeholder");
      return failure();
    }
  }

  return success();
}

static LogicalResult validatePlaceholders(Block &block) {
  for (Operation &operation : block.without_terminator()) {
    if (isa<PoolDomainOp>(operation)) {
      operation.emitError(
          "hipsr-partition-pool-domains does not support existing pool "
          "domains");
      return failure();
    }

    if (auto placeholder = dyn_cast<PlaceholderOp>(operation)) {
      if (failed(verifyDedicatedPlaceholder(placeholder, block))) {
        return failure();
      }
      continue;
    }

    if (failed(verifyNoNestedPlaceholders(operation))) {
      return failure();
    }
    if (failed(verifyDpsInitPlaceholders(operation, block))) {
      return failure();
    }
  }

  return success();
}

static void computeDomainResults(Domain &domain, Block &parentBlock) {
  llvm::SmallPtrSet<Operation *, 16> domainOperations(domain.operations.begin(),
                                                      domain.operations.end());

  for (Operation *operation : domain.operations) {
    for (Value result : operation->getResults()) {
      bool escapes = llvm::any_of(result.getUsers(), [&](Operation *user) {
        Operation *ancestor = parentBlock.findAncestorOpInBlock(*user);
        return !ancestor || !domainOperations.contains(ancestor);
      });
      if (escapes) {
        domain.results.push_back(result);
      }
    }
  }
}

static SmallVector<Domain> partitionIntoDomains(Block &block) {
  SmallVector<Domain> domains;
  llvm::SmallDenseSet<Value, 8> endBarrierResults;

  // TODO: When concrete barrier ops land, shape materialization must emit
  // start-barrier input readbacks at domain entry and end-barrier capacity
  // shapes before execution plus real shapes afterward. This pass only
  // establishes the boundaries required by those computations.
  for (Operation &operation : block.without_terminator()) {
    if (isa<PlaceholderOp>(operation)) {
      continue;
    }

    if (domains.empty() ||
        usesAnyEndBarrierResult(operation, endBarrierResults) ||
        isActiveStartBarrier(operation)) {
      domains.emplace_back();
      domains.back().insertionPoint = &operation;
      endBarrierResults.clear();
    }

    Domain &currentDomain = domains.back();
    currentDomain.operations.push_back(&operation);

    if (isActiveEndBarrier(operation)) {
      for (Value result : operation.getResults()) {
        endBarrierResults.insert(result);
      }
    }
  }

  for (Domain &domain : domains) {
    computeDomainResults(domain, block);
  }

  return domains;
}

static void recreatePlaceholders(IRRewriter &rewriter, const Domain &domain) {
  for (Operation *consumer : domain.operations) {
    if (!isHipsrDpsOp(*consumer)) {
      continue;
    }

    llvm::SmallSetVector<Operation *, 4> placeholders;
    auto dpsOp = cast<DestinationStyleOpInterface>(*consumer);
    for (Value init : dpsOp.getDpsInits()) {
      if (isa<TensorType>(init.getType())) {
        auto placeholder = cast<PlaceholderOp>(init.getDefiningOp());
        placeholders.insert(placeholder.getOperation());
      }
    }

    rewriter.setInsertionPoint(consumer);
    for (Operation *placeholder : placeholders) {
      Operation *replacement = rewriter.clone(*placeholder);
      rewriter.replaceOp(placeholder, replacement->getResults());
    }
  }
}

static void materializeDomains(ArrayRef<Domain> domains) {
  if (domains.empty()) {
    return;
  }

  IRRewriter rewriter(domains.front().insertionPoint->getContext());

  for (const Domain &domain : domains) {
    Operation *insertionPoint = domain.insertionPoint;
    rewriter.setInsertionPoint(insertionPoint);

    SmallVector<Type> resultTypes = llvm::map_to_vector(
        domain.results, [](Value result) { return result.getType(); });

    auto domainOp = rewriter.create<PoolDomainOp>(insertionPoint->getLoc(),
                                                  resultTypes, ValueRange{});
    Region &bodyRegion = domainOp.getBody();
    Block *body = rewriter.createBlock(&bodyRegion);
    for (Operation *operation : domain.operations) {
      rewriter.moveOpBefore(operation, body, body->end());
    }
    recreatePlaceholders(rewriter, domain);

    SmallVector<Value> capturedValues =
        makeRegionIsolatedFromAbove(rewriter, bodyRegion);
    rewriter.modifyOpInPlace(
        domainOp, [&] { domainOp->insertOperands(0, capturedValues); });

    Block &entryBlock = bodyRegion.front();
    // Ensure implicit captures in descendant regions are also remapped to the
    // entry arguments created by makeRegionIsolatedFromAbove.
    for (auto [capturedValue, argument] :
         llvm::zip_equal(capturedValues, entryBlock.getArguments())) {
      replaceAllUsesInRegionWith(capturedValue, argument, bodyRegion);
    }

    rewriter.setInsertionPointToEnd(&entryBlock);
    rewriter.create<PoolDomainYieldOp>(insertionPoint->getLoc(),
                                       domain.results);

    rewriter.replaceUsesWithIf(
        domain.results, domainOp.getResults(), [&](OpOperand &use) {
          return !domainOp->isProperAncestor(use.getOwner());
        });
  }
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
          "hipsr-partition-pool-domains only supports single-block functions");
      signalPassFailure();
      return;
    }

    Block &entryBlock = funcOp.front();
    if (failed(validatePlaceholders(entryBlock))) {
      signalPassFailure();
      return;
    }

    SmallVector<Domain> domains = partitionIntoDomains(entryBlock);
    materializeDomains(domains);
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
