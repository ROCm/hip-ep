/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
#include "hip/Dialect/Hipsr/IR/HipsrStartBarrierInterface.h"
#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"
#include "mlir/Transforms/RegionUtils.h"

#include "llvm/ADT/DenseMap.h"
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

  // A nested region can use an end-barrier result even when its parent op has
  // no matching operand.
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

static LogicalResult verifyNoUnsupportedNestedOps(Operation &operation) {
  WalkResult result = operation.walk([&](Operation *nested) {
    if (nested == &operation) {
      return WalkResult::advance();
    }
    if (isa<PoolDomainOp>(nested)) {
      nested->emitError(
          "hipsr-partition-pool-domains does not support existing pool "
          "domains");
      return WalkResult::interrupt();
    }
    if (auto placeholder = dyn_cast<PlaceholderOp>(nested)) {
      placeholder.emitOpError(
          "must be top-level when partitioning pool domains");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result.wasInterrupted() ? failure() : success();
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

static LogicalResult validatePartitionInput(Block &block) {
  for (Operation &operation : block.without_terminator()) {
    if (isa<PoolDomainOp>(operation)) {
      operation.emitError(
          "hipsr-partition-pool-domains does not support existing pool "
          "domains");
      return failure();
    }

    if (auto placeholder = dyn_cast<PlaceholderOp>(operation)) {
      Operation *consumer = placeholder.getDpsConsumer();
      if (!consumer || consumer->getBlock() != &block) {
        placeholder.emitOpError(
            "requires its DPS consumer to be top-level when partitioning pool "
            "domains");
        return failure();
      }
      continue;
    }

    if (failed(verifyNoUnsupportedNestedOps(operation))) {
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
    if (!isHipsrDpsOp(*operation)) {
      continue;
    }
    auto dpsOp = cast<DestinationStyleOpInterface>(*operation);
    for (Value init : dpsOp.getDpsInits()) {
      if (auto placeholder = init.getDefiningOp<PlaceholderOp>()) {
        domainOperations.insert(placeholder);
      }
    }
  }

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

static OpResult getConsumerResult(PlaceholderOp placeholder,
                                  Value placeholderResult) {
  Operation *consumer = placeholder.getDpsConsumer();
  auto dpsConsumer = cast<DestinationStyleOpInterface>(consumer);
  for (OpOperand &use : placeholderResult.getUses()) {
    if (use.getOwner() == consumer && dpsConsumer.isDpsInit(&use)) {
      return dpsConsumer.getTiedOpResult(&use);
    }
  }
  llvm_unreachable("verified placeholder has no DPS init use");
}

// A placeholder edge that crosses a domain boundary becomes a block argument
// backed by the matching data result:
//
// Before: %prev = hipsr.cast ... outs(%prev_init)
//         %next_init = hipsr.placeholder ... ins(%prev_init)
// After:  %prev = hipsr.cast ... outs(%prev_init)
//         %next_init = hipsr.placeholder ... ins(%prev)
//
// makeRegionIsolatedFromAbove then replaces %prev with the destination
// domain's block argument.
static void
redirectCrossDomainShapeDependencies(MutableArrayRef<Domain> domains) {
  llvm::DenseMap<Operation *, Domain *> operationDomains;
  for (Domain &domain : domains) {
    for (Operation *operation : domain.operations) {
      operationDomains.insert({operation, &domain});
    }
  }

  llvm::SmallPtrSet<Operation *, 16> visitedPlaceholders;
  for (Domain &domain : domains) {
    for (Operation *operation : domain.operations) {
      if (!isHipsrDpsOp(*operation)) {
        continue;
      }
      auto dpsOp = cast<DestinationStyleOpInterface>(*operation);
      for (Value init : dpsOp.getDpsInits()) {
        auto placeholder = init.getDefiningOp<PlaceholderOp>();
        if (!placeholder || !visitedPlaceholders.insert(placeholder).second) {
          continue;
        }

        for (Value result : placeholder.getResults()) {
          OpResult consumerResult = getConsumerResult(placeholder, result);
          for (OpOperand &use : llvm::make_early_inc_range(result.getUses())) {
            auto dependent = dyn_cast<PlaceholderOp>(use.getOwner());
            if (!dependent || operationDomains.lookup(
                                  dependent.getDpsConsumer()) == &domain) {
              continue;
            }
            use.set(consumerResult);
          }
        }
      }
    }
  }
}

static SmallVector<Domain> partitionIntoDomains(Block &block) {
  SmallVector<Domain> domains;
  llvm::SmallDenseSet<Value, 8> endBarrierResults;

  for (Operation &operation : block.without_terminator()) {
    // Placeholders do not set boundaries. movePlaceholders() handles them
    // later.
    if (isa<PlaceholderOp>(operation)) {
      continue;
    }

    if (domains.empty() ||
        usesAnyEndBarrierResult(operation, endBarrierResults) ||
        isActiveStartBarrier(operation)) {
      domains.emplace_back();
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

  redirectCrossDomainShapeDependencies(domains);
  for (Domain &domain : domains) {
    computeDomainResults(domain, block);
  }

  return domains;
}

static void movePlaceholders(IRRewriter &rewriter, const Domain &domain) {
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

    for (Operation *placeholder : placeholders) {
      rewriter.moveOpBefore(placeholder, consumer);
    }
  }
}

static void putContextFirst(Value context,
                            SmallVectorImpl<Value> &capturedValues,
                            Block &entryBlock) {
  if (!context) {
    return;
  }

  auto contextIt = llvm::find(capturedValues, context);
  if (contextIt != capturedValues.end() &&
      contextIt == capturedValues.begin()) {
    return;
  }

  BlockArgument contextArgument =
      entryBlock.insertArgument(0u, context.getType(), context.getLoc());
  if (contextIt != capturedValues.end()) {
    unsigned contextIndex = contextIt - capturedValues.begin();
    entryBlock.getArgument(contextIndex + 1)
        .replaceAllUsesWith(contextArgument);
    entryBlock.eraseArgument(contextIndex + 1);
    capturedValues.erase(contextIt);
  }
  capturedValues.insert(capturedValues.begin(), context);
}

static void materializeDomains(ArrayRef<Domain> domains, Value context) {
  if (domains.empty()) {
    return;
  }

  IRRewriter rewriter(domains.front().operations.front()->getContext());

  for (const Domain &domain : domains) {
    Operation *insertionPoint = domain.operations.front();
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
    movePlaceholders(rewriter, domain);

    SmallVector<Value> capturedValues =
        makeRegionIsolatedFromAbove(rewriter, bodyRegion);
    Block &entryBlock = bodyRegion.front();
    putContextFirst(context, capturedValues, entryBlock);
    rewriter.modifyOpInPlace(
        domainOp, [&] { domainOp->insertOperands(0, capturedValues); });

    // makeRegionIsolatedFromAbove updates uses only in this region's blocks.
    // Update uses in nested regions too.
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
    if (failed(validatePartitionInput(entryBlock))) {
      signalPassFailure();
      return;
    }

    SmallVector<Domain> domains = partitionIntoDomains(entryBlock);
    Value context;
    if (entryBlock.getNumArguments() > 0 &&
        isa<ContextType>(entryBlock.getArgument(0).getType())) {
      context = entryBlock.getArgument(0);
    }
    materializeDomains(domains, context);
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
