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
#include "llvm/Support/ErrorHandling.h"

#include <cstdint>
#include <utility>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_PARTITIONPOOLDOMAINSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {
namespace partition_analysis {

struct Domain {
  llvm::SmallVector<Operation *> operations;
  llvm::SmallVector<OpResult> results;
};

using DomainId = std::uint64_t;
using OperationDomains = llvm::DenseMap<Operation *, DomainId>;
using DependencyDomainIds = llvm::SmallDenseSet<DomainId, 4>;

struct DomainAssignment {
  llvm::SmallVector<Domain> domains;
  OperationDomains operationDomains;
};

// Find the domain of each operation that defines an input value.
DependencyDomainIds
collectOperandDependencyDomainIds(const ValueRange &operands,
                                  const OperationDomains &operationDomains) {
  DependencyDomainIds dependencyDomainIds;
  for (Value operand : operands) {
    if (Operation *definingOp = operand.getDefiningOp()) {
      auto dependency = operationDomains.find(definingOp);
      if (dependency == operationDomains.end()) {
        llvm::report_fatal_error(
            "pool-domain analysis expected an assigned dependency");
      }
      dependencyDomainIds.insert(dependency->second);
    }
  }
  return dependencyDomainIds;
}

// Use the highest input domain. A barrier moves its branch to the next domain,
// while other branches can stay in the same domain.
DomainId
computeOperationDomainId(const Operation &operation,
                         const DependencyDomainIds &dependencyDomainIds) {
  DomainId domainId = 0;
  if (!dependencyDomainIds.empty()) {
    domainId = *llvm::max_element(dependencyDomainIds);
    if (auto placeholder = dyn_cast<PlaceholderOp>(operation);
        placeholder &&
        placeholder.getPlaceholderType() == PlaceholderType::Barrier) {
      ++domainId;
    }
  }
  return domainId;
}

// Visit in block order so each operation is assigned before its users.
DomainAssignment assignOperationsToDomains(Block &block) {
  DomainAssignment assignment;
  for (Operation &operation : block.without_terminator()) {
    DependencyDomainIds dependencyDomainIds = collectOperandDependencyDomainIds(
        operation.getOperands(), assignment.operationDomains);
    DomainId domainId =
        computeOperationDomainId(operation, dependencyDomainIds);
    if (domainId > assignment.domains.size()) {
      llvm::report_fatal_error(
          "pool-domain analysis produced a non-contiguous domain");
    }
    if (domainId == assignment.domains.size()) {
      assignment.domains.emplace_back();
    }
    assignment.domains[domainId].operations.push_back(&operation);
    assignment.operationDomains.try_emplace(&operation, domainId);
  }
  return assignment;
}

// Check whether another domain or the function return uses this result.
bool isResultUsedOutsideDomain(Value result, DomainId domainId, Block &block,
                               const DomainAssignment &assignment) {
  return llvm::any_of(result.getUsers(), [&](Operation *user) {
    Operation *topLevelUser = block.findAncestorOpInBlock(*user);
    if (!topLevelUser) {
      llvm::report_fatal_error(
          "pool-domain analysis found a result use outside the function block");
    }
    auto userDomain = assignment.operationDomains.find(topLevelUser);
    return userDomain == assignment.operationDomains.end() ||
           userDomain->second != domainId;
  });
}

// Find the non-placeholder results that this domain must yield.
llvm::SmallVector<OpResult>
collectDomainResults(Block &block, DomainId domainId, const Domain &domain,
                     const DomainAssignment &assignment) {
  llvm::SmallVector<OpResult> results;
  for (Operation *operation : domain.operations) {
    if (isa<PlaceholderOp>(operation)) {
      continue;
    }
    for (OpResult result : operation->getResults()) {
      if (isResultUsedOutsideDomain(result, domainId, block, assignment)) {
        results.push_back(result);
      }
    }
  }
  return results;
}

// Assign all operations first, then find the results each domain must yield.
DomainAssignment buildDomainAssignment(Block &block) {
  DomainAssignment assignment = assignOperationsToDomains(block);
  for (auto [domainId, domain] : llvm::enumerate(assignment.domains)) {
    domain.results = collectDomainResults(block, domainId, domain, assignment);
  }
  return assignment;
}

// Print each operation's domain and each domain's operations and results.
void emitAnalysisReport(Block &block, const DomainAssignment &assignment) {
  llvm::DenseMap<Operation *, unsigned> operationIndices;
  for (auto [index, operation] : llvm::enumerate(block.without_terminator())) {
    operationIndices[&operation] = index;
  }

  InFlightDiagnostic operationReport = block.getParentOp()->emitRemark();
  operationReport << "hipsr-partition-pool-domains: operation domains [";
  for (auto [index, operation] : llvm::enumerate(block.without_terminator())) {
    if (index != 0) {
      operationReport << ",";
    }
    auto domain = assignment.operationDomains.find(&operation);
    if (domain == assignment.operationDomains.end()) {
      llvm::report_fatal_error(
          "pool-domain analysis expected an assigned operation");
    }
    operationReport << index << "->" << domain->second;
  }
  operationReport << "]";

  for (auto [domainId, domain] : llvm::enumerate(assignment.domains)) {
    InFlightDiagnostic report = domain.operations.front()->emitRemark();
    report << "hipsr-partition-pool-domains: domain " << domainId << " ops [";
    for (auto [index, operation] : llvm::enumerate(domain.operations)) {
      if (index != 0) {
        report << ",";
      }
      report << operationIndices.lookup(operation) << "="
             << operation->getName().getStringRef();
    }

    report << "] results [";
    for (auto [index, result] : llvm::enumerate(domain.results)) {
      if (index != 0) {
        report << ",";
      }
      report << operationIndices.lookup(result.getOwner()) << "#"
             << result.getResultNumber();
    }
    report << "]";
  }
}

} // namespace partition_analysis

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
  // Treat each DPS init placeholder as part of its consumer's domain.
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

  for (auto [domainId, domain] : llvm::enumerate(domains)) {
    Operation *insertionPoint = domain.operations.front();
    rewriter.setInsertionPoint(insertionPoint);

    SmallVector<Type> resultTypes = llvm::map_to_vector(
        domain.results, [](Value result) { return result.getType(); });

    auto domainOp = rewriter.create<PoolDomainOp>(
        insertionPoint->getLoc(), resultTypes, ValueRange{}, domainId);
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

namespace partition_materialization {

// Placeholder results do not cross domains. Record each shape input and its
// tied data result so the input can use the exported data value instead.
struct PlaceholderReplacement {
  OpOperand *placeholderInput;
  OpResult dataResult;
};

using PlaceholderReplacements = llvm::SmallVector<PlaceholderReplacement>;

// Find placeholder inputs that use this result as a shape edge.
llvm::SmallVector<OpOperand *> collectPlaceholderUses(Value result) {
  llvm::SmallVector<OpOperand *> uses;
  for (OpOperand &use : result.getUses()) {
    if (isa<PlaceholderOp>(use.getOwner())) {
      uses.push_back(&use);
    }
  }
  return uses;
}

// A placeholder result and its tied DPS result describe the same logical
// output. Placeholder results form the shape graph, while DPS results form the
// matching data graph:
//
//   shape graph: %init0 --------------------> %init1
//                  | initializes                | initializes
//                  v                            v
//   data graph:  %data0 --------------------> %data1
//
// Before partitioning, each graph uses its own value:
//   %init0 = hipsr.placeholder ...
//   %data0 = hipsr.cast ... outs(%init0)
//   %init1 = hipsr.placeholder ... ins(%init0)
//   %data1 = hipsr.cast ... ins(%data0) outs(%init1)
//
// Placeholder results stay inside their domain, so %init0 cannot cross the
// boundary. The source domain exports tied result %data0 as %data0_out.
// Redirect %init1's shape input to that exported value:
//
//                          +-- shape --> %init1
//   %data0_out ------------|
//                          +-- data  --> %data1
//
// After partitioning, the target domain captures %data0_out once and uses its
// block argument for both edges:
//   %data0_out = hipsr.pool_domain(...) {
//     ...
//     hipsr.pool_domain_yield %data0
//   }
//   hipsr.pool_domain(%data0_out) {
//   ^bb0(%input: tensor<...>):
//     %init1 = hipsr.placeholder ... ins(%input)
//     %data1 = hipsr.cast ... ins(%input) outs(%init1)
//   }
// Find shape inputs that must use an exported data value.
FailureOr<PlaceholderReplacements> collectPlaceholderReplacements(
    partition_analysis::DomainId sourceDomainId,
    const partition_analysis::Domain &domain,
    const partition_analysis::OperationDomains &operationDomains) {
  PlaceholderReplacements replacements;
  for (Operation *operation : domain.operations) {
    auto placeholder = dyn_cast<PlaceholderOp>(operation);
    if (!placeholder) {
      continue;
    }

    Operation *consumer = placeholder.getDpsConsumer();
    auto consumerDomain = operationDomains.find(consumer);
    if (consumerDomain == operationDomains.end() ||
        consumerDomain->second != sourceDomainId) {
      placeholder.emitOpError(
          "and its DPS consumer must be in the same pool domain");
      return failure();
    }
    auto dpsConsumer = cast<DestinationStyleOpInterface>(consumer);

    // Match each data result with the placeholder result that initializes it.
    for (auto [dataResult, placeholderResult] :
         llvm::zip_equal(consumer->getResults(), dpsConsumer.getDpsInits())) {
      if (placeholderResult.getDefiningOp() != placeholder.getOperation()) {
        continue;
      }

      for (OpOperand *placeholderUse :
           collectPlaceholderUses(placeholderResult)) {
        auto downstreamPlaceholder =
            cast<PlaceholderOp>(placeholderUse->getOwner());
        auto targetDomain =
            operationDomains.find(downstreamPlaceholder.getOperation());
        if (targetDomain == operationDomains.end()) {
          downstreamPlaceholder.emitOpError(
              "does not have an assigned pool domain");
          return failure();
        }
        // Same-domain shape edges stay inside the region and need no rewrite.
        if (targetDomain->second < sourceDomainId) {
          downstreamPlaceholder.emitOpError(
              "depends on a placeholder in a later pool domain");
          return failure();
        } else if (targetDomain->second > sourceDomainId) {
          // Forward shape edges use the exported data result as their bridge.
          if (!llvm::is_contained(domain.results, dataResult)) {
            placeholder.emitOpError(
                "requires its tied data result to leave the pool domain");
            return failure();
          }

          replacements.push_back(
              PlaceholderReplacement{placeholderUse, dataResult});
        }
      }
    }
  }
  return replacements;
}

// Move one assigned group into a pool_domain and wire its inputs and results.
void materializeDomain(IRRewriter &rewriter,
                       partition_analysis::DomainId domainId,
                       const partition_analysis::Domain &domain,
                       ArrayRef<PlaceholderReplacement> replacements,
                       Value context) {
  SmallVector<Value> results(domain.results.begin(), domain.results.end());
  SmallVector<Type> resultTypes = llvm::map_to_vector(
      domain.results, [](OpResult result) { return result.getType(); });
  Operation *insertionPoint = domain.operations.front();
  rewriter.setInsertionPoint(insertionPoint);

  auto domainOp = PoolDomainOp::create(rewriter, insertionPoint->getLoc(),
                                       resultTypes, ValueRange{}, domainId);
  Region &bodyRegion = domainOp.getBody();
  Block *body = rewriter.createBlock(&bodyRegion);
  // Seed context first so isolation appends every other capture after it.
  body->addArgument(context.getType(), context.getLoc());
  for (Operation *operation : domain.operations) {
    rewriter.moveOpBefore(operation, body, body->end());
  }
  // Reuse argument zero so isolation does not capture context again.
  replaceAllUsesInRegionWith(context, body->getArgument(0), bodyRegion);

  // Capture the remaining values used from outside this domain.
  SmallVector<Value> capturedValues =
      makeRegionIsolatedFromAbove(rewriter, bodyRegion);
  Block &entryBlock = bodyRegion.front();

  // Domain operands mirror block arguments: context, then captured values.
  SmallVector<Value> domainOperands{context};
  llvm::append_range(domainOperands, capturedValues);
  rewriter.modifyOpInPlace(
      domainOp, [&] { domainOp->insertOperands(0, domainOperands); });

  // Region isolation updates direct uses. Nested regions need the same mapping.
  auto capturedArguments =
      entryBlock.getArguments().take_back(capturedValues.size());
  for (auto [capturedValue, argument] :
       llvm::zip_equal(capturedValues, capturedArguments)) {
    replaceAllUsesInRegionWith(capturedValue, argument, bodyRegion);
  }

  rewriter.setInsertionPointToEnd(&entryBlock);
  PoolDomainYieldOp::create(rewriter, insertionPoint->getLoc(), results);

  llvm::DenseMap<Value, Value> exportedValues;
  for (auto [result, domainResult] :
       llvm::zip_equal(results, domainOp.getResults())) {
    exportedValues.try_emplace(result, domainResult);
  }
  // External users must use the domain results; internal uses stay unchanged.
  rewriter.replaceUsesWithIf(
      results, domainOp.getResults(), [&](OpOperand &use) {
        return !domainOp->isProperAncestor(use.getOwner());
      });

  // Replace shape inputs only after their tied data result has been exported.
  for (auto [placeholderInput, dataResult] : replacements) {
    Value exportedValue = exportedValues.lookup(dataResult);
    if (!exportedValue) {
      llvm::report_fatal_error(
          "pool-domain materialization expected an exported data result");
    }
    placeholderInput->set(exportedValue);
  }
}

// Find all shape-edge redirects first, then build domains in order.
LogicalResult
materializeDomains(const partition_analysis::DomainAssignment &assignment,
                   Value context) {
  if (assignment.domains.empty()) {
    return success();
  }

  // Plan all shape bridges before moving any operation into a domain.
  llvm::SmallVector<PlaceholderReplacements> replacementsByDomain(
      assignment.domains.size());
  for (auto [domainId, domain] : llvm::enumerate(assignment.domains)) {
    FailureOr<PlaceholderReplacements> replacements =
        collectPlaceholderReplacements(domainId, domain,
                                       assignment.operationDomains);
    if (failed(replacements)) {
      return failure();
    }
    replacementsByDomain[domainId] = std::move(*replacements);
  }

  IRRewriter rewriter(
      assignment.domains.front().operations.front()->getContext());
  // Build domains in order so each exported value exists before its users.
  for (auto [domainId, domain] : llvm::enumerate(assignment.domains)) {
    materializeDomain(rewriter, domainId, domain,
                      replacementsByDomain[domainId], context);
  }
  return success();
}

} // namespace partition_materialization

struct PartitionPoolDomainsPass
    : impl::PartitionPoolDomainsPassBase<PartitionPoolDomainsPass> {
  using impl::PartitionPoolDomainsPassBase<
      PartitionPoolDomainsPass>::PartitionPoolDomainsPassBase;

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
    if (entryBlock.getNumArguments() == 0 ||
        !isa<ContextType>(entryBlock.getArgument(0).getType())) {
      funcOp.emitError(
          "hipsr-partition-pool-domains requires function argument zero to be "
          "!hipsr.context");
      signalPassFailure();
      return;
    }
    Value context = entryBlock.getArgument(0);

    if (failed(validatePartitionInput(entryBlock))) {
      signalPassFailure();
      return;
    }

    partition_analysis::DomainAssignment assignment =
        partition_analysis::buildDomainAssignment(entryBlock);
    if (emitAnalysisReport) {
      partition_analysis::emitAnalysisReport(entryBlock, assignment);
    }

    if (failed(partition_materialization::materializeDomains(assignment,
                                                             context))) {
      signalPassFailure();
      return;
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
