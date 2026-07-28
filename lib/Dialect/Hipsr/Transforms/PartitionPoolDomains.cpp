/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- PartitionPoolDomains.cpp - Split hipsr pool domains ----------------===//
//
// Splits a single-block function into IsolatedFromAbove pool domains. A start
// barrier begins a domain; the first op that uses an end-barrier result begins
// the next. The pass checks the input before changing the IR. Placeholders do
// not set domain boundaries.
//
// "hipsr.example_end_barrier" stands in for the end-barrier op that will be
// added later.
//
// Before:
//   func.func @graph(%ctx: !hipsr.context, %input: tensor<?x4xf32>,
//       %shape: tensor<2xi64>) -> tensor<?x4xf16> {
//     %cast_init = hipsr.placeholder : tensor<?x4xf16>
//     %cast = hipsr.cast(%ctx) ins(%input : tensor<?x4xf32>)
//         outs(%cast_init : tensor<?x4xf16>) : tensor<?x4xf16>
//     %ready = "hipsr.example_end_barrier"(%ctx, %cast)
//         : (!hipsr.context, tensor<?x4xf16>) -> tensor<?x4xf16>
//     %add_init = hipsr.placeholder : tensor<?x4xf16>
//     %sum = hipsr.add(%ctx)
//         ins(%ready, %ready : tensor<?x4xf16>, tensor<?x4xf16>)
//         outs(%add_init : tensor<?x4xf16>) : tensor<?x4xf16>
//     %expand_init = hipsr.placeholder : tensor<?x4xf16>
//     %expanded = hipsr.expand(%ctx)
//         ins(%sum, %shape : tensor<?x4xf16>, tensor<2xi64>)
//         outs(%expand_init : tensor<?x4xf16>) : tensor<?x4xf16>
//     %result_init = hipsr.placeholder : tensor<?x4xf16>
//     %result = hipsr.add(%ctx)
//         ins(%expanded, %expanded : tensor<?x4xf16>, tensor<?x4xf16>)
//         outs(%result_init : tensor<?x4xf16>) : tensor<?x4xf16>
//     return %result : tensor<?x4xf16>
//   }
//
// After:
//   func.func @graph(%ctx: !hipsr.context, %input: tensor<?x4xf32>,
//       %shape: tensor<2xi64>) -> tensor<?x4xf16> {
//     %0 = hipsr.pool_domain(%ctx, %input
//         : !hipsr.context, tensor<?x4xf32>) {
//     ^bb0(%domain_ctx: !hipsr.context,
//          %domain_input: tensor<?x4xf32>):
//       %cast_init = hipsr.placeholder : tensor<?x4xf16>
//       %cast = hipsr.cast(%domain_ctx)
//           ins(%domain_input : tensor<?x4xf32>)
//           outs(%cast_init : tensor<?x4xf16>) : tensor<?x4xf16>
//       %ready = "hipsr.example_end_barrier"(%domain_ctx, %cast)
//           : (!hipsr.context, tensor<?x4xf16>) -> tensor<?x4xf16>
//       hipsr.pool_domain_yield %ready : tensor<?x4xf16>
//     } -> tensor<?x4xf16>
//     %1 = hipsr.pool_domain(%ctx, %0
//         : !hipsr.context, tensor<?x4xf16>) {
//     ^bb0(%domain_ctx: !hipsr.context,
//          %domain_input: tensor<?x4xf16>):
//       %add_init = hipsr.placeholder : tensor<?x4xf16>
//       %sum = hipsr.add(%domain_ctx)
//           ins(%domain_input, %domain_input
//               : tensor<?x4xf16>, tensor<?x4xf16>)
//           outs(%add_init : tensor<?x4xf16>) : tensor<?x4xf16>
//       hipsr.pool_domain_yield %sum : tensor<?x4xf16>
//     } -> tensor<?x4xf16>
//     %2 = hipsr.pool_domain(%ctx, %1, %shape
//         : !hipsr.context, tensor<?x4xf16>, tensor<2xi64>) {
//     ^bb0(%domain_ctx: !hipsr.context,
//          %domain_input: tensor<?x4xf16>,
//          %domain_shape: tensor<2xi64>):
//       %expand_init = hipsr.placeholder : tensor<?x4xf16>
//       %expanded = hipsr.expand(%domain_ctx)
//           ins(%domain_input, %domain_shape
//               : tensor<?x4xf16>, tensor<2xi64>)
//           outs(%expand_init : tensor<?x4xf16>) : tensor<?x4xf16>
//       %result_init = hipsr.placeholder : tensor<?x4xf16>
//       %result = hipsr.add(%domain_ctx)
//           ins(%expanded, %expanded : tensor<?x4xf16>, tensor<?x4xf16>)
//           outs(%result_init : tensor<?x4xf16>) : tensor<?x4xf16>
//       hipsr.pool_domain_yield %result : tensor<?x4xf16>
//     } -> tensor<?x4xf16>
//     return %2 : tensor<?x4xf16>
//   }
//
// The end barrier stays in the first domain. Its first user starts the second
// domain. Expand is a start barrier, so it starts the third domain.
//
// Placeholders and the ops that use them must be top-level. The pass ignores
// placeholders while finding boundaries, then moves each one before the op
// that uses it.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/IR/HipsrEndBarrierInterface.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"
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
      Operation *consumer = *placeholder->getResult(0).getUsers().begin();
      if (consumer->getBlock() != &block) {
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

  // TODO: Add runtime shape reads and writes at these boundaries.
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
