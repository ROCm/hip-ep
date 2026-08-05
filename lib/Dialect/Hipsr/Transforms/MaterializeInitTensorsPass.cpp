/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MaterializeInitTensorsPass.cpp - Materialize placeholder inits -----===//
//
// Rewrites each pool domain into the virtual 3-region form: every
// hipsr.placeholder shape region becomes an scf.execute_region yielding
// !shape.shape, every placeholder result becomes a tensor.empty built from
// that shape, and the data ops keep their order.
//
// Before:
//   %init = hipsr.placeholder(%ctx) ins(%a, %b) : tensor<?x512xf16>
//       shape_region { ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
//         hipsr.shape_yield2 %result_shape : !shape.shape }
//   %0 = hipsr.matmul(%ctx) ins(%a, %b) outs(%init) : tensor<?x512xf16>
// After:
//   %shape = scf.execute_region -> !shape.shape { ... }
//   %init = tensor.empty(%d0) : tensor<?x512xf16>
//   %0 = hipsr.matmul(%ctx) ins(%a, %b) outs(%init) : tensor<?x512xf16>
//
// The phases run per domain: collect and group the placeholders, build the
// shape regions, build the tensor allocations, then replace and erase.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_MATERIALIZEINITTENSORSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Collects a domain's placeholders and moves them to the front of the domain
// block. SSA form already orders them topologically, so grouping only has to
// keep their relative order.
FailureOr<SmallVector<PlaceholderOp>>
collectAndGroupPlaceholders(PoolDomainOp poolDomain) {
  Block &domainBlock = poolDomain.getBody().front();

  SmallVector<PlaceholderOp> placeholders;
  for (Operation &operation : domainBlock) {
    auto placeholder = dyn_cast<PlaceholderOp>(&operation);
    if (!placeholder) {
      continue;
    }
    if (placeholder.getShapeRegion().empty()) {
      placeholder.emitOpError(
          "shape region must be populated by -hipsr-populate-shape-region");
      return failure();
    }
    placeholders.push_back(placeholder);
  }

  // Placeholders already sitting at the front are skipped rather than moved:
  // splicing an operation before itself corrupts the block's operation list.
  Block::iterator insertionPoint = domainBlock.begin();
  for (PlaceholderOp placeholder : placeholders) {
    if (&*insertionPoint == placeholder.getOperation()) {
      ++insertionPoint;
      continue;
    }
    placeholder->moveBefore(&domainBlock, insertionPoint);
  }

  return placeholders;
}

struct MaterializeInitTensorsPass
    : impl::MaterializeInitTensorsPassBase<MaterializeInitTensorsPass> {
  void runOnOperation() override {
    WalkResult walkResult = getOperation().walk([](PoolDomainOp poolDomain) {
      if (failed(collectAndGroupPlaceholders(poolDomain))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
