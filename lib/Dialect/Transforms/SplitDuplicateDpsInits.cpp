/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SplitDuplicateDpsInits.cpp - Split shared DPS init seeds ----------===//
//
// `tensor.empty` has undefined contents and is pure, so CSE may merge
// same-typed empties that seed unrelated destination-style operations. Such
// sharing carries no tensor-level data dependence. Preserving it through
// bufferization, however, can alias outputs that must be distinct and can form
// very large alias classes in One-Shot Bufferize.
//
// This pass preserves the first DPS init use of each `tensor.empty` and
// replaces later init uses with fresh `bufferization.alloc_tensor` operations.
// The rewrite is value-preserving because both operations produce undefined
// tensor contents. Non-empty init values are left unchanged because their
// contents may be semantically significant. Downstream lifetime analysis may
// still reuse the resulting buffers.
//
// Before (post-CSE; one %e feeds inits of two ops):
//
//   %e = tensor.empty() : tensor<AxBxf16>
//   %a = hip.sigmoid ... outs(%e)
//   %b = hip.tanh    ... outs(%e)
//
// After:
//
//   %e  = tensor.empty()               : tensor<AxBxf16>
//   %e2 = bufferization.alloc_tensor() : tensor<AxBxf16>
//   %a = hip.sigmoid ... outs(%e)
//   %b = hip.tanh    ... outs(%e2)
//
// The pass runs after CSE, which creates the sharing, and before One-Shot
// Bufferize, which consumes the aliasing information.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/Interfaces/DestinationStyleOpInterface.h"

#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-split-duplicate-dps-inits"

STATISTIC(NumInitsSplit,
          "Number of repeated DPS init seeds replaced with fresh "
          "bufferization.alloc_tensor");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_SPLITDUPLICATEDPSINITSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct SplitDuplicateDpsInitsPass
    : public impl::SplitDuplicateDpsInitsPassBase<SplitDuplicateDpsInitsPass> {

  void runOnOperation() override {
    llvm::SmallPtrSet<Value, 32> seenEmptyInitSeeds;
    getOperation().walk([&](DestinationStyleOpInterface dpsOp) {
      for (OpOperand &init : dpsOp.getDpsInitsMutable()) {
        Value initValue = init.get();
        auto emptyOp = initValue.getDefiningOp<tensor::EmptyOp>();
        if (!emptyOp || seenEmptyInitSeeds.insert(initValue).second)
          continue;

        OpBuilder builder(dpsOp);
        auto freshInit = bufferization::AllocTensorOp::create(
            builder, emptyOp.getLoc(), emptyOp.getType(),
            emptyOp.getDynamicSizes());
        init.set(freshInit);
        ++NumInitsSplit;
        LLVM_DEBUG(llvm::dbgs() << "[" DEBUG_TYPE "] " << dpsOp->getName()
                                << " init #" << init.getOperandNumber()
                                << ": replaced repeated tensor.empty seed\n");
      }
    });
  }
};

} // namespace
} // namespace hip
} // namespace mlir
