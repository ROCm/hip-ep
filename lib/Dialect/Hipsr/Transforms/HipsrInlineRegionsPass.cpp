/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipsrInlineRegionsPass.cpp - Inline pool_domain and compute --------===//
//
// Flattens hipsr.pool_domain and hipsr.compute regions.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/Visitors.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_HIPSRINLINEREGIONSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Splice the region onto the parent and replace `op` with the yield operands.
//
// Before:
//   %r = op(%a, %b) {
//   ^bb0(%x, %y):
//     %v = foo %x, %y
//     yield %v
//   }
//   bar %r
// After:
//   %v = foo %a, %b
//   bar %v
void inlineIsolatedRegion(RewriterBase &rewriter, Operation *op) {
  Block *body = &op->getRegion(0).front();
  Operation *terminator = body->getTerminator();
  ValueRange results = terminator->getOperands();
  rewriter.inlineBlockBefore(body, op, op->getOperands());
  rewriter.replaceOp(op, results);
  rewriter.eraseOp(terminator);
}

struct HipsrInlineRegionsPass
    : impl::HipsrInlineRegionsPassBase<HipsrInlineRegionsPass> {
  void runOnOperation() override {
    SmallVector<Operation *> ops;
    getOperation().walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (isa<PoolDomainOp, ComputeOp>(op)) {
        ops.push_back(op);
      }
    });

    IRRewriter rewriter(&getContext());
    for (Operation *op : ops) {
      inlineIsolatedRegion(rewriter, op);
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
