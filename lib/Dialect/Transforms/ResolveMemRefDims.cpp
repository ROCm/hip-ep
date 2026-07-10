/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ResolveMemRefDims.cpp - Post-bufferize `memref.dim` resolver ------===//
//
// `--hip-resolve-memref-dims`: fold `memref.dim` of a view op (`subview` /
// `expand_shape` / `collapse_shape` / `view`) to a dim of the root buffer --
// ultimately `memref.dim` of a function argument.  Post-bufferize twin of
// `--hip-resolve-tensor-dims`: it re-applies the reify-based `memref` dim folds
// once the memref views exist (they are created during bufferization and by
// `--hip-promote-strided-hip-operands`, i.e. after resolve-tensor-dims ran).
//
// Why: `--hip-pool-allocs` opens a new pool "domain" for every alloc whose
// size SSA is defined below the earliest pooled alloc.  `memref.dim` of a
// mid-block view is such a size -- equal to a func-arg dim but pinned to the
// view's position, so `--hip-hoist-alloc-size-arith` cannot lift it.  Folding
// it to the arg dim lets the size hoist to block top, so one `hip.get_pool`
// serves many allocs instead of degenerating to one domain per alloc.
//
// Before:
//   %v = memref.subview %buf ...     // view, created mid-block
//   %d = memref.dim %v, %c0          // size pinned below earlier allocs
//   %a = memref.alloc(%d)
// After:
//   %d = memref.dim %arg, %c0        // size at block top, hoistable
//   %a = memref.alloc(%d)            // --hip-pool-allocs merges the domain
//
// Placement: after --hip-promote-strided-hip-operands /
// --hip-materialize-host-scalars, before --hip-hoist-alloc-size-arith /
// --hip-pool-allocs.  Idempotent.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir::hip {

#define GEN_PASS_DEF_RESOLVEMEMREFDIMSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct ResolveMemRefDimsPass final
    : public impl::ResolveMemRefDimsPassBase<ResolveMemRefDimsPass> {
  using Base::Base;
  void runOnOperation() override;
};

void ResolveMemRefDimsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  // The reify-based dim folds rewrite `dim(view)` via the view's
  // `reifyResultShapes` implementation; the greedy driver then chases
  // `dim(view) -> ... -> dim(%arg)` down to the function argument.
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);

  // Compose the chain with the `memref.dim` canonicalisers only -- NOT the
  // view ops' own patterns: fold the dim queries, leave the views intact
  // (pool-allocs and the strided-operand ABI depend on them).
  memref::DimOp::getCanonicalizationPatterns(patterns, ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    return signalPassFailure();
}

} // namespace
} // namespace mlir::hip
