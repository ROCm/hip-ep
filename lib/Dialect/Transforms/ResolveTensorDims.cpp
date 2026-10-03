/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ResolveTensorDims.cpp - Pre-bufferize `tensor.dim` resolver --------===//
//
// Per-`func.func` greedy rewrite that wraps upstream MLIR's
// `populateResolveRankedShapedTypeResultDimsPatterns` (plus the
// `InferShapedTypeOpInterface` companion populator and the tensor-dialect
// canonicalisers) in a single fixed point.  Folds `tensor.dim` queries on
// any op carrying `ReifyRankedShapedTypeOpInterface` -- including
// `tensor.expand_shape`, `tensor.collapse_shape`, `tensor.pad`, and the
// HIP DPS ops that implement the interface directly -- and
// composes the upstream `Compose{Expand,Collapse}OfX` patterns so chains
// like `dim(collapse(expand(arg)))` collapse end-to-end in one pass.
//
// Before:  // %arg : tensor<?x?x4096xf16>
//          %e  = tensor.expand_shape   %arg ... -> tensor<?x?x32x128xf16>
//          %c  = tensor.collapse_shape %e   ... -> tensor<?x?x128xf16>
//          %d  = tensor.dim %c, %c1                  // dim of a reshape result
// After:   %d1 = tensor.dim %arg, %c1                // on the chain root
//          %d  = affine.apply affine_map<()[s0] -> (s0 * 32)>()[%d1]
//          // the expand/collapse ops are dead-code-eliminated
//
// Coverage assumes
// `mlir::tensor::registerInferTypeOpInterfaceExternalModels` has been
// wired into the dialect registry (see `InitAllPasses.h` and
// `tools/hip-mlir-opt/hip-mlir-opt.cpp`).  Without that registration the
// upstream reify patterns silently no-op on
// `tensor.{expand,collapse}_shape`, the `tensor.dim` queries leak
// through bufferize as `memref.dim` of a reshape op, and downstream
// `--hip-pool-allocs` then sees one scattered dim query per reshape
// site and fragments them into many single-alloc dominance domains --
// a pooling-efficiency cost (one tiny pool each), not a hard failure.
// The canonical trigger is per-layer same-rank dynamic `onnx.Reshape`
// pairs (norm / projection chains).
//
// Pipeline placement: between `--hip-infer-shapes` and
// `--one-shot-bufferize` so refined dim values propagate into
// bufferize's allocation sizing.  Idempotent.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/Transforms/Transforms.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#define DEBUG_TYPE "hip-resolve-tensor-dims"

namespace mlir::hip {

#define GEN_PASS_DEF_RESOLVETENSORDIMSPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

struct FoldSelectOfSameValue final : public OpRewritePattern<arith::SelectOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(arith::SelectOp op,
                                PatternRewriter &rewriter) const override {
    if (op.getTrueValue() != op.getFalseValue())
      return failure();
    rewriter.replaceOp(op, op.getTrueValue());
    return success();
  }
};

struct ResolveTensorDimsPass final
    : public impl::ResolveTensorDimsPassBase<ResolveTensorDimsPass> {
  using Base::Base;
  void runOnOperation() override;
};

void ResolveTensorDimsPass::runOnOperation() {
  func::FuncOp funcOp = getOperation();
  if (funcOp.empty())
    return;

  MLIRContext *ctx = &getContext();
  RewritePatternSet patterns(ctx);

  // Upstream reify-driven dim folds.  `populateResolveRankedShapedType...`
  // contributes
  // `DimOfReifyRankedShapedTypeOpInterface<{memref,tensor}::DimOp>`, which
  // dispatches through `ReifyRankedShapedTypeOpInterface::reifyDimOfResult`.
  // HIP DPS ops implement that direct hook from the tied destination, so one
  // query does not materialize every dimension of every result. The companion
  // populator covers ops on `InferShapedTypeOpInterface`.
  memref::populateResolveRankedShapedTypeResultDimsPatterns(patterns);
  memref::populateResolveShapedTypeResultDimsPatterns(patterns);

  // Tensor canonicalisers needed to compose with the reify folds in a
  // single fixed point: `ComposeExpandOfCollapseOp`, type-driven
  // `tensor.dim` constant fold, dead reshape DCE.
  tensor::DimOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::ExpandShapeOp::getCanonicalizationPatterns(patterns, ctx);
  tensor::CollapseShapeOp::getCanonicalizationPatterns(patterns, ctx);
  patterns.add<FoldSelectOfSameValue>(ctx);

  if (failed(applyPatternsGreedily(funcOp, std::move(patterns))))
    return signalPassFailure();
}

} // namespace
} // namespace mlir::hip
