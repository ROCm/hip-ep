/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RelaxMultiDynExpandShape.cpp - multi-dyn expand-shape rewrite ------===//
//
// Rewrites `memref.expand_shape` ops whose reassociation has any group with
// MORE THAN ONE dynamic output dimension into `memref.reinterpret_cast` with
// explicit runtime strides, BEFORE the upstream `expand-strided-metadata`
// pass runs.
//
// Why this pass exists
// --------------------
// Upstream `expand-strided-metadata` (LLVM as of our prebuilt) lowers
// `memref.expand_shape` by reading the source's strides and reverse-deriving
// each output dim's stride.  Its algorithm asserts on encountering a second
// dynamic output dim within a single reassociation group:
//
//     assert(!dynSizeIdx && "There must be at most one dynamic size per
//     group");
//
// The implementation does NOT consult the explicit `output_shape` operands
// that the op carries; it only uses static type info.  Newer upstream MLIR
// has gained that support, but we cannot upgrade locally.
//
// The canonical IR that hits this assert in practice is the ONNX
// `Range -> Reshape([bs, ss])` pattern used to materialise 2-D position_ids:
//
//   %R   = hip.range ... : memref<?xi64>             // size = bs * ss
//   %pos = memref.expand_shape %R [[0, 1]]
//          output_shape [%bs, %ss]
//          : memref<?xi64> into memref<?x?xi64>      // both dyn in one group
//
// The user-supplied `%bs` and `%ss` are sufficient to compute every output
// dim's stride directly — no reverse-derivation needed.  This pass does
// exactly that and emits a `memref.reinterpret_cast`, which the existing
// downstream LLVM-conversion patterns already know how to lower.
//
// Single-dyn-per-group `memref.expand_shape` (and every
// `memref.collapse_shape`) is LEFT UNTOUCHED — upstream
// `expand-strided-metadata` handles those correctly.  This pass is purely a
// workaround for the upstream multi-dyn limitation and naturally degrades to a
// no-op when the IR contains no affected ops.
//
// Example IR
// ----------
// Before:
//
//   %expand = memref.expand_shape %src [[0, 1]]
//             output_shape [%bs, %ss]
//             : memref<?xi64> into memref<?x?xi64>
//
// After (source is identity-layout, contiguous, offset 0):
//
//   %c1   = arith.constant 1 : index
//   // stride[1] = src stride[0] = 1
//   // stride[0] = stride[1] * size[1] = %ss
//   %cast = memref.reinterpret_cast %src to
//             offset: [0], sizes: [%bs, %ss], strides: [%ss, 1]
//             : memref<?xi64> to memref<?x?xi64>
//
// Pipeline placement
// ------------------
// Runs as the very first sub-pass of `buildHipToLLVMPipeline`, immediately
// before `memref::createExpandStridedMetadataPass()`.  Any later position
// would either miss its window or fight with upstream lowering.
//
// Retirement path
// ---------------
// When the prebuilt LLVM is upgraded to a version whose
// `expand-strided-metadata` consults `output_shape`, this pass becomes a
// no-op (no matches) and can be deleted in one commit — including its
// `Passes.td` entry, the pipeline insertion in `Pipelines.cpp`, and the
// CMake entry.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "hip-relax-multi-dyn-expand-shape"

STATISTIC(NumExpandShapesRewritten,
          "Number of multi-dyn-per-group memref.expand_shape ops rewritten "
          "to memref.reinterpret_cast");

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_RELAXMULTIDYNEXPANDSHAPEPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

/// Rewrites `memref.expand_shape` whose reassociation has any group with
/// more than one dynamic output dim into `memref.reinterpret_cast`.  See the
/// file header for the rationale and the IR-snippet contract.
struct RelaxMultiDynExpandShapePattern
    : public OpRewritePattern<memref::ExpandShapeOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(RelaxMultiDynExpandShapePattern)
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(memref::ExpandShapeOp op,
                                PatternRewriter &rewriter) const override {
    MemRefType srcType = op.getSrcType();
    MemRefType resType = op.getType();
    SmallVector<ReassociationIndices> reassoc = op.getReassociationIndices();

    // 1) Match filter: at least one reassociation group must contain >1
    //    dynamic output dim.  Single-dyn-per-group and all-static cases are
    //    delegated to upstream `expand-strided-metadata`, which handles them
    //    correctly.
    auto isDynOutDim = [&](int64_t outIdx) {
      return resType.isDynamicDim(outIdx);
    };
    bool hasMultiDynGroup = llvm::any_of(reassoc, [&](ArrayRef<int64_t> group) {
      return llvm::count_if(group, isDynOutDim) > 1;
    });
    if (!hasMultiDynGroup)
      return failure();

    // 2) Read the source layout.  We require known strides and a known
    //    offset; in practice the bufferize output for ONNX models is always
    //    identity-layout (no offset, contiguous), so this is trivially
    //    satisfied.  Dynamic strides would require a
    //    `memref.extract_strided_metadata` fan-out which we don't need for any
    //    observed case — bail out and let upstream handle (or fail loudly) if
    //    encountered.
    int64_t srcOffset = 0;
    SmallVector<int64_t> srcStrides;
    if (failed(srcType.getStridesAndOffset(srcStrides, srcOffset)))
      return failure();
    if (ShapedType::isDynamic(srcOffset))
      return failure();
    if (llvm::any_of(srcStrides, ShapedType::isDynamic))
      return failure();
    if (srcStrides.size() != reassoc.size()) {
      // Shape/reassoc rank mismatch; pattern would be invalid IR, just bail.
      return failure();
    }

    Location loc = op.getLoc();

    // 3) Per-output-dim sizes.  `getMixedOutputShape` interleaves the static
    //    result-type dims (as `IndexAttr`) with the dynamic `output_shape`
    //    operands (as `Value`) in result-dim order — exactly what
    //    `memref.reinterpret_cast` expects, and the same accessor upstream
    //    `getExpandedSizes` uses.
    SmallVector<OpFoldResult> mixedSizes = op.getMixedOutputShape();

    // 4) Per-output-dim strides, computed group-by-group in row-major order.
    //    For a group covering output dims `g[0..k-1]` (outer..inner) rooted at
    //    source dim `s`, the inner stride is the source stride and each outer
    //    stride is the running product of the strides/sizes to its right:
    //
    //      stride[g[k-1]] = srcStride[s]
    //      stride[g[i]]   = stride[g[i+1]] * size[g[i+1]]   for i = k-2 .. 0
    //
    //    `makeComposedFoldedAffineApply` folds the product to a constant when
    //    both operands are static and otherwise emits a single `affine.apply`
    //    (lowered later by the pipeline's `lower-affine` pass) — mirroring
    //    upstream `getExpandedStrides`.
    AffineExpr s0, s1;
    bindSymbols(rewriter.getContext(), s0, s1);
    auto mul = [&](OpFoldResult lhs, OpFoldResult rhs) -> OpFoldResult {
      return affine::makeComposedFoldedAffineApply(rewriter, loc, s0 * s1,
                                                   {lhs, rhs});
    };

    SmallVector<OpFoldResult> mixedStrides(resType.getRank());
    for (auto [s, group] : llvm::enumerate(reassoc)) {
      OpFoldResult runningStride = rewriter.getIndexAttr(srcStrides[s]);
      for (int64_t i = static_cast<int64_t>(group.size()) - 1; i >= 0; --i) {
        mixedStrides[group[i]] = runningStride;
        runningStride = mul(runningStride, mixedSizes[group[i]]);
      }
    }

    // 5) Emit `memref.reinterpret_cast` with the original op's result type.
    //    The result type's layout (identity vs explicit strided) must be
    //    consistent with our (offset, sizes, strides) — for the canonical
    //    case (identity source) the resulting output is identity too and
    //    matches the no-layout result type.  The verifier will catch any
    //    mismatch loudly rather than miscompile.
    OpFoldResult mixedOffset = rewriter.getIndexAttr(srcOffset);
    auto cast = memref::ReinterpretCastOp::create(rewriter, loc, resType,
                                                  op.getSrc(), mixedOffset,
                                                  mixedSizes, mixedStrides);

    LLVM_DEBUG({
      llvm::dbgs() << "[relax-multi-dyn-expand-shape] rewrote " << op << " -> "
                   << cast << "\n";
    });
    rewriter.replaceOp(op, cast.getResult());
    ++NumExpandShapesRewritten;
    return success();
  }
};

struct RelaxMultiDynExpandShapePass
    : public impl::RelaxMultiDynExpandShapePassBase<
          RelaxMultiDynExpandShapePass> {

  void runOnOperation() override {
    RewritePatternSet patterns(&getContext());
    patterns.add<RelaxMultiDynExpandShapePattern>(&getContext());
    // Greedy driver; the pattern is single-shot per op and never produces a
    // new `memref.expand_shape`, so it converges in one pass over the func.
    if (failed(applyPatternsGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace
} // namespace hip
} // namespace mlir
