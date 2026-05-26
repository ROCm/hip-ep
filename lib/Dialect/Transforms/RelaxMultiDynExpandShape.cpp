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

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
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

/// Multiply two `OpFoldResult` values, folding when both sides are constant.
/// Materialises constants to `arith.constant index` only when at least one
/// side is a Value.
static OpFoldResult mulOpFold(OpBuilder &b, Location loc, OpFoldResult lhs,
                              OpFoldResult rhs) {
  std::optional<int64_t> lhsC = getConstantIntValue(lhs);
  std::optional<int64_t> rhsC = getConstantIntValue(rhs);
  if (lhsC && rhsC)
    return b.getIndexAttr(*lhsC * *rhsC);
  // Identity simplifications keep the IR small and avoid pointless arith.muli
  // around an `output_shape` operand or a known stride-of-1.
  if (lhsC && *lhsC == 1)
    return rhs;
  if (rhsC && *rhsC == 1)
    return lhs;
  Value lv = getValueOrCreateConstantIndexOp(b, loc, lhs);
  Value rv = getValueOrCreateConstantIndexOp(b, loc, rhs);
  return arith::MulIOp::create(b, loc, lv, rv).getResult();
}

/// Rewrites `memref.expand_shape` whose reassociation has any group with
/// more than one dynamic output dim into `memref.reinterpret_cast`.  See the
/// file header for the rationale and the IR-snippet contract.
struct RelaxMultiDynExpandShapePattern
    : public OpRewritePattern<memref::ExpandShapeOp> {
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
    bool hasMultiDynGroup = false;
    for (const auto &group : reassoc) {
      int dynCount = 0;
      for (int64_t outIdx : group) {
        if (resType.isDynamicDim(outIdx))
          ++dynCount;
      }
      if (dynCount > 1) {
        hasMultiDynGroup = true;
        break;
      }
    }
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

    // 3) Build per-output-dim sizes as OpFoldResult.  Static dims come from
    //    the result type; dynamic dims come from the `output_shape` operands
    //    in order of appearance among the dynamic dims.
    SmallVector<OpFoldResult> mixedSizes;
    mixedSizes.reserve(resType.getRank());
    ValueRange dynOutputSizes = op.getOutputShape();
    int64_t dynIdx = 0;
    for (int64_t i = 0, e = resType.getRank(); i < e; ++i) {
      if (resType.isDynamicDim(i)) {
        if (dynIdx >= static_cast<int64_t>(dynOutputSizes.size()))
          return failure(); // malformed op; let verifier flag it
        mixedSizes.push_back(dynOutputSizes[dynIdx++]);
      } else {
        mixedSizes.push_back(rewriter.getIndexAttr(resType.getDimSize(i)));
      }
    }

    // 4) Compute per-output-dim strides.  For each reassociation group at
    //    source dim `s` covering output dims `g[0], g[1], ..., g[k-1]` in
    //    C-order:
    //
    //      stride_out[g[k-1]] = srcStrides[s]
    //      stride_out[g[j]]   = stride_out[g[j+1]] * size_out[g[j+1]]
    //                                              for j = k-2 .. 0
    //
    //    This matches the row-major layout convention used everywhere in
    //    this pipeline (identity bufferize output, hip ops' bare-ptr ABI).
    SmallVector<OpFoldResult> mixedStrides(resType.getRank());
    for (size_t s = 0; s < reassoc.size(); ++s) {
      const auto &group = reassoc[s];
      if (group.empty())
        return failure();
      int64_t srcStride = srcStrides[s];
      int64_t inner = group.back();
      mixedStrides[inner] = rewriter.getIndexAttr(srcStride);
      OpFoldResult acc = mixedStrides[inner];
      for (int j = static_cast<int>(group.size()) - 2; j >= 0; --j) {
        int64_t outIdx = group[j];
        int64_t innerIdx = group[j + 1];
        // acc *= size_out[innerIdx]
        acc = mulOpFold(rewriter, loc, acc, mixedSizes[innerIdx]);
        mixedStrides[outIdx] = acc;
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
