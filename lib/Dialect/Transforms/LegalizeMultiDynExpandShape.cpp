/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LegalizeMultiDynExpandShape.cpp ------------------------------------===//
//
// Rewrites `memref.expand_shape` ops whose reassociation groups carry more
// than one dynamic dim into `memref.reinterpret_cast`, sidestepping the
// upstream `expand-strided-metadata` pass's hard limit of "at most one
// dynamic dim per group".
//
// Background:
//   The upstream pass's `getExpandedSizes` (and `getExpandedStrides`) helper
//   in `ExpandStridedMetadata.cpp` carries a long-standing assumption baked
//   into the assert
//     `assert(!dynSizeIdx && "There must be at most one dynamic size per
//     group")`.
//   That assumption pre-dates the addition of the `output_shape` operand to
//   `memref.expand_shape` (which already supplies the per-dim runtime size).
//   Upstream commit eca7d83 (LLVM main, May 2025) removed the restriction,
//   but our prebuilt LLVM/MLIR predates that fix.
//
// Approach:
//   Walk every `memref.expand_shape` in the function. For each op whose
//   reassociation groups include a multi-dynamic-dim group, rewrite it
//   into a `memref.reinterpret_cast`:
//
//     %src : memref<?xT, identity-layout>           // contiguous, offset 0
//     %dst = memref.expand_shape %src [[0, 1]]
//              output_shape [%d0, %d1]
//              : memref<?xT> into memref<?x?xT>
//
//   becomes
//
//     %s1 = arith.constant 1 : index            // innermost stride
//     %s0 = arith.muli  %d1, %s1                // stride for dim 0
//     %dst = memref.reinterpret_cast %src to
//              offset: [0],
//              sizes:   [%d0, %d1],
//              strides: [%s0, %s1]
//              : memref<?xT, ...> to memref<?x?xT, strided<[?,1]>>
//
//   The strides are right-to-left products of the inner-most sizes — the
//   exact layout the source memref already had after `IdentityLayoutMap`
//   bufferization, just reinterpreted at a coarser rank. This matches what
//   the upstream pass would have produced for the same case.
//
// Safety:
//   The pass refuses to rewrite if the source memref has a non-zero offset
//   or non-contiguous strides — in that case the simple "right-to-left
//   product" strides used here would be wrong. Such inputs are not produced
//   today by the OnnxToHip Reshape converter (it always emits contiguous
//   destinations through bufferization with `IdentityLayoutMap`), but the
//   check keeps the pass safe in the face of future IR changes.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Transforms/Passes.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_LEGALIZEMULTIDYNEXPANDSHAPEPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

} // namespace hip
} // namespace mlir

using namespace mlir;

namespace {

/// Returns true iff `expand` has any reassociation group containing more
/// than one dynamic dimension in the EXPANDED (result) type.
static bool hasMultiDynGroup(memref::ExpandShapeOp expand) {
  MemRefType resultType = expand.getResultType();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  for (const auto &group : expand.getReassociationIndices()) {
    unsigned dynCount = 0;
    for (int64_t resultDimIdx : group) {
      if (ShapedType::isDynamic(resultShape[resultDimIdx]))
        ++dynCount;
    }
    if (dynCount > 1)
      return true;
  }
  return false;
}

/// Returns true iff `src` has identity layout (offset == 0, strides ==
/// right-to-left product of sizes). The source has dynamic shape, but the
/// stride pattern can still be statically known if the layout is identity
/// (which is what bufferization with `IdentityLayoutMap` always produces).
static bool sourceHasIdentityLayout(MemRefType srcType) {
  // No layout attr => default = canonical identity (offset 0, contiguous).
  if (srcType.getLayout().isIdentity())
    return true;

  // Strided layout: must be offset = 0 and last stride = 1, with each
  // earlier stride equal to product of all later sizes. We allow ? in
  // sizes (those would have been collapsed away to a single ? stride in
  // any non-identity case, so this is conservative but correct).
  int64_t offset;
  SmallVector<int64_t> strides;
  if (failed(srcType.getStridesAndOffset(strides, offset)))
    return false;
  if (offset != 0)
    return false;

  // Bail if any stride is non-static — the right-to-left contiguous check
  // below would be ambiguous in that case.
  ArrayRef<int64_t> sizes = srcType.getShape();
  int64_t expectedStride = 1;
  for (int i = sizes.size() - 1; i >= 0; --i) {
    if (ShapedType::isDynamic(strides[i]))
      return false;
    if (strides[i] != expectedStride)
      return false;
    // If the size is dynamic, downstream products are dynamic too and we
    // would have bailed above. If static, multiply through.
    if (ShapedType::isDynamic(sizes[i]))
      return false;
    expectedStride *= sizes[i];
  }
  return true;
}

/// Rewrite the multi-dyn expand_shape into memref.reinterpret_cast. Builds
/// the output sizes/strides from the op's existing output_shape operands.
static LogicalResult rewriteExpandShape(memref::ExpandShapeOp expand,
                                        IRRewriter &rewriter) {
  Location loc = expand.getLoc();

  // Refuse to rewrite when the source layout is not identity — see the
  // safety note in the file-header comment.
  MemRefType srcType = cast<MemRefType>(expand.getSrc().getType());
  if (!sourceHasIdentityLayout(srcType))
    return failure();

  MemRefType resultType = expand.getResultType();
  unsigned resultRank = resultType.getRank();

  // Collect the per-dim sizes for the result. `output_shape` of expand_shape
  // is mixed: static dims are absent and present in static_output_shape,
  // dynamic dims have a Value operand. The op's `getMixedOutputShape()`
  // already returns the merged form indexed by result dim, exactly what we
  // need here.
  SmallVector<OpFoldResult> mixedSizes = expand.getMixedOutputShape();
  assert(mixedSizes.size() == resultRank &&
         "mixed output shape must match result rank");

  // Build SSA values for each size dim. For static dims, `mixedSizes[i]`
  // is an IntegerAttr; materialize it as `arith.constant`. For dynamic
  // dims it is already a Value.
  SmallVector<Value> sizeValues;
  sizeValues.reserve(resultRank);
  for (OpFoldResult ofr : mixedSizes) {
    if (auto val = dyn_cast<Value>(ofr)) {
      sizeValues.push_back(val);
    } else {
      auto attr = cast<IntegerAttr>(cast<Attribute>(ofr));
      sizeValues.push_back(
          rewriter.create<arith::ConstantIndexOp>(loc, attr.getInt()));
    }
  }

  // Compute right-to-left contiguous strides. Innermost stride = 1 (kept
  // as an IndexAttr so reinterpret_cast infers a static `1` in the result
  // strides layout — mixing a Value-typed constant 1 with the static-shape
  // entry kStaticUnit would make the type verifier complain). All earlier
  // strides are products of dynamic sizes, so they must be SSA Values.
  SmallVector<OpFoldResult> stridesOfr(resultRank);
  stridesOfr[resultRank - 1] = rewriter.getIndexAttr(1);
  Value runningStride;
  for (int i = static_cast<int>(resultRank) - 2; i >= 0; --i) {
    if (!runningStride) {
      // strides[rank-2] = sizes[rank-1].
      runningStride = sizeValues[i + 1];
    } else {
      runningStride =
          rewriter.create<arith::MulIOp>(loc, runningStride, sizeValues[i + 1]);
    }
    stridesOfr[i] = runningStride;
  }

  // Build the OpFoldResult arrays that reinterpret_cast wants. Sizes are
  // SSA Values for now (each `output_shape` dim is dynamic at the type
  // level for the cases this pass targets); if a future caller passes us
  // a static-dim entry we could refine these to IndexAttr too, but it's
  // not necessary today.
  SmallVector<OpFoldResult> sizesOfr(sizeValues.begin(), sizeValues.end());
  OpFoldResult offsetOfr = rewriter.getIndexAttr(0);

  // The result memref type must match the expand_shape's result rank /
  // element type / memory space, but its layout must be a strided<[?,...]>
  // (identity-shaped strides, dynamic per-dim values) — exactly what
  // reinterpret_cast naturally infers when given dynamic stride values.
  SmallVector<int64_t> staticStrides(resultRank, ShapedType::kDynamic);
  staticStrides.back() = 1;
  auto stridedLayout = StridedLayoutAttr::get(rewriter.getContext(),
                                              /*offset=*/0, staticStrides);
  auto castType =
      MemRefType::get(resultType.getShape(), resultType.getElementType(),
                      stridedLayout, resultType.getMemorySpace());

  auto castOp = rewriter.create<memref::ReinterpretCastOp>(
      loc, castType, expand.getSrc(), offsetOfr, sizesOfr, stridesOfr);

  // The result of reinterpret_cast has a strided layout, but consumers of
  // the original expand_shape expect the original result type (likely the
  // contiguous default-layout form). Cast back through `memref.cast` so the
  // SSA types match exactly.
  Value finalResult = castOp.getResult();
  if (castType != resultType) {
    finalResult = rewriter.create<memref::CastOp>(loc, resultType, finalResult);
  }
  rewriter.replaceOp(expand, finalResult);
  return success();
}

struct LegalizeMultiDynExpandShapePass
    : public mlir::hip::impl::LegalizeMultiDynExpandShapePassBase<
          LegalizeMultiDynExpandShapePass> {
  using LegalizeMultiDynExpandShapePassBase::
      LegalizeMultiDynExpandShapePassBase;

  void runOnOperation() override {
    func::FuncOp func = getOperation();
    IRRewriter rewriter(&getContext());

    // Collect candidate ops first; modifying during a walk is awkward.
    SmallVector<memref::ExpandShapeOp> candidates;
    func.walk([&](memref::ExpandShapeOp op) {
      if (hasMultiDynGroup(op))
        candidates.push_back(op);
    });

    for (memref::ExpandShapeOp op : candidates) {
      rewriter.setInsertionPoint(op);
      if (failed(rewriteExpandShape(op, rewriter))) {
        op.emitError() << "hip-legalize-multi-dyn-expand-shape: cannot "
                          "rewrite expand_shape with multi-dyn group "
                          "(source memref has non-identity layout)";
        signalPassFailure();
        return;
      }
    }
  }
};

} // namespace
