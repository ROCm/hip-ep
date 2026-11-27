/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===----------------------------------------------------------------------===//
// A Transpose that only moves size-1 dimensions is a reshape.
//
// Memory order is determined by the dimensions whose extent is not 1: a size-1
// dimension contributes a stride but never a distinct index. When the
// permutation keeps the non-unit dimensions in their original relative order,
// input and result hold the same elements in the same order, so no data has to
// move:
//
//   %y = hip.transpose(%x) perm = [1, 0, 2]   // [1,N,C] -> [N,1,C]
//
// becomes
//
//   %c = tensor.collapse_shape %x             // [1,N,C] -> [N,C]
//   %y = tensor.expand_shape %c               // [N,C]   -> [N,1,C]
//
// Both reshapes are views. Dimensions that are not statically 1, dynamic ones
// included, are treated as non-unit: a dynamic extent may turn out to be 1, but
// the result is then still correct, only not removed as a copy.
//
// Tensor form only, so the pattern runs in the canonicalizer before
// bufferization.
//===----------------------------------------------------------------------===//

#include "HipTransposeUtilsInternal.h"
#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/PatternMatch.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

struct UnitDimTransposeToReshape : public OpRewritePattern<TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(TransposeOp transpose,
                                PatternRewriter &rewriter) const override {
    if (transpose->getNumResults() != 1)
      return rewriter.notifyMatchFailure(transpose, "not in tensor form");

    auto inputType = dyn_cast<RankedTensorType>(transpose.getInput().getType());
    auto resultType =
        dyn_cast<RankedTensorType>(transpose->getResult(0).getType());
    if (!inputType || !resultType)
      return rewriter.notifyMatchFailure(transpose, "unranked tensors");

    SmallVector<int64_t> perm;
    if (!hip::detail::getPermutation(transpose, perm))
      return rewriter.notifyMatchFailure(transpose, "unreadable permutation");
    // The verifier does not relate the result type to the permutation, and a
    // reshape built from a result type that disagrees with it would change the
    // program.
    if (!hip::detail::permutesIntoResultType(transpose.getInput(),
                                             transpose->getResult(0), perm))
      return rewriter.notifyMatchFailure(
          transpose, "result type disagrees with the permutation");

    // The non-unit dimensions, in the order the result lists them, must be in
    // ascending input order.
    SmallVector<int64_t> nonUnitDims;
    for (int64_t dim : perm)
      if (inputType.getDimSize(dim) != 1)
        nonUnitDims.push_back(dim);
    if (!llvm::is_sorted(nonUnitDims))
      return rewriter.notifyMatchFailure(transpose,
                                         "a non-unit dimension moves");

    if (inputType == resultType) {
      rewriter.replaceOp(transpose, transpose.getInput());
      return success();
    }

    // `nonUnitDims` is sorted, so it is the non-unit dimensions in input order.
    SmallVector<int64_t> collapsedShape;
    for (int64_t dim : nonUnitDims)
      collapsedShape.push_back(inputType.getDimSize(dim));
    auto collapsedType =
        RankedTensorType::get(collapsedShape, inputType.getElementType());
    auto collapseReassoc =
        getReassociationIndicesForReshape(inputType, collapsedType);
    auto expandReassoc =
        getReassociationIndicesForReshape(collapsedType, resultType);
    if (!collapseReassoc || !expandReassoc)
      return rewriter.notifyMatchFailure(transpose,
                                         "no reassociation between the shapes");

    // Read each dynamic result extent from the input dimension it came from.
    Location loc = transpose.getLoc();
    SmallVector<OpFoldResult> outputShape;
    for (auto [resultDim, inputDim] : llvm::enumerate(perm)) {
      if (resultType.isDynamicDim(resultDim))
        outputShape.push_back(
            tensor::DimOp::create(rewriter, loc, transpose.getInput(), inputDim)
                .getResult());
      else
        outputShape.push_back(
            rewriter.getIndexAttr(resultType.getDimSize(resultDim)));
    }

    Value collapsed = tensor::CollapseShapeOp::create(
        rewriter, loc, collapsedType, transpose.getInput(), *collapseReassoc);
    rewriter.replaceOpWithNewOp<tensor::ExpandShapeOp>(
        transpose, resultType, collapsed, *expandReassoc, outputShape);
    return success();
  }
};

} // namespace

void hip::detail::populateUnitDimTransposePatterns(
    RewritePatternSet &patterns) {
  patterns.add<UnitDimTransposeToReshape>(patterns.getContext());
}
