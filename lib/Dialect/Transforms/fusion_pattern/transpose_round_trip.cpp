/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- transpose_round_trip.cpp - Eliminate inverse transposes ------------===//
//
// Tensor semantics allow a round trip to forward its original value:
//   %a = hip.transpose(%ctx) ins(%x) outs(%tmp) {perm = [1, 2, 0]}
//   %b = hip.transpose(%ctx) ins(%a) outs(%out) {perm = [2, 0, 1]}
// becomes %x. Buffer-form operations write their destinations and cannot be
// removed this way. A remaining use of %a keeps the first transpose alive.
//
//===----------------------------------------------------------------------===//

#include "transpose_round_trip.hpp"

#include "hip/Dialect/IR/HipDialect.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/Sequence.h"

namespace hip {
namespace fusion_transform {
namespace {

struct TransposeRoundTrip : mlir::OpRewritePattern<mlir::hip::TransposeOp> {
  using OpRewritePattern::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(mlir::hip::TransposeOp outer,
                  mlir::PatternRewriter &rewriter) const override {
    if (outer->getNumResults() != 1)
      return mlir::failure();
    auto inner = outer.getInput().getDefiningOp<mlir::hip::TransposeOp>();
    if (!inner || inner->getNumResults() != 1 ||
        inner.getCtx() != outer.getCtx())
      return mlir::failure();

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(inner.getInput().getType());
    auto middleType =
        mlir::dyn_cast<mlir::RankedTensorType>(inner.getResult(0).getType());
    if (!inputType || !middleType ||
        inputType != outer.getResult(0).getType() ||
        inputType.getElementType() != middleType.getElementType() ||
        inputType.getRank() != middleType.getRank())
      return mlir::failure();

    mlir::ArrayAttr first = inner.getPerm();
    mlir::ArrayAttr second = outer.getPerm();
    int64_t rank = inputType.getRank();
    if (first.size() != static_cast<size_t>(rank) ||
        second.size() != static_cast<size_t>(rank))
      return mlir::failure();
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      int64_t firstAxis = mlir::cast<mlir::IntegerAttr>(first[i]).getInt();
      int64_t secondAxis = mlir::cast<mlir::IntegerAttr>(second[i]).getInt();
      if (firstAxis < 0 || firstAxis >= rank || secondAxis < 0 ||
          secondAxis >= rank ||
          inputType.getDimSize(firstAxis) != middleType.getDimSize(i) ||
          mlir::cast<mlir::IntegerAttr>(first[secondAxis]).getInt() != i)
        return mlir::failure();
    }

    rewriter.replaceOp(outer, inner.getInput());
    return mlir::success();
  }
};

} // namespace

void populateTransposeRoundTripPattern(mlir::RewritePatternSet &patterns) {
  patterns.add<TransposeRoundTrip>(patterns.getContext());
}

} // namespace fusion_transform
} // namespace hip
