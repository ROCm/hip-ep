/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Flatten -> standard tensor ops (zero-cost metadata reinterpretation)
//===----------------------------------------------------------------------===//
//
// `onnx.Flatten` reshapes a rank-r input into a 2D output:
//   output[0] = d_0 * d_1 * ... * d_{axis-1}     (1 when axis == 0)
//   output[1] = d_axis * d_{axis+1} * ... * d_n  (1 when axis == r)
//
// This is pure shape metadata reinterpretation — no data movement. We lower
// to `tensor.collapse_shape` (and `tensor.expand_shape` for the corner cases
// where one of the two output dims is 1 and not directly producible by a
// single collapse). Bufferization later turns those into zero-copy
// `memref.collapse_shape` / `memref.expand_shape` descriptor edits.
//
// Decomposition by case (a = normalised axis, r = input rank):
//
//   General case  (1 <= a <= r-1, r >= 2):
//     Before:
//       %y = "onnx.Flatten"(%x) {axis = a} : tensor<d0 x ... x dn>
//                                          -> tensor<P0 x P1>
//     After:
//       %y = tensor.collapse_shape %x [[0..a-1], [a..r-1]]
//              : tensor<d0 x ... x dn> into tensor<P0 x P1>
//
//   axis = 0, r = 1:
//     Before:  tensor<d0> -> tensor<1 x d0>
//     After:   tensor.expand_shape %x [[0, 1]] output_shape [1, d0]
//
//   axis = 0, r >= 2:
//     Before:  tensor<d0 x ... x dn> -> tensor<1 x P>
//     After:
//       %t = tensor.collapse_shape %x [[0, 1, ..., r-1]] : ... into tensor<P>
//       %y = tensor.expand_shape %t [[0, 1]] output_shape [1, P]
//                                                : tensor<P> into tensor<1 x P>
//
//   axis = r, r = 1:
//     Before:  tensor<d0> -> tensor<d0 x 1>
//     After:   tensor.expand_shape %x [[0, 1]] output_shape [d0, 1]
//
//   axis = r, r >= 2:
//     Before:  tensor<d0 x ... x dn> -> tensor<P x 1>
//     After:
//       %t = tensor.collapse_shape %x [[0, 1, ..., r-1]] : ... into tensor<P>
//       %y = tensor.expand_shape %t [[0, 1]] output_shape [P, 1]
//                                                : tensor<P> into tensor<P x 1>
//
// Dynamic shapes are supported. `tensor.collapse_shape` naturally preserves
// dynamic dims (the collapsed extent becomes dynamic if any source dim is).
// For the expand step, the output_shape uses the static `1` literal and
// `tensor.dim` on the intermediate for the dynamic side.

struct FlattenDecompose : public mlir::RewritePattern {
  FlattenDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Flatten", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    if (inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(op, "element type mismatch");
    if (outputType.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "Flatten output must be rank-2");

    int64_t r = inputType.getRank();
    if (r < 1)
      return rewriter.notifyMatchFailure(
          op, "rank-0 input not supported by Flatten lowering");

    // Normalise axis to [0, r]. ONNX allows axis in [-r, r]; default = 1.
    // ONNX attributes are typically `si64` (signed) — use getSExtValue so
    // we don't trip the signless-only assert inside IntegerAttr::getInt.
    int64_t axis = 1;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
      axis = attr.getValue().getSExtValue();
    if (axis < 0)
      axis += r;
    if (axis < 0 || axis > r)
      return rewriter.notifyMatchFailure(op, "axis out of range [-r, r]");

    mlir::Location loc = op->getLoc();

    // Helper: build a single-group reassociation [[0, 1, ..., n-1]] that
    // collapses an entire rank-n tensor into rank-1.
    auto buildFullCollapseReassoc = [](int64_t n) {
      mlir::ReassociationIndices group;
      group.reserve(n);
      for (int64_t i : llvm::seq<int64_t>(n))
        group.push_back(i);
      return llvm::SmallVector<mlir::ReassociationIndices>{group};
    };

    // Helper: produce the 1D intermediate `tensor<P x elem>` (P may be
    // dynamic) by collapsing the entire input. Only called for r >= 2.
    auto buildFullCollapse1D = [&]() -> mlir::Value {
      // Compute the single dim of the 1D result: static iff every input dim
      // is static, otherwise dynamic.
      int64_t flatDim = 1;
      bool anyDyn = false;
      for (int64_t i : llvm::seq<int64_t>(r)) {
        if (inputType.isDynamicDim(i)) {
          anyDyn = true;
          break;
        }
        flatDim *= inputType.getDimSize(i);
      }
      int64_t flatExtent = anyDyn ? mlir::ShapedType::kDynamic : flatDim;
      auto flatType =
          mlir::RankedTensorType::get({flatExtent}, inputType.getElementType());
      auto reassoc = buildFullCollapseReassoc(r);
      auto collapseOp = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, flatType, input, reassoc);
      return collapseOp.getResult();
    };

    // Case A: 1 <= axis <= r-1 (with r >= 2 implied). Pure collapse from
    // rank-r to rank-2 with two contiguous groups.
    if (axis >= 1 && axis <= r - 1) {
      mlir::ReassociationIndices g0, g1;
      for (int64_t i : llvm::seq<int64_t>(axis))
        g0.push_back(i);
      for (int64_t i = axis; i < r; ++i)
        g1.push_back(i);
      llvm::SmallVector<mlir::ReassociationIndices> reassoc{g0, g1};
      auto collapseOp = mlir::tensor::CollapseShapeOp::create(
          rewriter, loc, outputType, input, reassoc);
      rewriter.replaceOp(op, collapseOp.getResult());
      return mlir::success();
    }

    // Cases B/C: axis == 0 or axis == r. Both require an expand step to
    // insert the size-1 dim.
    //
    // If r == 1, expand the rank-1 input directly into the rank-2 output.
    // If r >= 2, first collapse the input into rank-1, then expand.

    mlir::Value src;
    mlir::RankedTensorType srcType;
    if (r == 1) {
      src = input;
      srcType = inputType;
    } else {
      src = buildFullCollapse1D();
      srcType = mlir::cast<mlir::RankedTensorType>(src.getType());
    }

    // The rank-1 -> rank-2 expand: single reassociation group [[0, 1]].
    // The size-1 dim is whichever output dim equals 1 (axis == 0 -> dim 0,
    // axis == r -> dim 1).
    llvm::SmallVector<mlir::ReassociationIndices> expandReassoc{{0, 1}};

    // Build output_shape as mixed (static 1 attr + runtime dim for the other).
    llvm::SmallVector<mlir::OpFoldResult> outputShape;
    outputShape.reserve(2);
    for (int64_t i : llvm::seq<int64_t>(2)) {
      if (!outputType.isDynamicDim(i)) {
        outputShape.push_back(rewriter.getIndexAttr(outputType.getDimSize(i)));
      } else {
        // Dynamic side mirrors the rank-1 source's single dim.
        mlir::Value dim = mlir::tensor::DimOp::create(rewriter, loc, src, 0);
        outputShape.push_back(dim);
      }
    }
    (void)srcType; // type info embedded in the value

    auto expandOp = mlir::tensor::ExpandShapeOp::create(
        rewriter, loc, outputType, src, expandReassoc, outputShape);
    rewriter.replaceOp(op, expandOp.getResult());
    return mlir::success();
  }
};

} // namespace

void populateFlattenConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<FlattenDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
