/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Concat lowering
//===----------------------------------------------------------------------===//
//
// ONNX Concat is fully expressible at compile time via
// `tensor.empty` + N x `tensor.insert_slice`:
//
//   - Build an output buffer with the inferred result shape.
//   - For each input i, emit one `tensor.insert_slice` that writes the
//     input into the running output at offset = sum of preceding inputs'
//     extents along the concat axis (offset is 0 on every other axis).
//
// `tensor.insert_slice` accepts mixed static / dynamic offsets and sizes
// via `OpFoldResult`, so the same pattern covers both fully static and
// fully dynamic Concat operands. Bufferization later rewrites each
// insert into a `memref.subview` + `memref.copy` against a pooled output
// buffer; no concat-specific runtime kernel is needed.
//
// Before:
//
//   %r = "onnx.Concat"(%a, %b, %c) {axis = 1 : si64}
//       : (tensor<2x3xf32>, tensor<2x4xf32>, tensor<2x5xf32>)
//       -> tensor<2x12xf32>
//
// After:
//
//   %init = tensor.empty() : tensor<2x12xf32>
//   %0 = tensor.insert_slice %a into %init[0, 0] [2, 3] [1, 1]
//       : tensor<2x3xf32> into tensor<2x12xf32>
//   %1 = tensor.insert_slice %b into %0   [0, 3] [2, 4] [1, 1]
//       : tensor<2x4xf32> into tensor<2x12xf32>
//   %r = tensor.insert_slice %c into %1   [0, 7] [2, 5] [1, 1]
//       : tensor<2x5xf32> into tensor<2x12xf32>
//
// Dynamic case (mixed static / dynamic along axis):
//
// Before:
//
//   %r = "onnx.Concat"(%a, %b) {axis = 0 : si64}
//       : (tensor<?x4xf16>, tensor<3x4xf16>) -> tensor<?x4xf16>
//
// After (offset for %b is %dim_a, output dim 0 is %dim_a + 3):
//
//   %c0   = arith.constant 0 : index
//   %dim_a = tensor.dim %a, %c0 : tensor<?x4xf16>
//   %c3   = arith.constant 3 : index
//   %out_d0 = arith.addi %dim_a, %c3 : index
//   %init = tensor.empty(%out_d0) : tensor<?x4xf16>
//   %0    = tensor.insert_slice %a into %init[0, 0] [%dim_a, 4] [1, 1]
//   %r    = tensor.insert_slice %b into %0  [%dim_a, 0] [3, 4] [1, 1]

struct ConcatDecompose : public mlir::RewritePattern {
  ConcatDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Concat", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() == 0 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected >=1 input, 1 output");

    auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis");
    if (!axisAttr)
      return rewriter.notifyMatchFailure(op, "missing 'axis' attribute");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result is not a ranked tensor");
    int64_t rank = resultType.getRank();
    if (rank == 0)
      return rewriter.notifyMatchFailure(op, "rank-0 Concat is not allowed");

    // ONNX `axis` is a signed integer attribute (`si64`); `IntegerAttr::getInt`
    // asserts the underlying type is signless/index. Use `getSInt` to extract
    // the value with the correct sign-extension for ONNX's signed encoding.
    int64_t axis = axisAttr.getSInt();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "axis out of range");

    // All inputs must be ranked tensors of the same rank and element type
    // as the result. ONNX Concat allows differing sizes only along `axis`;
    // sizes along the other axes must either match or be ?/static-mismatched
    // in a way the IR-supplied result type already encodes.
    llvm::SmallVector<mlir::RankedTensorType> inputTypes;
    inputTypes.reserve(op->getNumOperands());
    for (mlir::Value input : op->getOperands()) {
      auto t = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
      if (!t || t.getRank() != rank ||
          t.getElementType() != resultType.getElementType())
        return rewriter.notifyMatchFailure(
            op, "input rank / element type mismatch with result");
      inputTypes.push_back(t);
    }

    mlir::Location loc = op->getLoc();

    // static/dynamic shape unified handling
    auto getDimExtent = [&](mlir::Value input, mlir::RankedTensorType type,
                            int64_t dim) -> mlir::OpFoldResult {
      if (type.isDynamicDim(dim))
        return mlir::OpFoldResult(
            mlir::tensor::DimOp::create(rewriter, loc, input, dim).getResult());
      return rewriter.getIndexAttr(type.getDimSize(dim));
    };

    // Build the dynamic-size operand list for `tensor.empty` against the
    // (already inferred) result type. Untouched dims forward from the
    // first input that has a static value for that dim (every other input
    // must agree by ONNX semantics); if no input has a static value, we
    // emit a tensor.dim on input 0. The concat axis is the sum of all
    // inputs' axis-dim extents.
    auto materializeExtent = [&](mlir::OpFoldResult fr) -> mlir::Value {
      if (auto v = llvm::dyn_cast_if_present<mlir::Value>(fr))
        return v;
      auto intAttr = mlir::cast<mlir::IntegerAttr>(
          llvm::dyn_cast_if_present<mlir::Attribute>(fr));
      return mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                  intAttr.getInt())
          .getResult();
    };

    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t d : llvm::seq<int64_t>(rank)) {
      if (!resultType.isDynamicDim(d))
        continue;

      if (d == axis) {
        // Sum of all inputs' extents along `axis`.
        mlir::Value sum;
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
          mlir::Value extent = materializeExtent(
              getDimExtent(op->getOperand(i), inputTypes[i], axis));
          sum = i == 0 ? extent
                       : mlir::arith::AddIOp::create(rewriter, loc, sum, extent)
                             .getResult();
        }
        dynSizes.push_back(sum);
      } else {
        // Untouched dim: prefer a static input value; else fall back to
        // input 0's runtime size.
        mlir::Value srcDim;
        for (size_t i = 0; i < op->getNumOperands(); ++i) {
          if (!inputTypes[i].isDynamicDim(d)) {
            srcDim = mlir::arith::ConstantIndexOp::create(
                         rewriter, loc, inputTypes[i].getDimSize(d))
                         .getResult();
            break;
          }
        }
        if (!srcDim)
          srcDim =
              mlir::tensor::DimOp::create(rewriter, loc, op->getOperand(0), d)
                  .getResult();
        dynSizes.push_back(srcDim);
      }
    }

    mlir::Value current =
        mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                      resultType.getElementType(), dynSizes)
            .getResult();

    // Insert each input into `current` at the running axis offset. We
    // track the offset as an OpFoldResult: a static int64 fast path
    // (collapsed into an IndexAttr) is used until we encounter a
    // dynamic-axis input, at which point we switch to an arith.addi
    // chain. This keeps the IR clean in the common all-static case.
    int64_t staticOffset = 0;
    mlir::Value dynamicOffset; // null while staticOffset alone is exact.

    for (size_t i = 0; i < op->getNumOperands(); ++i) {
      mlir::Value input = op->getOperand(i);
      mlir::RankedTensorType inputType = inputTypes[i];

      // Per-axis (offset, size, stride) lists for tensor.insert_slice.
      llvm::SmallVector<mlir::OpFoldResult> offsets, sizes, strides;
      offsets.reserve(rank);
      sizes.reserve(rank);
      strides.assign(rank, rewriter.getIndexAttr(1));

      for (int64_t d : llvm::seq<int64_t>(rank)) {
        if (d == axis) {
          if (!dynamicOffset) {
            offsets.push_back(rewriter.getIndexAttr(staticOffset));
          } else if (staticOffset == 0) {
            offsets.push_back(dynamicOffset);
          } else {
            mlir::Value c = mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                                 staticOffset)
                                .getResult();
            offsets.push_back(
                mlir::arith::AddIOp::create(rewriter, loc, dynamicOffset, c)
                    .getResult());
          }
        } else {
          offsets.push_back(rewriter.getIndexAttr(0));
        }
        sizes.push_back(getDimExtent(input, inputType, d));
      }

      current = mlir::tensor::InsertSliceOp::create(
                    rewriter, loc, input, current, offsets, sizes, strides)
                    .getResult();

      // Advance the running axis offset by this input's axis-dim extent.
      if (inputType.isDynamicDim(axis)) {
        mlir::Value extent =
            mlir::tensor::DimOp::create(rewriter, loc, input, axis).getResult();
        if (!dynamicOffset && staticOffset == 0) {
          dynamicOffset = extent;
        } else if (!dynamicOffset) {
          mlir::Value c =
              mlir::arith::ConstantIndexOp::create(rewriter, loc, staticOffset)
                  .getResult();
          dynamicOffset =
              mlir::arith::AddIOp::create(rewriter, loc, c, extent).getResult();
          staticOffset = 0;
        } else {
          dynamicOffset =
              mlir::arith::AddIOp::create(rewriter, loc, dynamicOffset, extent)
                  .getResult();
        }
      } else {
        staticOffset += inputType.getDimSize(axis);
      }
    }

    rewriter.replaceOp(op, current);
    return mlir::success();
  }
};

} // namespace

void populateConcatConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ConcatDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
