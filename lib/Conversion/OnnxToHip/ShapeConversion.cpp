/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ShapeConversion.cpp - lower onnx.Shape to tensor ops ---*- C++ -*-===//
//
// Lowers `onnx.Shape(input)` to a chain of `tensor.dim` queries combined
// via `tensor.from_elements`, so the Shape op participates in the standard
// shape-inference and bufferization machinery instead of surviving as an
// opaque ONNX-dialect op into the lowering passes.
//
// IR example
// ----------
//   Before (input has one dynamic batch dim):
//     %x : tensor<?x128x32xf16>
//     %s = onnx.Shape(%x) : tensor<3xi64>
//
//   After (start = 0, end = 3 — full shape):
//     %d0      = tensor.dim %x, %c0 : tensor<?x128x32xf16>
//     %d0_i64  = arith.index_cast %d0 : index to i64
//     %c128    = arith.constant 128 : i64
//     %c32     = arith.constant 32 : i64
//     %s       = tensor.from_elements %d0_i64, %c128, %c32 : tensor<3xi64>
//
// Why this pass exists
// --------------------
// With fully static shapes, `onnx.Shape` folds to a compile-time
// `onnx.Constant` that GenerateInterface places in the constants blob
// (already on GPU at init -- no per-step work).  Dynamic input dims break
// the constant fold for any shape that references a symbolic dim, so
// `Shape` then needs an actual runtime lowering.
//
// Lowering to `tensor.dim + tensor.from_elements` keeps the result in the
// tensor world long enough that one-shot-bufferize converts the chain into
// `memref.dim + memref.alloc + memref.store + ...` -- a form the rest of
// the pipeline (PromoteStridedHipOperands, MaterializeHostScalars,
// PoolAllocs) already knows how to handle.  Lowering directly to the
// memref dialect here would have to reinvent that chain.
//
// ONNX `Shape` carries optional `start`/`end` attributes that pick a
// sub-range of the input's dims.  Negative bounds are handled per spec
// (add rank, then clamp); the resulting `tensor.dim` ops reference absolute
// dim indices directly.
//
// Non-goals
// ---------
//   - Unranked input: returns notifyMatchFailure so the downstream
//     conversion framework surfaces a clear error rather than silently
//     producing an opaque Shape op.  See test_shape.mlir for the
//     negative-case lockdown.
//   - Reproducing the upstream constant-fold path: ONNX-spec canonicalization
//     handles fully-static Shape ops earlier in the pipeline.  This pass
//     fires only for the surviving (non-folded) cases, which by construction
//     reference at least one dynamic dim.
//   - Folding `onnx.Shape` followed by `onnx.Gather`: that idiom is handled
//     by `GatherShapeFold.cpp` BEFORE this pass runs.  The two passes are
//     intentionally orthogonal so they compose cleanly.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// Lower onnx.Shape(input) -> tensor<Nxi64> containing the (possibly
// sub-ranged) shape of the input.  Empty ranges (start >= end after
// normalization) emit a 0-element constant tensor per ONNX spec rather than
// failing the match -- preserving op semantics is the priority over
// surfacing a "shouldn't happen" error.
// Benefit 2 (Expand is 1): onnx.Shape must lower before onnx.Expand so
// ExpandConversion can traceback tensor.from_elements or onnx.Shape directly.
struct ShapeToTensorDims : public mlir::RewritePattern {
  ShapeToTensorDims(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");

    int64_t rank = inputType.getRank();

    // ONNX Shape op supports start/end attributes for partial shape queries
    int64_t start = 0;
    int64_t end = rank;
    if (auto startAttr = op->getAttrOfType<mlir::IntegerAttr>("start"))
      start = startAttr.getSInt();
    if (auto endAttr = op->getAttrOfType<mlir::IntegerAttr>("end"))
      end = endAttr.getSInt();

    if (start < 0)
      start += rank;
    if (end < 0)
      end += rank;
    start = std::max(start, int64_t(0));
    end = std::min(end, rank);

    auto i64Type = rewriter.getI64Type();

    // ONNX spec: when the requested range is empty (start >= end after
    // normalization), Shape returns a 1-D tensor with zero elements.
    // Returning notifyMatchFailure here would leave onnx.Shape in the IR
    // and crash later — emit the empty constant directly.
    if (start >= end) {
      auto emptyType = mlir::RankedTensorType::get({0}, i64Type);
      auto emptyAttr =
          mlir::DenseElementsAttr::get(emptyType, llvm::ArrayRef<int64_t>{});
      mlir::Value result =
          arith::ConstantOp::create(rewriter, loc, emptyType, emptyAttr);
      rewriter.replaceOp(op, result);
      return mlir::success();
    }

    llvm::SmallVector<mlir::Value> dimValues;
    for (int64_t i = start; i < end; ++i) {
      if (inputType.isDynamicDim(i)) {
        mlir::Value dimIdx = arith::ConstantIndexOp::create(rewriter, loc, i);
        mlir::Value dimVal =
            tensor::DimOp::create(rewriter, loc, input, dimIdx);
        mlir::Value dimI64 =
            arith::IndexCastOp::create(rewriter, loc, i64Type, dimVal);
        dimValues.push_back(dimI64);
      } else {
        int64_t staticDim = inputType.getDimSize(i);
        dimValues.push_back(arith::ConstantOp::create(
            rewriter, loc, rewriter.getI64IntegerAttr(staticDim)));
      }
    }

    mlir::Value result =
        tensor::FromElementsOp::create(rewriter, loc, dimValues);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

} // namespace

void populateShapeConversionPatterns(mlir::RewritePatternSet &patterns,
                                     mlir::MLIRContext *ctx) {
  patterns.add<ShapeToTensorDims>(ctx);
}

} // namespace hip
} // namespace mlir
