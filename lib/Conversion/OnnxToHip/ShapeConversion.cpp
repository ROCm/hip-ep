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
// Why this pass exists
// --------------------
// Pre-dynseqlen, every input was statically shaped and `onnx.Shape` folded
// to a compile-time `onnx.Constant` that GenerateInterface placed in the
// constants blob (already on GPU at init -- no per-step work).  Dynseqlen
// breaks the constant fold for inputs whose dims are symbolic, so `Shape`
// now needs an actual runtime lowering.
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
struct ShapeToTensorDims : public mlir::RewritePattern {
  ShapeToTensorDims(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Shape", /*benefit=*/1, ctx) {}

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
