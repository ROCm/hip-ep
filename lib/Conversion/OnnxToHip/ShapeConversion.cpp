/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// onnx.Shape(input) -> tensor<Nxi64> containing the shape of the input.
// With static shapes this is folded away to a constant. With dynamic shapes
// it survives into the conversion pass. Lower it to tensor.dim +
// tensor.from_elements so the standard MLIR bufferization can handle it.
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

void mlir::hip::populateShapeConversionPatterns(
    mlir::RewritePatternSet &patterns, mlir::MLIRContext *ctx) {
  patterns.add<ShapeToTensorDims>(ctx);
}

} // namespace hip
} // namespace mlir
