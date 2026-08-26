/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// onnx.GridSample -> hip.grid_sample (4-D NCHW)
//===----------------------------------------------------------------------===//
//
// Covers the ONNX GridSample subset used by BEV / STN exporters:
//   * 4-D input (N, C, H_in, W_in)
//   * 4-D grid  (N, H_out, W_out, 2) with last dim (x, y) in [-1, 1]
//   * mode in {"bilinear", "linear", "nearest"}   (no cubic)
//   * padding_mode in {"zeros", "border", "reflection"}
//   * align_corners in {0, 1}
//
// Output shape is (N, C, H_out, W_out). Dynamic N/C come from the input;
// dynamic H_out/W_out come from the grid.
//
// Before:
//   %y = "onnx.GridSample"(%x, %grid)
//          {mode = "bilinear", padding_mode = "zeros", align_corners = 0 :
//          si64} : (tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>) ->
//          tensor<1x3x4x4xf32>
//
// After:
//   %init = tensor.empty() : tensor<1x3x4x4xf32>
//   %y = hip.grid_sample(%ctx)
//          ins(%x, %grid : tensor<1x3x8x8xf32>, tensor<1x4x4x2xf32>)
//          outs(%init : tensor<1x3x4x4xf32>)
//          {mode = 1, padding_mode = 0, align_corners = 0}

struct GridSampleToHip : public mlir::RewritePattern {
  GridSampleToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.GridSample", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "GridSample expects 2 operands (X, grid) and 1 result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = op->getOperand(0);
    mlir::Value grid = op->getOperand(1);

    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto gridType = mlir::dyn_cast<mlir::RankedTensorType>(grid.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !gridType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");
    if (inputType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "only 4-D GridSample (N, C, H, W) is supported");
    if (gridType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "grid must be 4-D (N, H_out, W_out, 2)");
    if (outputType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "output must be 4-D (N, C, H_out, W_out)");
    if (!gridType.isDynamicDim(3) && gridType.getDimSize(3) != 2)
      return rewriter.notifyMatchFailure(op, "grid last dim must be 2");
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
        inputType.getElementType() != outputType.getElementType() ||
        inputType.getElementType() != gridType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "GridSample requires matching float types on X, grid, and Y");

    auto getStrAttr = [&](mlir::StringRef name,
                          mlir::StringRef defaultVal) -> std::string {
      if (auto attr = op->getAttrOfType<mlir::StringAttr>(name))
        return attr.getValue().str();
      return defaultVal.str();
    };

    std::string mode = getStrAttr("mode", "linear");
    int64_t modeId;
    if (mode == "nearest")
      modeId = 0;
    else if (mode == "bilinear" || mode == "linear")
      modeId = 1;
    else
      return rewriter.notifyMatchFailure(
          op, "GridSample mode must be 'nearest', 'bilinear', or 'linear'");

    std::string padding = getStrAttr("padding_mode", "zeros");
    int64_t paddingId;
    if (padding == "zeros")
      paddingId = 0;
    else if (padding == "border")
      paddingId = 1;
    else if (padding == "reflection")
      paddingId = 2;
    else
      return rewriter.notifyMatchFailure(
          op, "GridSample padding_mode must be 'zeros', 'border', or "
              "'reflection'");

    int64_t alignCorners = 0;
    if (auto a = op->getAttrOfType<mlir::IntegerAttr>("align_corners"))
      alignCorners = a.getValue().getSExtValue();
    if (alignCorners != 0 && alignCorners != 1)
      return rewriter.notifyMatchFailure(op, "align_corners must be 0 or 1");

    // Output (N, C, H_out, W_out): N/C from input, spatial from grid.
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i : llvm::seq<int64_t>(outputType.getRank())) {
      if (!outputType.isDynamicDim(i))
        continue;
      if (i == 0 || i == 1)
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, input, i));
      else
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, grid, i - 1));
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                      outputType.getElementType(), dynSizes);

    auto hipOp = mlir::hip::GridSampleOp::create(
        rewriter, loc, context, input, grid, init,
        rewriter.getI64IntegerAttr(modeId),
        rewriter.getI64IntegerAttr(paddingId),
        rewriter.getI64IntegerAttr(alignCorners));
    rewriter.replaceOp(op, hipOp.getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateGridSampleConversionPatterns(RewritePatternSet &patterns,
                                          MLIRContext *ctx) {
  patterns.add<GridSampleToHip>(ctx);
}

} // namespace hip
} // namespace mlir
