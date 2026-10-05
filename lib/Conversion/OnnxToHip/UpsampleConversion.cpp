/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <cmath>
#include <optional>

namespace mlir {
namespace hip {
namespace {

// onnx.Upsample (schema 9) is the deprecated form of Resize. ONNX defines it
// as Resize with coordinate_transformation_mode="asymmetric" and, for nearest,
// nearest_mode="floor". Those are hip.resize enum ids 1 and 2.
//
// Scales are not passed to the kernel. hip.resize recovers each axis as
// out_dim / in_dim from the result type the importer already computed
// (output_dim = floor(input_dim * scale)).
//
// Before:
//   %y = "onnx.Upsample"(%x, %scales) {mode = "linear"}
//        : (tensor<1x256x1x1xf32>, tensor<4xf32>) -> tensor<1x256x65x65xf32>
// After:
//   %init = tensor.empty() : tensor<1x256x65x65xf32>
//   %y = hip.resize(%ctx) ins(%x : tensor<1x256x1x1xf32>)
//                         outs(%init : tensor<1x256x65x65xf32>)
//                         {mode = 1, coord_transform = 1, nearest_mode = 2}

constexpr int64_t kResizeModeNearest = 0;
constexpr int64_t kResizeModeLinear = 1;
constexpr int64_t kResizeCoordAsymmetric = 1;
constexpr int64_t kResizeNearestFloor = 2;

std::optional<llvm::SmallVector<double>>
readConstantScales(mlir::Value scales) {
  mlir::Operation *def = scales.getDefiningOp();
  if (!def)
    return std::nullopt;
  auto attr = def->getAttrOfType<mlir::DenseElementsAttr>("value");
  if (!attr || !attr.getElementType().isF32())
    return std::nullopt;
  llvm::SmallVector<double> values;
  values.reserve(attr.getNumElements());
  for (llvm::APFloat value : attr.getValues<llvm::APFloat>())
    values.push_back(value.convertToDouble());
  return values;
}

struct UpsampleToResize : public mlir::RewritePattern {
  UpsampleToResize(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Upsample", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 2 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.Upsample expects X, scales, and one result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    mlir::Value input = op->getOperand(0);
    mlir::Value scales = op->getOperand(1);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    auto scalesType = mlir::dyn_cast<mlir::RankedTensorType>(scales.getType());
    if (!inputType || !outputType || !scalesType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensors");

    int64_t rank = inputType.getRank();
    if (rank < 3 || rank != outputType.getRank())
      return rewriter.notifyMatchFailure(
          op, "Upsample requires rank >= 3 and matching in/out ranks");
    int64_t spatialRank = rank - 2;
    if (spatialRank < 1 || spatialRank > 3)
      return rewriter.notifyMatchFailure(
          op, "only 1D / 2D / 3D spatial Upsample supported");
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
        inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "Upsample runtime supports only matching float types");
    if (!scalesType.getElementType().isF32() || scalesType.getRank() != 1 ||
        (scalesType.hasStaticShape() && scalesType.getDimSize(0) != rank))
      return rewriter.notifyMatchFailure(
          op, "Upsample scales must be a rank-1 f32 tensor of length rank");

    for (int64_t i : llvm::seq<int64_t>(2)) {
      if (!inputType.isDynamicDim(i) && !outputType.isDynamicDim(i) &&
          inputType.getDimSize(i) != outputType.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "Upsample: only spatial-axis resampling supported");
    }
    for (int64_t i : llvm::seq<int64_t>(spatialRank)) {
      if (outputType.isDynamicDim(2 + i))
        return rewriter.notifyMatchFailure(
            op, "Upsample: dynamic output spatial dims not supported");
    }

    if (auto scaleValues = readConstantScales(scales)) {
      if (static_cast<int64_t>(scaleValues->size()) != rank)
        return rewriter.notifyMatchFailure(
            op, "Upsample scales length must equal the input rank");
      for (int64_t i : llvm::seq<int64_t>(rank)) {
        double scale = (*scaleValues)[i];
        if (!(scale > 0.0) || !std::isfinite(scale))
          return rewriter.notifyMatchFailure(
              op, "Upsample scales must be finite and positive");
        if (inputType.isDynamicDim(i) || outputType.isDynamicDim(i))
          continue;
        auto expected = static_cast<int64_t>(
            std::floor(static_cast<double>(inputType.getDimSize(i)) * scale));
        if (expected != outputType.getDimSize(i))
          return rewriter.notifyMatchFailure(
              op, "Upsample result shape must be floor(input * scales)");
      }
    }

    std::string mode = "nearest";
    if (auto attr = op->getAttrOfType<mlir::StringAttr>("mode"))
      mode = attr.getValue().str();
    int64_t modeId;
    if (mode == "nearest")
      modeId = kResizeModeNearest;
    else if (mode == "linear")
      modeId = kResizeModeLinear;
    else
      return rewriter.notifyMatchFailure(
          op, "Upsample mode must be 'nearest' or 'linear'");

    mlir::Location loc = op->getLoc();
    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i : llvm::seq<int64_t>(2)) {
      if (outputType.isDynamicDim(i))
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, input, i));
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                      outputType.getElementType(), dynSizes);
    auto hipOp = mlir::hip::ResizeOp::create(
        rewriter, loc, outputType, *ctxOrFailure, input, init,
        rewriter.getI64IntegerAttr(modeId),
        rewriter.getI64IntegerAttr(kResizeCoordAsymmetric),
        rewriter.getI64IntegerAttr(kResizeNearestFloor));
    rewriter.replaceOp(op, hipOp.getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateUpsampleConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<UpsampleToResize>(ctx);
}

} // namespace hip
} // namespace mlir
