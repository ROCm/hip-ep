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

// onnx.Upsample is the deprecated form of Resize. ONNX defines it as Resize
// with coordinate_transformation_mode="asymmetric" and, for nearest,
// nearest_mode="floor". Those are hip.resize enum ids 1 and 2.
//
// Scales are not passed to the kernel. hip.resize recovers each axis as
// out_dim / in_dim from the result type the importer already computed
// (output_dim = floor(input_dim * scale)).
//
// Schema 9 passes scales as a second input. Opset 7 and 8 store that same
// list as an f32 attribute and have only the X input. The importer prints
// the attribute as an array of f32.
//
// Before (schema 9):
//   %y = "onnx.Upsample"(%x, %scales) {mode = "linear"}
//        : (tensor<1x256x1x1xf32>, tensor<4xf32>) -> tensor<1x256x65x65xf32>
// Before (opset 7-8):
//   %y = "onnx.Upsample"(%x) {mode = "nearest",
//        scales = [1.0 : f32, 1.0 : f32, 2.0 : f32, 2.0 : f32]}
//        : (tensor<1x128x56x56xf32>) -> tensor<1x128x112x112xf32>
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

// Opset <= 8. The importer uses an ArrayAttr of f32. A dense f32 attribute
// is the same list written as an MLIR elements attribute.
std::optional<llvm::SmallVector<double>>
readScalesAttribute(mlir::Operation *op) {
  if (auto dense = op->getAttrOfType<mlir::DenseElementsAttr>("scales")) {
    if (!dense.getElementType().isF32())
      return std::nullopt;
    llvm::SmallVector<double> values;
    values.reserve(dense.getNumElements());
    for (llvm::APFloat value : dense.getValues<llvm::APFloat>())
      values.push_back(value.convertToDouble());
    return values;
  }
  auto array = op->getAttrOfType<mlir::ArrayAttr>("scales");
  if (!array)
    return std::nullopt;
  llvm::SmallVector<double> values;
  values.reserve(array.size());
  for (mlir::Attribute element : array) {
    auto floatAttr = mlir::dyn_cast<mlir::FloatAttr>(element);
    if (!floatAttr || !floatAttr.getType().isF32())
      return std::nullopt;
    values.push_back(floatAttr.getValue().convertToDouble());
  }
  return values;
}

mlir::LogicalResult checkConstantScales(mlir::Operation *op,
                                        mlir::PatternRewriter &rewriter,
                                        mlir::RankedTensorType inputType,
                                        mlir::RankedTensorType outputType,
                                        llvm::ArrayRef<double> scaleValues) {
  int64_t rank = inputType.getRank();
  if (static_cast<int64_t>(scaleValues.size()) != rank)
    return rewriter.notifyMatchFailure(
        op, "Upsample scales length must equal the input rank");
  for (int64_t i : llvm::seq<int64_t>(rank)) {
    double scale = scaleValues[i];
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
  return mlir::success();
}

struct UpsampleToResize : public mlir::RewritePattern {
  UpsampleToResize(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Upsample", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op,
                                         "onnx.Upsample expects one result");
    // Schema 9: (X, scales). Opset 7-8: (X) plus a scales attribute.
    const bool scalesAreInput = op->getNumOperands() == 2;
    const bool scalesAreAttribute = op->getNumOperands() == 1;
    if (!scalesAreInput && !scalesAreAttribute)
      return rewriter.notifyMatchFailure(
          op, "onnx.Upsample expects X and a scales input or attribute");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();

    mlir::Value input = op->getOperand(0);
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
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
    std::optional<llvm::SmallVector<double>> scaleValues;
    if (scalesAreInput) {
      mlir::Value scales = op->getOperand(1);
      auto scalesType =
          mlir::dyn_cast<mlir::RankedTensorType>(scales.getType());
      if (!scalesType || !scalesType.getElementType().isF32() ||
          scalesType.getRank() != 1 ||
          (scalesType.hasStaticShape() && scalesType.getDimSize(0) != rank))
        return rewriter.notifyMatchFailure(
            op, "Upsample scales must be a rank-1 f32 tensor of length rank");
      scaleValues = readConstantScales(scales);
    } else {
      // Opset 7-8 has no scales operand. A missing or non-f32 attribute
      // leaves the op unconverted.
      scaleValues = readScalesAttribute(op);
      if (!scaleValues)
        return rewriter.notifyMatchFailure(
            op, "opset 7-8 Upsample requires an f32 scales attribute");
    }

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

    if (scaleValues && mlir::failed(checkConstantScales(
                           op, rewriter, inputType, outputType, *scaleValues)))
      return mlir::failure();

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
