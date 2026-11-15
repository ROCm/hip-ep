/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// onnx.RoiAlign -> hip.roi_align
//===----------------------------------------------------------------------===//
//
// Operands: X (N, C, H, W), rois (num_rois, 4), batch_indices (num_rois) i64.
// Result:   Y (num_rois, C, output_height, output_width).
//
// Every extent may be dynamic. Unranked operands are cast to the rank the
// spec requires; num_rois comes from rois and C from X at runtime.
//
// Before:
//   %y = "onnx.RoiAlign"(%x, %rois, %idx)
//          {mode = "avg", output_height = 7 : si64, output_width = 7 : si64,
//           sampling_ratio = 2 : si64, spatial_scale = 2.500000e-01 : f32}
//          : (tensor<1x256x128x128xf32>, tensor<?x4xf32>, tensor<?xi64>)
//          -> tensor<?x256x7x7xf32>
//
// After:
//   %c0 = arith.constant 0 : index
//   %n = tensor.dim %rois, %c0 : tensor<?x4xf32>
//   %init = tensor.empty(%n) : tensor<?x256x7x7xf32>
//   %y = hip.roi_align(%ctx)
//          ins(%x, %rois, %idx : ...) outs(%init : tensor<?x256x7x7xf32>)
//          {half_pixel = 1, mode = 0, output_height = 7, output_width = 7,
//           sampling_ratio = 2, spatial_scale = 2.500000e-01 : f32}

/// The ranked type `value` is used as: its own type when it already has
/// `rank`, all-dynamic extents when unranked, null otherwise. Builds no IR.
mlir::RankedTensorType rankedTypeFor(mlir::Value value, int64_t rank) {
  mlir::Type type = value.getType();
  if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type))
    return ranked.getRank() == rank ? ranked : mlir::RankedTensorType();
  auto unranked = mlir::dyn_cast<mlir::UnrankedTensorType>(type);
  if (!unranked)
    return {};
  llvm::SmallVector<int64_t> shape(rank, mlir::ShapedType::kDynamic);
  return mlir::RankedTensorType::get(shape, unranked.getElementType());
}

mlir::Value castTo(mlir::PatternRewriter &rewriter, mlir::Location loc,
                   mlir::Value value, mlir::RankedTensorType type) {
  if (value.getType() == type)
    return value;
  return mlir::tensor::CastOp::create(rewriter, loc, type, value).getResult();
}

struct RoiAlignToHip : public mlir::RewritePattern {
  RoiAlignToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.RoiAlign", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 3 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "RoiAlign expects 3 operands (X, rois, batch_indices) and 1 "
              "result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Type resultType = op->getResult(0).getType();
    auto resultShaped = mlir::dyn_cast<mlir::TensorType>(resultType);
    if (!resultShaped)
      return rewriter.notifyMatchFailure(op, "expected tensor result");
    mlir::Type elemTy = resultShaped.getElementType();
    if (!elemTy.isF32() && !elemTy.isF16() && !elemTy.isBF16() &&
        !elemTy.isF64())
      return rewriter.notifyMatchFailure(
          op, "RoiAlign supports f32, f16, bf16 and f64");

    auto getElemTy = [](mlir::Value v) {
      return mlir::cast<mlir::TensorType>(v.getType()).getElementType();
    };
    if (getElemTy(op->getOperand(0)) != elemTy ||
        getElemTy(op->getOperand(1)) != elemTy)
      return rewriter.notifyMatchFailure(
          op, "RoiAlign requires matching float types on X, rois, and Y");
    if (!getElemTy(op->getOperand(2)).isInteger(64))
      return rewriter.notifyMatchFailure(op, "batch_indices must be i64");

    if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(resultType))
      if (ranked.getRank() != 4)
        return rewriter.notifyMatchFailure(op, "expected 4-D result");

    auto getIntAttr = [&](mlir::StringRef name, int64_t defaultVal) {
      if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
        return attr.getValue().getSExtValue();
      return defaultVal;
    };
    auto getStrAttr = [&](mlir::StringRef name,
                          mlir::StringRef defaultVal) -> std::string {
      if (auto attr = op->getAttrOfType<mlir::StringAttr>(name))
        return attr.getValue().str();
      return defaultVal.str();
    };

    std::string mode = getStrAttr("mode", "avg");
    int64_t modeId;
    if (mode == "avg")
      modeId = 0;
    else if (mode == "max")
      modeId = 1;
    else
      return rewriter.notifyMatchFailure(op,
                                         "RoiAlign mode must be avg or max");

    // half_pixel is the opset-16 default. RoiAlign-10 has no attribute and
    // samples like output_half_pixel; the ORT import records that explicitly.
    std::string ctm =
        getStrAttr("coordinate_transformation_mode", "half_pixel");
    int64_t halfPixel;
    if (ctm == "half_pixel")
      halfPixel = 1;
    else if (ctm == "output_half_pixel")
      halfPixel = 0;
    else
      return rewriter.notifyMatchFailure(
          op, "coordinate_transformation_mode must be half_pixel or "
              "output_half_pixel");

    int64_t outH = getIntAttr("output_height", 1);
    int64_t outW = getIntAttr("output_width", 1);
    int64_t samplingRatio = getIntAttr("sampling_ratio", 0);
    if (outH < 1 || outW < 1)
      return rewriter.notifyMatchFailure(
          op, "output_height and output_width must be >= 1");
    if (samplingRatio < 0)
      return rewriter.notifyMatchFailure(op, "sampling_ratio must be >= 0");

    float spatialScale = 1.0f;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("spatial_scale"))
      spatialScale = static_cast<float>(attr.getValueAsDouble());

    // Every check runs before the first cast, so a failure leaves the IR
    // unchanged.
    auto inputType = rankedTypeFor(op->getOperand(0), 4);
    auto roisType = rankedTypeFor(op->getOperand(1), 2);
    auto indicesType = rankedTypeFor(op->getOperand(2), 1);
    if (!inputType || !roisType || !indicesType)
      return rewriter.notifyMatchFailure(
          op, "expected 4-D X, 2-D rois and 1-D batch_indices");
    if (!roisType.isDynamicDim(1) && roisType.getDimSize(1) != 4)
      return rewriter.notifyMatchFailure(op, "rois last dim must be 4");

    mlir::Value input = castTo(rewriter, loc, op->getOperand(0), inputType);
    mlir::Value rois = castTo(rewriter, loc, op->getOperand(1), roisType);
    mlir::Value indices = castTo(rewriter, loc, op->getOperand(2), indicesType);

    // Static extents from whichever side knows them; the result type of the
    // ONNX op wins when it is more precise than the operands.
    auto resultRanked = mlir::dyn_cast<mlir::RankedTensorType>(resultType);
    auto pick = [](int64_t a, int64_t b) {
      return mlir::ShapedType::isDynamic(a) ? b : a;
    };
    int64_t numRois = pick(roisType.getDimSize(0), indicesType.getDimSize(0));
    int64_t channels = inputType.getDimSize(1);
    if (resultRanked) {
      numRois = pick(resultRanked.getDimSize(0), numRois);
      channels = pick(resultRanked.getDimSize(1), channels);
    }

    llvm::SmallVector<mlir::Value> dynSizes;
    if (mlir::ShapedType::isDynamic(numRois))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, rois, 0));
    if (mlir::ShapedType::isDynamic(channels))
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, input, 1));

    auto outputType =
        mlir::RankedTensorType::get({numRois, channels, outH, outW}, elemTy);
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, outputType.getShape(), elemTy, dynSizes);

    auto hipOp = mlir::hip::RoiAlignOp::create(
        rewriter, loc, context, input, rois, indices, init,
        rewriter.getI64IntegerAttr(modeId), rewriter.getI64IntegerAttr(outH),
        rewriter.getI64IntegerAttr(outW),
        rewriter.getI64IntegerAttr(samplingRatio),
        rewriter.getF32FloatAttr(spatialScale),
        rewriter.getI64IntegerAttr(halfPixel));

    mlir::Value result = hipOp.getResult(0);
    if (result.getType() != resultType)
      result = mlir::tensor::CastOp::create(rewriter, loc, resultType, result);
    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

} // namespace

void populateRoiAlignConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<RoiAlignToHip>(ctx);
}

} // namespace hip
} // namespace mlir
