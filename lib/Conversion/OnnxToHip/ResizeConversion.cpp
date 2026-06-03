/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// onnx.Resize -> hip.resize (native, spatial-only resampling)
//===----------------------------------------------------------------------===//
//
// ONNX Resize is a multi-dimensional resampler whose full attribute surface
// is huge.  This pass implements the slice that covers ONNX exporters'
// common image / volume use-case:
//
//   * mode in {"nearest", "linear"}                                  (no cubic)
//   * coordinate_transformation_mode in
//       {"half_pixel", "asymmetric", "align_corners"}
//   * nearest_mode = "round_prefer_floor"            (ONNX default)
//   * antialias = 0, exclude_outside = 0             (defaults)
//   * keep_aspect_ratio_policy = "stretch"           (default)
//   * roi must be absent / NoValue                   (no tf_crop_and_resize)
//   * scale on the leading two axes (N, C) must be 1 — i.e. resampling
//     happens only on trailing spatial axes, which is the universal
//     contract for image/volume CNNs and is what onnx-mlir's importer
//     produces for typical exports.
//
// The actual `scales` / `sizes` operand is NOT passed through to runtime —
// the upstream importer has already used it to compute the static result
// type, and per-axis scale is recovered at runtime as `in_dim / out_dim`.
//
// Compile-time work:
//   * decode the three string attributes into i64 enums baked onto the
//     hip.resize op
//   * verify the (N, C) axes are pass-through (input & output extents
//     match)
//
// Before:
//   %y = "onnx.Resize"(%x, %roi, %scales, %sizes)
//          {mode = "linear", coordinate_transformation_mode = "half_pixel"}
//          : (tensor<1x3x16x16xf16>, none, tensor<4xf32>, none)
//          -> tensor<1x3x32x32xf16>
//
// After:
//   %dim0 = tensor.dim %x, %c0          // only when N is dynamic
//   %init = tensor.empty(%dim0) : tensor<?x3x32x32xf16>
//   %y = hip.resize(%ctx) ins(%x : tensor<?x3x16x16xf16>)
//                         outs(%init : tensor<?x3x32x32xf16>)
//                         {mode = 1, coord_transform = 0, nearest_mode = 0}

struct ResizeToHip : public mlir::RewritePattern {
  ResizeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Resize", /*benefit=*/1, ctx) {}

  static bool isAbsent(mlir::Value v) {
    return !v || mlir::isa<mlir::NoneType>(v.getType());
  }

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected single output");

    // ONNX Resize accepts (X, roi?, scales?, sizes?) — between 1 and 4
    // operands depending on what the exporter supplied.  We accept any
    // count but require `roi` (operand 1) to be NoValue.
    auto operands = op->getOperands();
    if (operands.empty())
      return rewriter.notifyMatchFailure(op, "no input");
    if (operands.size() >= 2 && !isAbsent(operands[1]))
      return rewriter.notifyMatchFailure(
          op, "Resize: roi (tf_crop_and_resize) not supported");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value input = operands[0];
    auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(input.getType());
    auto outputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked tensor types");

    int64_t rank = inputType.getRank();
    if (rank < 3 || rank != outputType.getRank())
      return rewriter.notifyMatchFailure(
          op, "Resize requires rank >= 3 and matching in/out ranks");
    int64_t spatialRank = rank - 2;
    if (spatialRank < 1 || spatialRank > 3)
      return rewriter.notifyMatchFailure(
          op, "only 1D / 2D / 3D spatial Resize supported");
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
        inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "Resize runtime supports only matching float types");

    // (N, C) must be pass-through.  If both extents are static and differ,
    // bail.  Dynamic on either side is OK as long as both are dynamic
    // (tensor.dim of input feeds tensor.empty for the output).
    for (int64_t i : llvm::seq<int64_t>(2)) {
      bool inDyn = inputType.isDynamicDim(i);
      bool outDyn = outputType.isDynamicDim(i);
      if (!inDyn && !outDyn &&
          inputType.getDimSize(i) != outputType.getDimSize(i))
        return rewriter.notifyMatchFailure(
            op, "Resize: only spatial-axis resampling supported "
                "(N, C must match between input and output)");
    }

    // ===== Decode string attrs to enum-like i64 values =====================

    auto getStrAttr = [&](mlir::StringRef name,
                          mlir::StringRef defaultVal) -> std::string {
      if (auto attr = op->getAttrOfType<mlir::StringAttr>(name))
        return attr.getValue().str();
      return defaultVal.str();
    };

    std::string mode = getStrAttr("mode", "nearest");
    int64_t modeId;
    if (mode == "nearest")
      modeId = 0;
    else if (mode == "linear")
      modeId = 1;
    else
      return rewriter.notifyMatchFailure(op, "Resize mode must be "
                                             "'nearest' or 'linear'");

    std::string ct = getStrAttr("coordinate_transformation_mode", "half_pixel");
    int64_t coordId;
    if (ct == "half_pixel")
      coordId = 0;
    else if (ct == "asymmetric")
      coordId = 1;
    else if (ct == "align_corners")
      coordId = 2;
    else
      return rewriter.notifyMatchFailure(
          op, "Resize coordinate_transformation_mode must be one of "
              "{half_pixel, asymmetric, align_corners}");

    std::string nm = getStrAttr("nearest_mode", "round_prefer_floor");
    if (nm != "round_prefer_floor")
      return rewriter.notifyMatchFailure(
          op, "Resize nearest_mode must be round_prefer_floor");
    int64_t nearestId = 0;

    // Reject features outside the supported subset.
    auto getI64 = [&](mlir::StringRef name, int64_t defaultVal) -> int64_t {
      if (auto attr = op->getAttrOfType<mlir::IntegerAttr>(name))
        return attr.getValue().getSExtValue();
      return defaultVal;
    };
    if (getI64("antialias", 0) != 0)
      return rewriter.notifyMatchFailure(op, "antialias not supported");
    if (getI64("exclude_outside", 0) != 0)
      return rewriter.notifyMatchFailure(op, "exclude_outside not supported");
    if (auto a = op->getAttrOfType<mlir::ArrayAttr>("axes"))
      if (!a.empty())
        return rewriter.notifyMatchFailure(
            op, "explicit `axes` attribute not supported (defaults only)");
    std::string kar = getStrAttr("keep_aspect_ratio_policy", "stretch");
    if (kar != "stretch")
      return rewriter.notifyMatchFailure(
          op, "keep_aspect_ratio_policy must be 'stretch'");

    // ===== Build DPS init =================================================
    //
    // Output shape: leading (N, C) inherit from the input (their dynamic
    // status is required to match by the check above).  Trailing spatial
    // dims must be static — at runtime the kernel reads them off the
    // output memref descriptor; if they were dynamic we'd need arith ops
    // for the per-axis output extents, left out of scope.
    for (int64_t i : llvm::seq<int64_t>(spatialRank)) {
      if (outputType.isDynamicDim(2 + i))
        return rewriter.notifyMatchFailure(
            op, "Resize: dynamic output spatial dims not supported");
    }

    llvm::SmallVector<mlir::Value> dynSizes;
    for (int64_t i : llvm::seq<int64_t>(rank)) {
      if (outputType.isDynamicDim(i)) {
        if (i >= 2)
          return rewriter.notifyMatchFailure(op, "unreachable");
        dynSizes.push_back(
            mlir::tensor::DimOp::create(rewriter, loc, input, i));
      }
    }
    mlir::Value init =
        mlir::tensor::EmptyOp::create(rewriter, loc, outputType.getShape(),
                                      outputType.getElementType(), dynSizes);

    auto modeAttr = rewriter.getI64IntegerAttr(modeId);
    auto coordAttr = rewriter.getI64IntegerAttr(coordId);
    auto nearestAttr = rewriter.getI64IntegerAttr(nearestId);

    auto hipOp = mlir::hip::ResizeOp::create(rewriter, loc, outputType, context,
                                             input, init, modeAttr, coordAttr,
                                             nearestAttr);
    rewriter.replaceOp(op, hipOp.getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateResizeConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ResizeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
