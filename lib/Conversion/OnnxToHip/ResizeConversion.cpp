/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../ResizeLayout.h"
#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"

#include <cmath>
#include <limits>

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// onnx.Resize -> hip.resize
//===----------------------------------------------------------------------===//
//
// ONNX Resize is a multi-dimensional resampler whose full attribute surface
// is huge.  This pass implements the slice that covers ONNX exporters'
// common image / volume use-case:
//
//   * mode in {"nearest", "linear"}                                  (no cubic)
//   * coordinate_transformation_mode in
//       {"half_pixel", "pytorch_half_pixel", "asymmetric", "align_corners"}
//   * nearest_mode in {round_prefer_floor, round_prefer_ceil, floor, ceil}
//   * antialias = 0, exclude_outside = 0             (defaults)
//   * keep_aspect_ratio_policy = "stretch"           (default)
//   * roi must be absent: none, or a 0-element tensor (no tf_crop_and_resize)
//   * rank in [3, 5], input and output ranks equal
//
// The runtime kernel is unchanged: a copied prefix of at most two axes and a
// trailing window of at most three.  Channels-first `1x3x16x16 -> 1x3x32x32`
// is prefix (N, C) and window (H, W).  Channels-last `1x16x16x3 -> 1x32x32x3`
// is prefix (N), an empty channel slot, and window (H, W, C).  The channel
// extent is unchanged, so that window axis copies through.
//
// Static window extents come from the result type.  When a window extent is
// dynamic, `scales` must be a compile-time constant (a hip.constant /
// arith.constant, possibly behind a cast).  A copied prefix axis must have
// scale 1 and copies `tensor.dim` when that dim is dynamic.  Any other
// positive scale is floor(input_dim * scale) in f64, then index.
//
// When `scales` is absent, operand `sizes` supplies the output extents.  A
// compile-time constant vector is written into the result type.  A runtime
// vector is not a shape record: each element is read with
// `hip.readback_scalar` (device-to-host copy and stream sync) and that host
// index sizes `tensor.empty`.  A static input extent paired with one of those
// dynamic outputs is resized only because the index exists; the launch is
// saved on the op so lowering does not reject the memref types.
//
// Compile-time work:
//   * decode the three string attributes into i64 enums baked onto the
//     hip.resize op
//   * reject a shape the runtime kernel cannot express: more than two copied
//     prefix axes, or a trailing window longer than three
//
// Before (dynamic H/W, constant scales [1, 1, 2, 2]):
//   %y = "onnx.Resize"(%x, %roi, %scales)
//          : (tensor<?x3x?x?xf16>, none, tensor<4xf32>) -> tensor<?x3x?x?xf16>
//
// After:
//   %n = tensor.dim %x, %c0
//   %h = arith.index_cast (arith.fptosi (arith.mulf (arith.sitofp
//          (arith.index_cast (tensor.dim %x, %c2))) * 2.0))
//   %w = ... dim %c3 ...
//   %init = tensor.empty(%n, %h, %w) : tensor<?x3x?x?xf16>
//   %y = hip.resize(%ctx) ins(%x) outs(%init) {mode = ...}

// none, or the 0-element tensor exporters use for an omitted optional.
// ONNX only reads roi when coordinate_transformation_mode is
// tf_crop_and_resize, which this conversion rejects, so an empty tensor
// carries no crop.
static bool isAbsent(mlir::Value v) {
  if (!v || mlir::isa<mlir::NoneType>(v.getType()))
    return true;
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(v.getType());
  return shaped && shaped.hasStaticShape() && shaped.getNumElements() == 0;
}

/// Dense elements backing a compile-time constant. Recognizes arith constants
/// and the hip.constant carrier `lowerOnnxConstants` leaves behind.
static mlir::DenseElementsAttr getCompileTimeConstantTensor(mlir::Value value) {
  mlir::Operation *defOp = value.getDefiningOp();
  if (!defOp)
    return nullptr;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(defOp))
    return mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  if (auto attr = defOp->getAttr("value"))
    if (auto dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr))
      return dense;
  return nullptr;
}

static mlir::Value unwrapCast(mlir::Value v) {
  for (int i = 0; i < 4; ++i) {
    mlir::Operation *def = v.getDefiningOp();
    if (!def)
      break;
    llvm::StringRef name = def->getName().getStringRef();
    if (name == "onnx.Cast")
      v = def->getOperand(0);
    else if (mlir::isa<mlir::hip::CastOp>(def))
      v = def->getOperand(1);
    else
      break;
  }
  return v;
}

static bool foldScales(mlir::Value scales, int64_t rank,
                       llvm::SmallVectorImpl<double> &out) {
  if (!scales || isAbsent(scales))
    return false;
  mlir::DenseElementsAttr dense =
      getCompileTimeConstantTensor(unwrapCast(scales));
  if (!dense)
    return false;
  auto tensorType = mlir::dyn_cast<mlir::RankedTensorType>(dense.getType());
  if (!tensorType || tensorType.getRank() != 1 ||
      tensorType.getNumElements() != static_cast<int64_t>(rank) ||
      !mlir::isa<mlir::FloatType>(tensorType.getElementType()))
    return false;
  out.clear();
  for (mlir::APFloat entry : dense.getValues<mlir::APFloat>())
    out.push_back(entry.convertToDouble());
  return out.size() == static_cast<size_t>(rank);
}

/// floor(input_dim * scale) as an index. Static input dims fold here. A
/// positive product truncates toward zero in f64, which is floor.
static mlir::Value floorScaledDim(mlir::OpBuilder &b, mlir::Location loc,
                                  mlir::Value input,
                                  mlir::RankedTensorType inputType,
                                  int64_t axis, double scale) {
  if (!(scale > 0.0) || !std::isfinite(scale))
    return {};
  if (!inputType.isDynamicDim(axis)) {
    double prod = static_cast<double>(inputType.getDimSize(axis)) * scale;
    if (!(prod > 0.0) ||
        prod > static_cast<double>(std::numeric_limits<int64_t>::max()))
      return {};
    return mlir::arith::ConstantIndexOp::create(
        b, loc, static_cast<int64_t>(std::floor(prod)));
  }
  mlir::Value dim = mlir::tensor::DimOp::create(b, loc, input, axis);
  mlir::Value dim64 =
      mlir::arith::IndexCastOp::create(b, loc, b.getI64Type(), dim);
  mlir::Value dimF =
      mlir::arith::SIToFPOp::create(b, loc, b.getF64Type(), dim64);
  mlir::Value scaleV = mlir::arith::ConstantOp::create(
      b, loc, b.getF64Type(), b.getF64FloatAttr(scale));
  mlir::Value prod = mlir::arith::MulFOp::create(b, loc, dimF, scaleV);
  mlir::Value floored =
      mlir::arith::FPToSIOp::create(b, loc, b.getI64Type(), prod);
  return mlir::arith::IndexCastOp::create(b, loc, b.getIndexType(), floored);
}

/// Constant extents, or `kDynamic` when the element is only available by
/// reading the sizes tensor. Does not build IR: a later plan failure must
/// leave the onnx op untouched.
static std::optional<llvm::SmallVector<int64_t>>
classifyHostSizes(mlir::Value sizes, int64_t rank) {
  if (!sizes || isAbsent(sizes))
    return std::nullopt;
  sizes = unwrapCast(sizes);
  auto sizesType = mlir::dyn_cast<mlir::RankedTensorType>(sizes.getType());
  if (!sizesType || sizesType.getRank() != 1 || !sizesType.hasStaticShape() ||
      sizesType.getNumElements() != rank ||
      !sizesType.getElementType().isInteger())
    return std::nullopt;
  llvm::SmallVector<int64_t> extents(rank, mlir::ShapedType::kDynamic);
  if (mlir::DenseElementsAttr dense = getCompileTimeConstantTensor(sizes)) {
    if (!dense.getElementType().isInteger() ||
        static_cast<int64_t>(dense.getNumElements()) != rank)
      return std::nullopt;
    int64_t axis = 0;
    for (mlir::APInt entry : dense.getValues<mlir::APInt>()) {
      if (!entry.isStrictlyPositive() || entry.getSignificantBits() > 63)
        return std::nullopt;
      extents[axis++] = entry.getSExtValue();
    }
  }
  return extents;
}

/// One host `index` per axis of a rank-1 integer `sizes` tensor. A dense
/// constant stays a host constant. Anything else is a synchronized read of
/// that element: after Concat lowering the payload is a buffer, and
/// `tensor.dim` of it is the length, not the extent.
static std::optional<llvm::SmallVector<mlir::Value>>
readHostSizeIndexes(mlir::OpBuilder &b, mlir::Location loc,
                    mlir::Value context, mlir::Value sizes, int64_t rank) {
  if (!sizes || isAbsent(sizes))
    return std::nullopt;
  sizes = unwrapCast(sizes);
  auto sizesType = mlir::dyn_cast<mlir::RankedTensorType>(sizes.getType());
  if (!sizesType || sizesType.getRank() != 1 || !sizesType.hasStaticShape() ||
      sizesType.getNumElements() != rank ||
      !sizesType.getElementType().isInteger())
    return std::nullopt;

  llvm::SmallVector<mlir::Value> indexes;
  if (mlir::DenseElementsAttr dense = getCompileTimeConstantTensor(sizes)) {
    if (!dense.getElementType().isInteger() ||
        static_cast<int64_t>(dense.getNumElements()) != rank)
      return std::nullopt;
    for (mlir::APInt entry : dense.getValues<mlir::APInt>()) {
      if (!entry.isStrictlyPositive() || entry.getSignificantBits() > 63)
        return std::nullopt;
      indexes.push_back(mlir::arith::ConstantIndexOp::create(
          b, loc, entry.getSExtValue()));
    }
    return indexes;
  }

  auto sliceType =
      mlir::RankedTensorType::get({1}, sizesType.getElementType());
  mlir::SmallVector<mlir::OpFoldResult, 1> one{b.getIndexAttr(1)};
  mlir::SmallVector<mlir::OpFoldResult, 1> strides{b.getIndexAttr(1)};
  for (int64_t axis : llvm::seq<int64_t>(rank)) {
    mlir::SmallVector<mlir::OpFoldResult, 1> offsets{b.getIndexAttr(axis)};
    mlir::Value slice = mlir::tensor::ExtractSliceOp::create(
        b, loc, sliceType, sizes, offsets, one, strides);
    mlir::Value raw = mlir::hip::ReadbackScalarOp::create(
        b, loc, sizesType.getElementType(), context, slice);
    indexes.push_back(mlir::arith::IndexCastOp::create(
        b, loc, b.getIndexType(), raw));
  }
  return indexes;
}

static mlir::Value copyAxisExtent(mlir::OpBuilder &b, mlir::Location loc,
                                  mlir::Value input,
                                  mlir::RankedTensorType inputType,
                                  int64_t axis) {
  if (inputType.isDynamicDim(axis))
    return mlir::tensor::DimOp::create(b, loc, input, axis);
  return mlir::arith::ConstantIndexOp::create(b, loc,
                                              inputType.getDimSize(axis));
}

struct ResizeToHip : public mlir::RewritePattern {
  ResizeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Resize", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected single output");

    // ONNX Resize accepts (X, roi?, scales?, sizes?) — between 1 and 4
    // operands depending on what the exporter supplied.  We accept any
    // count but require `roi` (operand 1) to be absent.
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
    if (rank < 3 || rank > 5 || rank != outputType.getRank())
      return rewriter.notifyMatchFailure(
          op, "Resize requires rank in [3, 5] and matching in/out ranks");
    if (!mlir::isa<mlir::FloatType>(inputType.getElementType()) ||
        inputType.getElementType() != outputType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "Resize runtime supports only matching float types");

    // Channels-last rank 4 fits the existing kernel: the batch is the prefix,
    // and (H, W, C) is the trailing window.  See planHipResizeLaunch. A
    // sizes vector can still match when this type-only plan rejects a static
    // input extent paired with a dynamic output extent.
    std::optional<HipResizeLaunch> launch =
        planHipResizeLaunch(inputType, outputType);

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
    else if (ct == "pytorch_half_pixel")
      // Same map as half_pixel, except an output axis of length 1 samples
      // input coordinate 0. The kernel applies that guard; the id stays
      // distinct so a length-1 axis is not treated as half_pixel.
      coordId = 3;
    else
      return rewriter.notifyMatchFailure(
          op, "Resize coordinate_transformation_mode must be one of "
              "{half_pixel, pytorch_half_pixel, asymmetric, align_corners}");

    std::string nm = getStrAttr("nearest_mode", "round_prefer_floor");
    int64_t nearestId;
    if (nm == "round_prefer_floor")
      nearestId = 0;
    else if (nm == "round_prefer_ceil")
      nearestId = 1;
    else if (nm == "floor")
      nearestId = 2;
    else if (nm == "ceil")
      nearestId = 3;
    else
      return rewriter.notifyMatchFailure(
          op, "Resize nearest_mode must be one of "
              "{round_prefer_floor, round_prefer_ceil, floor, ceil}");

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

    auto modeAttr = rewriter.getI64IntegerAttr(modeId);
    auto coordAttr = rewriter.getI64IntegerAttr(coordId);
    auto nearestAttr = rewriter.getI64IntegerAttr(nearestId);

    auto emitResize = [&](mlir::RankedTensorType resultType,
                          mlir::ValueRange dynSizes,
                          const std::optional<HipResizeLaunch> &stamped)
        -> mlir::LogicalResult {
      mlir::Value init = mlir::tensor::EmptyOp::create(
          rewriter, loc, resultType.getShape(), resultType.getElementType(),
          dynSizes);
      auto hipOp = mlir::hip::ResizeOp::create(
          rewriter, loc, resultType, context, input, init, modeAttr, coordAttr,
          nearestAttr);
      if (stamped) {
        hipOp->setAttr("prefix_count",
                       rewriter.getI64IntegerAttr(stamped->prefixCount));
        hipOp->setAttr("spatial_rank",
                       rewriter.getI64IntegerAttr(stamped->spatialRank));
      }
      mlir::Value result = hipOp.getResult(0);
      if (result.getType() != outputType)
        result = mlir::tensor::CastOp::create(rewriter, loc, outputType, result);
      rewriter.replaceOp(op, result);
      return mlir::success();
    };

    // A static window already has its extents in the result type, so scales
    // may be a runtime value.  The kernel recovers scale = in_dim / out_dim.
    // A dynamic window extent is filled from a compile-time constant scale.
    // This path is unchanged for every resize the type-only plan accepts.
    bool scalesPath = false;
    llvm::SmallVector<mlir::Value> dynSizes;
    if (launch) {
      bool windowDynamic = false;
      for (int64_t i : llvm::seq<int64_t>(launch->spatialRank))
        if (outputType.isDynamicDim(launch->prefixCount + i))
          windowDynamic = true;
      if (!windowDynamic) {
        for (int64_t i : llvm::seq<int64_t>(launch->prefixCount)) {
          if (outputType.isDynamicDim(i))
            dynSizes.push_back(
                mlir::tensor::DimOp::create(rewriter, loc, input, i));
        }
        scalesPath = true;
      } else {
        mlir::Value scales = operands.size() > 2 ? operands[2] : mlir::Value();
        llvm::SmallVector<double> scaleVec;
        if (foldScales(scales, rank, scaleVec)) {
          for (int64_t axis : llvm::seq<int64_t>(launch->prefixCount)) {
            if (scaleVec[axis] != 1.0)
              return rewriter.notifyMatchFailure(
                  op, "Resize: only spatial-axis resampling supported "
                      "(copied prefix scale must be 1)");
          }
          for (double scale : scaleVec) {
            if (!(scale > 0.0) || !std::isfinite(scale))
              return rewriter.notifyMatchFailure(
                  op, "Resize: non-positive scale");
          }
          for (int64_t axis : llvm::seq<int64_t>(rank)) {
            if (inputType.isDynamicDim(axis) || outputType.isDynamicDim(axis))
              continue;
            double scale = scaleVec[axis];
            int64_t expected = inputType.getDimSize(axis);
            if (scale != 1.0) {
              double prod = static_cast<double>(expected) * scale;
              if (!(prod > 0.0))
                return rewriter.notifyMatchFailure(
                    op, "Resize: non-positive scale");
              expected = static_cast<int64_t>(std::floor(prod));
            }
            if (expected != outputType.getDimSize(axis))
              return rewriter.notifyMatchFailure(
                  op, "Resize: scale disagrees with static output extent");
          }
          for (int64_t axis : llvm::seq<int64_t>(rank)) {
            if (!outputType.isDynamicDim(axis))
              continue;
            double scale = scaleVec[axis];
            mlir::Value extent =
                scale == 1.0
                    ? copyAxisExtent(rewriter, loc, input, inputType, axis)
                    : floorScaledDim(rewriter, loc, input, inputType, axis,
                                     scale);
            if (!extent)
              return rewriter.notifyMatchFailure(
                  op, "Resize: output extent is not host-computable");
            dynSizes.push_back(extent);
          }
          scalesPath = true;
        }
      }
    }
    if (scalesPath)
      return emitResize(outputType, dynSizes, std::nullopt);

    mlir::Value sizes = operands.size() > 3 ? operands[3] : mlir::Value();
    std::optional<llvm::SmallVector<int64_t>> classified =
        classifyHostSizes(sizes, rank);
    if (!classified)
      return rewriter.notifyMatchFailure(
          op, launch ? "Resize: dynamic output spatial dims require a "
                       "compile-time constant scales vector or a sizes vector"
                     : "Resize does not fit a copied prefix of at most 2 axes "
                       "and a trailing window of at most 3");

    llvm::SmallVector<int64_t> refinedShape;
    llvm::SmallVector<int64_t> dynamicAxes;
    for (int64_t axis : llvm::seq<int64_t>(rank)) {
      int64_t extent = (*classified)[axis];
      bool known = extent != mlir::ShapedType::kDynamic;
      if (known && !outputType.isDynamicDim(axis) &&
          outputType.getDimSize(axis) != extent)
        return rewriter.notifyMatchFailure(
            op, "Resize: sizes disagrees with static output extent");
      // A constant extent of a static input axis is written into the type, so
      // the type-only plan still applies and no readback is required. A
      // runtime extent stays dynamic; its host index is the readback.
      if (known && !inputType.isDynamicDim(axis)) {
        refinedShape.push_back(extent);
        continue;
      }
      if (!outputType.isDynamicDim(axis)) {
        refinedShape.push_back(outputType.getDimSize(axis));
        continue;
      }
      refinedShape.push_back(mlir::ShapedType::kDynamic);
      dynamicAxes.push_back(axis);
    }
    auto refinedType = mlir::RankedTensorType::get(
        refinedShape, outputType.getElementType());
    std::optional<HipResizeLaunch> typeLaunch =
        planHipResizeLaunch(inputType, refinedType);
    std::optional<HipResizeLaunch> hostLaunch =
        planHipResizeLaunch(inputType, refinedType, /*hostExtents=*/true);
    if (!typeLaunch && !hostLaunch)
      return rewriter.notifyMatchFailure(
          op, "Resize: sizes vector does not fit a copied prefix of at most "
              "2 axes and a trailing window of at most 3");

    std::optional<llvm::SmallVector<mlir::Value>> sizeIndexes =
        readHostSizeIndexes(rewriter, loc, context, sizes, rank);
    if (!sizeIndexes)
      return rewriter.notifyMatchFailure(op, "Resize: sizes vector is not readable");
    llvm::SmallVector<mlir::Value> sizeDynSizes;
    for (int64_t axis : dynamicAxes)
      sizeDynSizes.push_back((*sizeIndexes)[axis]);
    // Every dynamic dim of refinedType has a host index in sizeDynSizes.
    return emitResize(refinedType, sizeDynSizes,
                      typeLaunch ? std::nullopt : hostLaunch);
  }
};

} // namespace

void populateResizeConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ResizeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
