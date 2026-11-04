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
// onnx.NonMaxSuppression -> hip.non_max_suppression
//===----------------------------------------------------------------------===//
//
// Operands: boxes (B, N, 4) f32, scores (B, C, N) f32, and the optional
// scalars max_output_boxes_per_class (i64), iou_threshold (f32) and
// score_threshold (f32). The scalars must be compile-time constants; they
// become attributes. Every extent of boxes / scores may be dynamic, and
// unranked boxes / scores are cast to rank 3.
//
// The number of selected triples is data-dependent. As for onnx.NonZero, the
// output is allocated at the worst-case capacity B * C * min(N, max_out), the
// kernel writes the true count to a device i32, and `hip.readback_dim` plus
// `tensor.extract_slice` trim the result to (count, 3).
//
// Before:
//   %sel = "onnx.NonMaxSuppression"(%boxes, %scores, %max, %iou)
//            {center_point_box = 0 : si64}
//            : (tensor<1x?x4xf32>, tensor<1x1x?xf32>, tensor<1xi64>,
//               tensor<1xf32>) -> tensor<?x3xi64>
//
// After:
//   %n = tensor.dim %boxes, %c1 : tensor<1x?x4xf32>
//   %m = arith.minsi %n, %c2000 : index
//   %init = tensor.empty(%m) : tensor<?x3xi64>
//   %cnt0 = tensor.empty() : tensor<i32>
//   %r:2 = hip.non_max_suppression(%ctx) ins(%boxes, %scores : ...)
//            outs(%init, %cnt0 : tensor<?x3xi64>, tensor<i32>)
//            {center_point_box = 0, has_score_threshold = 0,
//             iou_threshold = 0.7 : f32, max_output_boxes_per_class = 2000,
//             score_threshold = 0.0 : f32}
//   %count = hip.readback_dim(%ctx, %r#1 : tensor<i32>) -> index
//   %sel = tensor.extract_slice %r#0[0, 0] [%count, 3] [1, 1]

// none, or the 0-element tensor exporters use for an omitted optional.
bool isAbsentOperand(mlir::Operation *op, unsigned idx) {
  if (idx >= op->getNumOperands())
    return true;
  mlir::Value v = op->getOperand(idx);
  if (!v || mlir::isa<mlir::NoneType>(v.getType()))
    return true;
  auto shaped = mlir::dyn_cast<mlir::ShapedType>(v.getType());
  return shaped && shaped.hasStaticShape() && shaped.getNumElements() == 0;
}

/// Single-element dense constant behind `v`: an arith constant or the
/// hip.constant / onnx.Constant inline `value`.
mlir::DenseElementsAttr getScalarConstant(mlir::Value v) {
  while (auto cast = v.getDefiningOp<mlir::tensor::CastOp>())
    v = cast.getSource();
  mlir::Operation *def = v.getDefiningOp();
  if (!def)
    return nullptr;
  mlir::DenseElementsAttr dense;
  if (auto cst = mlir::dyn_cast<mlir::arith::ConstantOp>(def))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(cst.getValue());
  else if (auto attr = def->getAttr("value"))
    dense = mlir::dyn_cast<mlir::DenseElementsAttr>(attr);
  if (!dense || dense.getNumElements() != 1)
    return nullptr;
  return dense;
}

std::optional<int64_t> getConstantInt(mlir::Value v) {
  mlir::DenseElementsAttr dense = getScalarConstant(v);
  if (!dense || !mlir::isa<mlir::IntegerType>(dense.getElementType()))
    return std::nullopt;
  return (*dense.getValues<llvm::APInt>().begin()).getSExtValue();
}

std::optional<double> getConstantFloat(mlir::Value v) {
  mlir::DenseElementsAttr dense = getScalarConstant(v);
  if (!dense || !mlir::isa<mlir::FloatType>(dense.getElementType()))
    return std::nullopt;
  return (*dense.getValues<llvm::APFloat>().begin()).convertToDouble();
}

mlir::FailureOr<mlir::Value> castToRank3(mlir::PatternRewriter &rewriter,
                                         mlir::Location loc,
                                         mlir::Value value) {
  mlir::Type type = value.getType();
  if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(type)) {
    if (ranked.getRank() != 3)
      return mlir::failure();
    return value;
  }
  auto unranked = mlir::dyn_cast<mlir::UnrankedTensorType>(type);
  if (!unranked)
    return mlir::failure();
  auto rankedType = mlir::RankedTensorType::get({mlir::ShapedType::kDynamic,
                                                 mlir::ShapedType::kDynamic,
                                                 mlir::ShapedType::kDynamic},
                                                unranked.getElementType());
  return mlir::tensor::CastOp::create(rewriter, loc, rankedType, value)
      .getResult();
}

struct NonMaxSuppressionToHip : public mlir::RewritePattern {
  NonMaxSuppressionToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.NonMaxSuppression", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumOperands() > 5 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "NonMaxSuppression expects 2-5 operands and 1 result");

    mlir::Type resultType = op->getResult(0).getType();
    if (auto ranked = mlir::dyn_cast<mlir::RankedTensorType>(resultType)) {
      if (ranked.getRank() != 2 ||
          (!ranked.isDynamicDim(1) && ranked.getDimSize(1) != 3))
        return rewriter.notifyMatchFailure(op, "expected (N, 3) result");
    }
    auto resultTensor = mlir::dyn_cast<mlir::TensorType>(resultType);
    if (!resultTensor || !resultTensor.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "result must be an i64 tensor");

    for (unsigned i = 0; i < 2; ++i) {
      auto t = mlir::dyn_cast<mlir::TensorType>(op->getOperand(i).getType());
      if (!t || !t.getElementType().isF32())
        return rewriter.notifyMatchFailure(op, "boxes and scores must be f32");
    }

    int64_t maxOut = 0;
    if (!isAbsentOperand(op, 2)) {
      std::optional<int64_t> v = getConstantInt(op->getOperand(2));
      if (!v)
        return rewriter.notifyMatchFailure(
            op, "max_output_boxes_per_class must be a constant");
      maxOut = std::max<int64_t>(*v, 0);
    }
    float iouThreshold = 0.0f;
    if (!isAbsentOperand(op, 3)) {
      std::optional<double> v = getConstantFloat(op->getOperand(3));
      if (!v)
        return rewriter.notifyMatchFailure(op,
                                           "iou_threshold must be a constant");
      iouThreshold = static_cast<float>(*v);
    }
    float scoreThreshold = 0.0f;
    int64_t hasScoreThreshold = 0;
    if (!isAbsentOperand(op, 4)) {
      std::optional<double> v = getConstantFloat(op->getOperand(4));
      if (!v)
        return rewriter.notifyMatchFailure(
            op, "score_threshold must be a constant");
      scoreThreshold = static_cast<float>(*v);
      hasScoreThreshold = 1;
    }

    int64_t centerPointBox = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("center_point_box"))
      centerPointBox = attr.getValue().getSExtValue();
    if (centerPointBox != 0 && centerPointBox != 1)
      return rewriter.notifyMatchFailure(op, "center_point_box must be 0 or 1");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context argument");
    mlir::Value context = *ctxOrFailure;
    mlir::Location loc = op->getLoc();

    auto boxes = castToRank3(rewriter, loc, op->getOperand(0));
    auto scores = castToRank3(rewriter, loc, op->getOperand(1));
    if (mlir::failed(boxes) || mlir::failed(scores))
      return rewriter.notifyMatchFailure(op,
                                         "expected rank-3 boxes and scores");
    auto boxesType = mlir::cast<mlir::RankedTensorType>(boxes->getType());
    auto scoresType = mlir::cast<mlir::RankedTensorType>(scores->getType());
    if (!boxesType.isDynamicDim(2) && boxesType.getDimSize(2) != 4)
      return rewriter.notifyMatchFailure(op, "boxes last dim must be 4");

    // Extent of (tensor, dim), preferring a static size from `alt`.
    auto extent = [&](mlir::Value t, mlir::RankedTensorType ty, int64_t dim,
                      mlir::RankedTensorType altTy,
                      int64_t altDim) -> mlir::Value {
      if (!ty.isDynamicDim(dim))
        return mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                    ty.getDimSize(dim));
      if (!altTy.isDynamicDim(altDim))
        return mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                    altTy.getDimSize(altDim));
      return mlir::tensor::DimOp::create(rewriter, loc, t, dim);
    };

    // Worst-case triple count: B * C * min(N, max_out).
    mlir::Value capacity;
    int64_t b = boxesType.getDimSize(0), c = scoresType.getDimSize(1),
            n = boxesType.getDimSize(1);
    if (mlir::ShapedType::isDynamic(b))
      b = scoresType.getDimSize(0);
    if (mlir::ShapedType::isDynamic(n))
      n = scoresType.getDimSize(2);
    if (!mlir::ShapedType::isDynamic(b) && !mlir::ShapedType::isDynamic(c) &&
        !mlir::ShapedType::isDynamic(n)) {
      capacity = mlir::arith::ConstantIndexOp::create(
          rewriter, loc, b * c * std::min(n, maxOut));
    } else {
      mlir::Value bv = extent(*boxes, boxesType, 0, scoresType, 0);
      mlir::Value cv = extent(*scores, scoresType, 1, scoresType, 1);
      mlir::Value nv = extent(*boxes, boxesType, 1, scoresType, 2);
      mlir::Value maxv =
          mlir::arith::ConstantIndexOp::create(rewriter, loc, maxOut);
      mlir::Value perClass =
          mlir::arith::MinSIOp::create(rewriter, loc, nv, maxv);
      capacity = mlir::arith::MulIOp::create(
          rewriter, loc, mlir::arith::MulIOp::create(rewriter, loc, bv, cv),
          perClass);
    }

    auto i64Ty = rewriter.getI64Type();
    auto selectedType =
        mlir::RankedTensorType::get({mlir::ShapedType::kDynamic, 3}, i64Ty);
    mlir::Value selectedInit =
        mlir::tensor::EmptyOp::create(rewriter, loc, selectedType.getShape(),
                                      i64Ty, mlir::ValueRange{capacity});
    auto countType = mlir::RankedTensorType::get({}, rewriter.getI32Type());
    mlir::Value countInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, countType.getShape(), countType.getElementType());

    auto hipOp = mlir::hip::NonMaxSuppressionOp::create(
        rewriter, loc, mlir::TypeRange{selectedType, countType}, context,
        *boxes, *scores, selectedInit, countInit,
        rewriter.getI64IntegerAttr(maxOut),
        rewriter.getF32FloatAttr(iouThreshold),
        rewriter.getF32FloatAttr(scoreThreshold),
        rewriter.getI64IntegerAttr(hasScoreThreshold),
        rewriter.getI64IntegerAttr(centerPointBox));

    mlir::Value count =
        mlir::hip::ReadbackDimOp::create(rewriter, loc, rewriter.getIndexType(),
                                         context, hipOp.getResult(1))
            .getResult();
    mlir::SmallVector<mlir::OpFoldResult, 2> offsets = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    mlir::SmallVector<mlir::OpFoldResult, 2> sizes = {mlir::OpFoldResult(count),
                                                      rewriter.getIndexAttr(3)};
    mlir::SmallVector<mlir::OpFoldResult, 2> strides = {
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    mlir::Value trimmed = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, selectedType, hipOp.getResult(0), offsets, sizes,
        strides);

    if (trimmed.getType() != resultType)
      trimmed =
          mlir::tensor::CastOp::create(rewriter, loc, resultType, trimmed);
    rewriter.replaceOp(op, trimmed);
    return mlir::success();
  }
};

} // namespace

void populateNonMaxSuppressionConversionPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {
  patterns.add<NonMaxSuppressionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
