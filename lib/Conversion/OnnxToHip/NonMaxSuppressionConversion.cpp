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

static bool isAbsent(Operation *op, unsigned index) {
  return index >= op->getNumOperands() ||
         isa<NoneType>(op->getOperand(index).getType());
}

/// ONNX declares the three optional inputs as scalars; exporters also emit
/// one-element 1-D tensors.
static bool isScalarOf(Value value, Type elementType) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type && type.getElementType() == elementType &&
         type.hasStaticShape() && type.getNumElements() == 1;
}

static Value createScalarConstant(PatternRewriter &rewriter, Location loc,
                                  RankedTensorType type, Attribute value) {
  OperationState state(loc, "onnx.Constant");
  state.addTypes(type);
  state.addAttribute("value", DenseElementsAttr::get(type, value));
  return rewriter.create(state)->getResult(0);
}

static Value dimOrConstant(PatternRewriter &rewriter, Location loc, Value value,
                           int64_t dim) {
  auto type = cast<RankedTensorType>(value.getType());
  if (type.isDynamicDim(dim))
    return tensor::DimOp::create(rewriter, loc, value, dim);
  return arith::ConstantIndexOp::create(rewriter, loc, type.getDimSize(dim));
}

// onnx.NonMaxSuppression -> hip.non_max_suppression + readback + slice
//
// Before: %y = onnx.NonMaxSuppression(%boxes, %scores, %max, %iou, %score)
//              : (...) -> tensor<?x3xi64>
// After:  %r:2 = hip.non_max_suppression(%ctx) ins(...)
//                    outs(%empty_BxCxS_by_3, %count)
//         %n = hip.readback_dim(%ctx) %r#1
//         %rows = tensor.empty(%n) : tensor<?x3xi64>
//         %y = tensor.insert_slice
//                (tensor.extract_slice %r#0[0, 0] [%n, 3]) into %rows
struct NonMaxSuppressionToHip : public RewritePattern {
  NonMaxSuppressionToHip(MLIRContext *ctx)
      : RewritePattern("onnx.NonMaxSuppression", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 2 || op->getNumOperands() > 5 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 2-5 inputs, 1 output");

    Value boxes = op->getOperand(0);
    Value scores = op->getOperand(1);
    auto boxesType = dyn_cast<RankedTensorType>(boxes.getType());
    auto scoresType = dyn_cast<RankedTensorType>(scores.getType());
    Type f32 = rewriter.getF32Type();
    Type i64 = rewriter.getI64Type();
    if (!boxesType || !scoresType || boxesType.getRank() != 3 ||
        scoresType.getRank() != 3 || boxesType.getElementType() != f32 ||
        scoresType.getElementType() != f32)
      return rewriter.notifyMatchFailure(
          op, "boxes [B,S,4] and scores [B,C,S] must be ranked f32");
    if (!boxesType.isDynamicDim(2) && boxesType.getDimSize(2) != 4)
      return rewriter.notifyMatchFailure(op, "boxes last dim must be 4");

    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!resultType || resultType.getRank() != 2 ||
        resultType.getElementType() != i64 ||
        (!resultType.isDynamicDim(1) && resultType.getDimSize(1) != 3))
      return rewriter.notifyMatchFailure(op, "result must be [N,3] i64");

    int64_t centerPointBox = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("center_point_box"))
      centerPointBox = attr.getValue().getSExtValue();
    if (centerPointBox != 0 && centerPointBox != 1)
      return rewriter.notifyMatchFailure(op, "center_point_box must be 0 or 1");

    // Reject a supplied optional input before building defaults. A failure
    // after rewriter.create would leave those constants in the graph.
    Value maxOutput = isAbsent(op, 2) ? Value() : op->getOperand(2);
    Value iouThreshold = isAbsent(op, 3) ? Value() : op->getOperand(3);
    Value scoreThreshold = isAbsent(op, 4) ? Value() : op->getOperand(4);
    if ((maxOutput && !isScalarOf(maxOutput, i64)) ||
        (iouThreshold && !isScalarOf(iouThreshold, f32)) ||
        (scoreThreshold && !isScalarOf(scoreThreshold, f32)))
      return rewriter.notifyMatchFailure(
          op, "max_output_boxes_per_class must be i64 and the thresholds f32 "
              "one-element tensors");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    Value context = *ctxOrFailure;

    Location loc = op->getLoc();
    auto i64Scalar = RankedTensorType::get({}, i64);
    auto f32Scalar = RankedTensorType::get({}, f32);

    // Absent max_output_boxes_per_class selects nothing; absent iou_threshold
    // is 0. An absent score_threshold disables score filtering entirely, which
    // no threshold value reproduces, so it is carried as an attribute.
    bool hasScoreThreshold = static_cast<bool>(scoreThreshold);
    if (!maxOutput)
      maxOutput = createScalarConstant(rewriter, loc, i64Scalar,
                                       rewriter.getI64IntegerAttr(0));
    if (!iouThreshold)
      iouThreshold = createScalarConstant(rewriter, loc, f32Scalar,
                                          rewriter.getF32FloatAttr(0.0f));
    if (!scoreThreshold)
      scoreThreshold = createScalarConstant(rewriter, loc, f32Scalar,
                                            rewriter.getF32FloatAttr(0.0f));

    // Every (batch, class, box) triple selected is the worst case.
    Value capacity = arith::MulIOp::create(
        rewriter, loc,
        arith::MulIOp::create(rewriter, loc,
                              dimOrConstant(rewriter, loc, boxes, 0),
                              dimOrConstant(rewriter, loc, scores, 1)),
        dimOrConstant(rewriter, loc, boxes, 1));
    auto capacityType = RankedTensorType::get({ShapedType::kDynamic, 3}, i64);
    Value selectedInit = tensor::EmptyOp::create(
        rewriter, loc, capacityType.getShape(), i64, ValueRange{capacity});
    auto countType = RankedTensorType::get({}, rewriter.getI32Type());
    Value countInit = tensor::EmptyOp::create(
        rewriter, loc, countType.getShape(), countType.getElementType());

    auto hipOp = NonMaxSuppressionOp::create(
        rewriter, loc, TypeRange{capacityType, countType}, context, boxes,
        scores, maxOutput, iouThreshold, scoreThreshold, selectedInit,
        countInit, rewriter.getI64IntegerAttr(centerPointBox),
        rewriter.getBoolAttr(hasScoreThreshold));

    Value count = ReadbackDimOp::create(rewriter, loc, rewriter.getIndexType(),
                                        context, hipOp.getResult(1))
                      .getResult();
    SmallVector<OpFoldResult, 2> offsets = {rewriter.getIndexAttr(0),
                                            rewriter.getIndexAttr(0)};
    SmallVector<OpFoldResult, 2> sizes = {OpFoldResult(count),
                                          rewriter.getIndexAttr(3)};
    SmallVector<OpFoldResult, 2> strides = {rewriter.getIndexAttr(1),
                                            rewriter.getIndexAttr(1)};
    // The capacity buffer stays internal. The returned tensor is a fresh
    // [N, 3] allocation, so a graph output is sized from the selected-row
    // count rather than from B * C * S.
    Value selectedRows = tensor::EmptyOp::create(
        rewriter, loc, capacityType.getShape(), i64, ValueRange{count});
    Value slice = tensor::ExtractSliceOp::create(rewriter, loc, capacityType,
                                                 hipOp.getResult(0), offsets,
                                                 sizes, strides);
    Value trimmed =
        tensor::InsertSliceOp::create(rewriter, loc, slice, selectedRows,
                                      offsets, sizes, strides)
            .getResult();
    if (trimmed.getType() != resultType)
      trimmed = tensor::CastOp::create(rewriter, loc, resultType, trimmed);
    rewriter.replaceOp(op, trimmed);
    return success();
  }
};

} // namespace

void populateNonMaxSuppressionConversionPatterns(RewritePatternSet &patterns,
                                                 MLIRContext *ctx) {
  patterns.add<NonMaxSuppressionToHip>(ctx);
}

} // namespace hip
} // namespace mlir
