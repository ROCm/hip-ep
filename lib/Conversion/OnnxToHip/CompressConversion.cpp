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

// Number of true entries in the 1-D `condition`, as a host `index`.
//
// Compress keeps one slice per true entry, so the extent along the selection
// axis is data-dependent. It must be known BEFORE the destination is built:
// the destination extent is what sizes `hip.alloc_output` for a graph output,
// and ORT rejects an output request whose shape differs from the shape it
// computed for that run. Sizing the destination from the condition LENGTH
// instead (the worst case) reports the padded extent -- e.g. a vision encoder
// that pads its patch grid to a fixed capacity and drops the pad rows with
// Compress would claim all capacity rows while the caller expects only the
// un-padded ones.
//
// `hip.nonzero` already computes exactly this count into a device i32 scalar
// (its index result is unused here), and `hip.readback_dim` turns that scalar
// into a host `index` with a synchronized D2H copy.
static mlir::FailureOr<mlir::Value>
buildSelectedCount(mlir::PatternRewriter &rewriter, mlir::Operation *op,
                   mlir::Location loc, mlir::Value context,
                   mlir::Value condition,
                   mlir::RankedTensorType conditionType) {
  int64_t condDataType = getHipdnnInputDataType(conditionType.getElementType());
  if (condDataType < 0)
    return rewriter.notifyMatchFailure(
        op, "unsupported condition element type for the selected-count scan");

  mlir::Value condLen =
      mlir::tensor::DimOp::create(rewriter, loc, condition, 0);
  auto indicesType = mlir::RankedTensorType::get(
      {1, mlir::ShapedType::kDynamic}, rewriter.getI64Type());
  mlir::Value indicesInit = mlir::tensor::EmptyOp::create(
      rewriter, loc, indicesType.getShape(), indicesType.getElementType(),
      mlir::ValueRange{condLen});
  auto countType =
      mlir::RankedTensorType::get(/*shape=*/{}, rewriter.getI32Type());
  mlir::Value countInit = mlir::tensor::EmptyOp::create(
      rewriter, loc, countType.getShape(), countType.getElementType());

  auto scan = mlir::hip::NonZeroOp::create(
      rewriter, loc, mlir::TypeRange{indicesType, countType}, context,
      condition, indicesInit, countInit,
      rewriter.getI64IntegerAttr(condDataType));

  return mlir::hip::ReadbackDimOp::create(
             rewriter, loc, rewriter.getIndexType(), context, scan.getResult(1))
      .getResult();
}

/// onnx.Compress -> hip.compress
struct CompressToHip : public mlir::RewritePattern {
  CompressToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Compress", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
CompressToHip::matchAndRewrite(mlir::Operation *op,
                               mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value input = op->getOperand(0);
  mlir::Value condition = op->getOperand(1);

  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  auto inputType = mlir::cast<mlir::RankedTensorType>(input.getType());
  auto conditionType =
      mlir::dyn_cast<mlir::RankedTensorType>(condition.getType());
  if (!conditionType || conditionType.getRank() != 1)
    return rewriter.notifyMatchFailure(op, "condition must be a 1-D tensor");

  bool flatten = !op->hasAttr("axis");
  int64_t axis = 0;
  if (!flatten) {
    axis = op->getAttrOfType<mlir::IntegerAttr>("axis").getSInt();
    if (axis < 0)
      axis += inputType.getRank();
  }
  // Result dimension the selection shrinks: the flattened result is rank-1.
  int64_t selectDim = flatten ? 0 : axis;

  mlir::Value selectedCount;
  if (resultType.isDynamicDim(selectDim)) {
    auto countOrFailure = buildSelectedCount(rewriter, op, loc, context,
                                             condition, conditionType);
    if (mlir::failed(countOrFailure))
      return mlir::failure();
    selectedCount = *countOrFailure;
  }

  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t i : llvm::seq<int64_t>(0, resultType.getRank())) {
    if (!resultType.isDynamicDim(i))
      continue;
    if (i == selectDim)
      dynSizes.push_back(selectedCount);
    else
      dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, input, i));
  }

  mlir::Value init =
      mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                    resultType.getElementType(), dynSizes);

  auto compressOp = mlir::hip::CompressOp::create(
      rewriter, loc, context, input, condition, init,
      rewriter.getI64IntegerAttr(axis), rewriter.getBoolAttr(flatten));

  rewriter.replaceOp(op, compressOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateCompressConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<CompressToHip>(ctx);
}

} // namespace hip
} // namespace mlir
