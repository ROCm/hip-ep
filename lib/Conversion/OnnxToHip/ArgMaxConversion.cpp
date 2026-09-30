/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// Build the DPS init for ArgMax. Output dims are the input dims with `axis`
/// dropped (`keepdims = 0`) or kept as size 1 (`keepdims = 1`). Dynamic sizes
/// are read from the matching input dim, which is not the same index as the
/// output dim once the reduced axis has been removed.
static mlir::Value createArgMaxEmpty(mlir::PatternRewriter &rewriter,
                                     mlir::Location loc,
                                     mlir::RankedTensorType resultType,
                                     mlir::Value data, int64_t axis,
                                     int64_t keepdims) {
  llvm::SmallVector<mlir::Value> dynSizes;
  for (int64_t outIdx : llvm::seq<int64_t>(resultType.getRank())) {
    if (!resultType.isDynamicDim(outIdx))
      continue;
    if (keepdims != 0 && outIdx == axis) {
      dynSizes.push_back(
          mlir::arith::ConstantIndexOp::create(rewriter, loc, 1));
      continue;
    }
    int64_t inIdx = outIdx;
    if (keepdims == 0 && outIdx >= axis)
      inIdx = outIdx + 1;
    dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, data, inIdx));
  }
  return mlir::tensor::EmptyOp::create(rewriter, loc, resultType.getShape(),
                                       resultType.getElementType(), dynSizes);
}

static mlir::FailureOr<mlir::RankedTensorType>
inferArgMaxResultType(mlir::Operation *op, mlir::RankedTensorType inputType,
                      int64_t axis, int64_t keepdims) {
  if (auto ranked =
          mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType()))
    return ranked;

  llvm::SmallVector<int64_t> outShape;
  for (int64_t i = 0; i < inputType.getRank(); ++i) {
    if (i == axis) {
      if (keepdims != 0)
        outShape.push_back(1);
    } else {
      outShape.push_back(inputType.getDimSize(i));
    }
  }
  return mlir::RankedTensorType::get(
      outShape, mlir::IntegerType::get(op->getContext(), 64));
}

/// onnx.ArgMax -> hip.arg_max
struct ArgMaxToHip : public mlir::RewritePattern {
  ArgMaxToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ArgMax", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ArgMaxToHip::matchAndRewrite(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return rewriter.notifyMatchFailure(op, "missing context argument");
  mlir::Value context = *ctxOrFailure;

  if (op->getNumOperands() != 1 || op->getNumResults() != 1)
    return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);
  auto inputType = mlir::dyn_cast<mlir::RankedTensorType>(data.getType());
  if (!inputType)
    return rewriter.notifyMatchFailure(op, "ArgMax input must be ranked");

  mlir::Type elemType = inputType.getElementType();
  if (elemType.isUnsignedInteger(32) || elemType.isUnsignedInteger(64))
    return rewriter.notifyMatchFailure(op,
                                       "ArgMax does not support ui32 or ui64");

  int64_t axis = 0;
  if (auto axisAttr = op->getAttrOfType<mlir::IntegerAttr>("axis"))
    axis = axisAttr.getSInt();
  if (axis < 0)
    axis += inputType.getRank();
  if (axis < 0 || axis >= inputType.getRank())
    return rewriter.notifyMatchFailure(op, "ArgMax axis out of range");

  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = keepdimsAttr.getSInt();

  int64_t selectLastIndex = 0;
  if (auto selectAttr =
          op->getAttrOfType<mlir::IntegerAttr>("select_last_index"))
    selectLastIndex = selectAttr.getSInt();

  auto resultTypeOr = inferArgMaxResultType(op, inputType, axis, keepdims);
  if (mlir::failed(resultTypeOr))
    return rewriter.notifyMatchFailure(op, "ArgMax result type is unranked");
  mlir::RankedTensorType resultType = *resultTypeOr;
  if (!mlir::isa<mlir::IntegerType>(resultType.getElementType()))
    return rewriter.notifyMatchFailure(op, "ArgMax result must be an integer");

  int64_t expectedRank = inputType.getRank() - (keepdims == 0 ? 1 : 0);
  if (resultType.getRank() != expectedRank)
    return rewriter.notifyMatchFailure(
        op, "ArgMax result rank does not match keepdims");

  mlir::Value init =
      createArgMaxEmpty(rewriter, loc, resultType, data, axis, keepdims);

  auto hipOp = mlir::hip::ArgMaxOp::create(
      rewriter, loc, context, data, init, rewriter.getI64IntegerAttr(axis),
      rewriter.getI64IntegerAttr(keepdims),
      rewriter.getI64IntegerAttr(selectLastIndex));

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateArgMaxConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<ArgMaxToHip>(ctx);
}

} // namespace hip
} // namespace mlir
