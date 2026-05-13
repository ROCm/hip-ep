/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceProd -> hip.reduce_prod
///
/// Mirrors the existing ReduceSum/ReduceMax conversion: lifts an optional
/// `axes` operand or attribute into a required tensor operand, threads
/// `keepdims` and `noop_with_empty_axes` through, and shares lowering with
/// other reduction ops via the unified ReduceLowering template.
struct ReduceProdToHip : public mlir::RewritePattern {
  ReduceProdToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceProd", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
ReduceProdToHip::matchAndRewrite(mlir::Operation *op,
                                 mlir::PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value data = op->getOperand(0);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
  mlir::Value init = createEmptyTensor(rewriter, loc, resultType, data);

  int64_t noopWithEmptyAxes = 0;
  if (auto noopAttr =
          op->getAttrOfType<mlir::IntegerAttr>("noop_with_empty_axes")) {
    noopWithEmptyAxes = noopAttr.getSInt();
  }

  // Axes: opset 18+ supplies it as an operand; older opsets use an attribute.
  // The HIP op always takes a (possibly empty) tensor operand, so synthesize
  // one when the ONNX op uses the attribute form.
  mlir::Value axesOperand;
  if (op->getNumOperands() > 1 &&
      !mlir::isa<mlir::NoneType>(op->getOperand(1).getType())) {
    axesOperand = op->getOperand(1);
  } else {
    llvm::SmallVector<int64_t> axesVec;
    if (auto axesAttr = op->getAttrOfType<mlir::ArrayAttr>("axes")) {
      for (auto a : axesAttr)
        axesVec.push_back(
            mlir::cast<mlir::IntegerAttr>(a).getValue().getSExtValue());
    } else if (noopWithEmptyAxes == 0) {
      auto inputType = mlir::cast<mlir::RankedTensorType>(data.getType());
      for (int64_t i : llvm::seq<int64_t>(inputType.getRank()))
        axesVec.push_back(i);
    }
    auto axesType = mlir::RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        mlir::DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand =
        mlir::arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<mlir::IntegerAttr>("keepdims"))
    keepdims = keepdimsAttr.getSInt();

  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceProdOp::create(rewriter, loc, resultType,
                                               context, data, axesOperand,
                                               init, keepdimsAttr, noopAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void mlir::hip::populateReduceProdConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<ReduceProdToHip>(ctx);
}

} // namespace hip
} // namespace mlir
