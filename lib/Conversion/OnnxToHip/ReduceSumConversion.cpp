//===- ReduceSumConversion.cpp - ONNX-to-HIP ReduceSum conversion - *- C++
//-*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/IR/Arith.h"

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.ReduceSum -> hip.reduce_sum
struct ReduceSumToHip : public RewritePattern {
  ReduceSumToHip(MLIRContext* ctx)
      : RewritePattern("onnx.ReduceSum", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override;
};

LogicalResult ReduceSumToHip::matchAndRewrite(Operation* op,
                                              PatternRewriter& rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value data = op->getOperand(0);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, data);

  // Extract noop_with_empty_axes attribute (defaults to 0 in ONNX)
  int64_t noopWithEmptyAxes = 0;
  if (auto noopAttr = op->getAttrOfType<IntegerAttr>("noop_with_empty_axes")) {
    noopWithEmptyAxes = noopAttr.getSInt();
  }

  // Handle axes: can be operand (opset 13+) or attribute (opset < 13)
  // axes is always required in HIP dialect; create empty tensor<0xi64> when not
  // provided
  Value axesOperand;

  if (op->getNumOperands() > 1) {
    // Axes provided as operand (opset 13+)
    axesOperand = op->getOperand(1);
  } else {
    // Axes provided as attribute - convert to constant tensor.
    llvm::SmallVector<int64_t> axesVec = getInt64ArrayAttrOrDefault(op, "axes");
    if (axesVec.empty() && noopWithEmptyAxes == 0) {
      // Default: reduce all axes (when noop_with_empty_axes is 0).
      auto inputType = cast<RankedTensorType>(data.getType());
      for (int64_t i : llvm::seq<int64_t>(inputType.getRank()))
        axesVec.push_back(i);
    }
    // noop_with_empty_axes is 1 and no axes provided -> axesVec stays empty
    // (will create empty tensor<0xi64>).

    // Create constant tensor for axes
    auto axesType = RankedTensorType::get(
        {static_cast<int64_t>(axesVec.size())}, rewriter.getI64Type());
    auto axesAttr =
        DenseIntElementsAttr::get(axesType, llvm::ArrayRef(axesVec));
    axesOperand = arith::ConstantOp::create(rewriter, loc, axesType, axesAttr);
  }

  // Extract keepdims attribute (defaults to 1 in ONNX)
  int64_t keepdims = 1;
  if (auto keepdimsAttr = op->getAttrOfType<IntegerAttr>("keepdims")) {
    keepdims = keepdimsAttr.getSInt();
  }

  // Create hip.reduce_sum operation (axes always provided, may be empty)
  auto keepdimsAttr = rewriter.getI64IntegerAttr(keepdims);
  auto noopWithEmptyAxesAttr = rewriter.getI64IntegerAttr(noopWithEmptyAxes);
  auto hipOp = mlir::hip::ReduceSumOp::create(
      rewriter, loc, resultType, context, data, axesOperand, init, keepdimsAttr,
      noopWithEmptyAxesAttr);

  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateReduceSumConversionPatterns(RewritePatternSet& patterns,
                                                    MLIRContext* ctx) {
  patterns.add<ReduceSumToHip>(ctx);
}

} // namespace hip
} // namespace mlir
