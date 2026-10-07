/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LRNConversion.cpp - onnx.LRN -> hip.lrn ---------------------------===//
//
// Before:
//   %y = "onnx.LRN"(%x) {alpha = 9.99999974E-5 : f32, beta = 0.75 : f32,
//                        bias = 1.0 : f32, size = 5 : si64}
//        : (tensor<1x64x56x56xf32>) -> tensor<1x64x56x56xf32>
// After:
//   %init = tensor.empty() : tensor<1x64x56x56xf32>
//   %y = hip.lrn(%ctx) ins(%x : tensor<1x64x56x56xf32>)
//                      outs(%init : tensor<1x64x56x56xf32>)
//                      {size = 5 : i64}
//
// The window is across channels (axis 1). Rank below 2 stays onnx.LRN.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static bool isLrnFloat(Type type) {
  return type.isF16() || type.isBF16() || type.isF32() || type.isF64();
}

struct LRNToHip : public RewritePattern {
  explicit LRNToHip(MLIRContext *ctx)
      : RewritePattern("onnx.LRN", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "onnx.LRN expects 1 operand and 1 result");

    Value input = op->getOperand(0);
    auto inputType = dyn_cast<RankedTensorType>(input.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!inputType || !resultType || inputType.getRank() < 2 ||
        inputType.getRank() != resultType.getRank())
      return rewriter.notifyMatchFailure(
          op, "onnx.LRN lowering expects matching rank of at least 2");
    if (!isLrnFloat(inputType.getElementType()) ||
        inputType.getElementType() != resultType.getElementType())
      return rewriter.notifyMatchFailure(
          op, "onnx.LRN input and result must be f16, bf16, f32, or f64");

    for (int64_t dim : llvm::seq<int64_t>(inputType.getRank())) {
      if (inputType.isDynamicDim(dim) || resultType.isDynamicDim(dim))
        continue;
      if (inputType.getDimSize(dim) != resultType.getDimSize(dim))
        return rewriter.notifyMatchFailure(
            op, "onnx.LRN input and result shapes must match");
    }

    auto sizeAttr = op->getAttrOfType<IntegerAttr>("size");
    if (!sizeAttr || sizeAttr.getInt() < 1)
      return rewriter.notifyMatchFailure(op, "onnx.LRN size must be positive");

    double alpha = 0.0001;
    double beta = 0.75;
    double bias = 1.0;
    if (auto attr = op->getAttrOfType<FloatAttr>("alpha"))
      alpha = attr.getValueAsDouble();
    if (auto attr = op->getAttrOfType<FloatAttr>("beta"))
      beta = attr.getValueAsDouble();
    if (auto attr = op->getAttrOfType<FloatAttr>("bias"))
      bias = attr.getValueAsDouble();

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();

    Location loc = op->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, input);
    auto hipOp = LRNOp::create(rewriter, loc, resultType, *ctxOrFailure, input,
                               init, rewriter.getF64FloatAttr(alpha),
                               rewriter.getF64FloatAttr(beta),
                               rewriter.getF64FloatAttr(bias),
                               rewriter.getI64IntegerAttr(sizeAttr.getInt()));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void populateLRNConversionPatterns(RewritePatternSet &patterns,
                                   MLIRContext *ctx) {
  patterns.add<LRNToHip>(ctx);
}

} // namespace hip
} // namespace mlir
