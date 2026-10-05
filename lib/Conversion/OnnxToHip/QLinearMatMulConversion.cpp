/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

// onnx.QLinearMatMul -> hip.qlinear_matmul
//
// Before:
//   %y = "onnx.QLinearMatMul"(%a, %as, %az, %b, %bs, %bz, %ys, %yz)
//        : (tensor<1x4xui8>, tensor<f32>, tensor<ui8>,
//           tensor<4x3xi8>, tensor<f32>, tensor<i8>,
//           tensor<f32>, tensor<ui8>) -> tensor<1x3xui8>
// After:
//   %init = tensor.empty() : tensor<1x3xui8>
//   %y = hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
//            ...) outs(%init : tensor<1x3xui8>)
//
// Both operands are rank 2 and every scale and zero point is per-tensor.
// Rank other than 2, and per-row or per-column quantization, stay
// onnx.QLinearMatMul. hip.qmatmul is a different op and is not produced here.
static bool isEightBit(Type type) {
  auto integer = dyn_cast<IntegerType>(type);
  return integer && integer.getWidth() == 8;
}

static LogicalResult checkPerTensor(PatternRewriter &rewriter, Operation *op,
                                    Value scale, Value zp, Type storage,
                                    StringRef name) {
  auto scaleType = dyn_cast<RankedTensorType>(scale.getType());
  auto zpType = dyn_cast<RankedTensorType>(zp.getType());
  auto reject = [&](const char *suffix) {
    return rewriter.notifyMatchFailure(op, (llvm::Twine(name) + suffix).str());
  };
  if (!scaleType || !zpType || scaleType.getRank() > 1 || zpType.getRank() > 1)
    return reject(" scale and zero point must be rank 0 or 1");
  if (!scaleType.getElementType().isF32())
    return reject(" scale must be f32");
  if (zpType.getElementType() != storage)
    return reject(" zero point type must match the quantized tensor");
  if (!scaleType.hasStaticShape() || !zpType.hasStaticShape())
    return success();
  if (scaleType.getNumElements() == 1 && zpType.getNumElements() == 1)
    return success();
  return reject(" quantization must be per-tensor");
}

struct QLinearMatMulToHip : public RewritePattern {
  explicit QLinearMatMulToHip(MLIRContext *ctx)
      : RewritePattern("onnx.QLinearMatMul", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    if (op->getNumResults() != 1 || op->getNumOperands() != 8)
      return rewriter.notifyMatchFailure(
          op, "onnx.QLinearMatMul expects 8 operands and 1 result");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();

    Value a = op->getOperand(0);
    Value aScale = op->getOperand(1);
    Value aZp = op->getOperand(2);
    Value b = op->getOperand(3);
    Value bScale = op->getOperand(4);
    Value bZp = op->getOperand(5);
    Value yScale = op->getOperand(6);
    Value yZp = op->getOperand(7);

    auto aType = dyn_cast<RankedTensorType>(a.getType());
    auto bType = dyn_cast<RankedTensorType>(b.getType());
    auto resultType = dyn_cast<RankedTensorType>(op->getResult(0).getType());
    if (!aType || !bType || !resultType || aType.getRank() != 2 ||
        bType.getRank() != 2 || resultType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "onnx.QLinearMatMul lowering expects rank-2 a, b, and result");
    if (!aType.hasStaticShape() || !bType.hasStaticShape() ||
        !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "onnx.QLinearMatMul lowering expects static shapes");

    if (!isEightBit(aType.getElementType()) ||
        !isEightBit(bType.getElementType()) ||
        !isEightBit(resultType.getElementType()))
      return rewriter.notifyMatchFailure(
          op, "onnx.QLinearMatMul a, b, and result must be 8-bit");

    int64_t m = aType.getDimSize(0);
    int64_t k = aType.getDimSize(1);
    int64_t n = bType.getDimSize(1);
    if (m < 1 || k < 1 || n < 1)
      return rewriter.notifyMatchFailure(op, "M, K, and N must be positive");
    if (bType.getDimSize(0) != k)
      return rewriter.notifyMatchFailure(op, "b rows must equal a columns");
    if (resultType.getDimSize(0) != m || resultType.getDimSize(1) != n)
      return rewriter.notifyMatchFailure(
          op, "result shape must be [M, N] from a [M, K] and b [K, N]");

    if (failed(checkPerTensor(rewriter, op, aScale, aZp, aType.getElementType(),
                              "a")) ||
        failed(checkPerTensor(rewriter, op, bScale, bZp, bType.getElementType(),
                              "b")) ||
        failed(checkPerTensor(rewriter, op, yScale, yZp,
                              resultType.getElementType(), "y")))
      return failure();

    Location loc = op->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, a);
    auto hipOp =
        QLinearMatMulOp::create(rewriter, loc, resultType, *ctxOrFailure, a,
                                aScale, aZp, b, bScale, bZp, yScale, yZp, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return success();
  }
};

} // namespace

void populateQLinearMatMulConversionPatterns(RewritePatternSet &patterns,
                                             MLIRContext *ctx) {
  patterns.add<QLinearMatMulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
