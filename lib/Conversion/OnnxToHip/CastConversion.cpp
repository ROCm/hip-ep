//===- CastConversion.cpp - ONNX-to-HIP Cast conversion ------- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Cast -> hip.cast
struct CastToHip : public RewritePattern {
  CastToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Cast", /*benefit=*/1, ctx) {}

  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override;
};

LogicalResult CastToHip::matchAndRewrite(Operation *op,
                                         PatternRewriter &rewriter) const {
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (failed(ctxOrFailure))
    return failure();
  Value context = *ctxOrFailure;

  Location loc = op->getLoc();
  Value input = op->getOperand(0);
  auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
  Value init = createEmptyTensor(rewriter, loc, resultType, input);

  // Map MLIR element type to ONNX DataType enum
  Type targetType = resultType.getElementType();
  int64_t onnxDataType = 0;
  if (targetType.isF16())
    onnxDataType = 10;
  else if (targetType.isBF16())
    onnxDataType = 16;
  else if (targetType.isF32())
    onnxDataType = 1;
  else if (targetType.isF64())
    onnxDataType = 11;
  else if (targetType.isInteger(8))
    onnxDataType = 3;
  else if (targetType.isInteger(16))
    onnxDataType = 5;
  else if (targetType.isInteger(32))
    onnxDataType = 6;
  else if (targetType.isInteger(64))
    onnxDataType = 7;
  else if (targetType.isInteger(1))
    onnxDataType = 9;

  // Validate that we have a supported type
  if (onnxDataType == 0)
    return rewriter.notifyMatchFailure(op, "unsupported cast target type");

  auto toAttr = rewriter.getI64IntegerAttr(onnxDataType);

  auto hipOp = mlir::hip::CastOp::create(rewriter, loc, resultType, context,
                                         input, init, toAttr);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return success();
}

} // namespace

void mlir::hip::populateCastConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<CastToHip>(ctx);
}

} // namespace hip
} // namespace mlir
