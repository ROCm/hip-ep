/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "TorchToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// torch.aten.to.dtype -> hip.cast
/// Maps PyTorch dtype enum values to ONNX dtype values used by hip.cast.
struct TorchToDtypeToHip : public mlir::RewritePattern {
  TorchToDtypeToHip(mlir::MLIRContext *ctx)
      : RewritePattern("torch.aten.to.dtype", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    // Operands: (self, dtype, non_blocking, copy, memory_format)
    mlir::Value input = op->getOperand(0);

    // Get PyTorch dtype enum value from torch.constant.int
    auto pytorchDtype = getTorchConstantInt(op->getOperand(1));
    if (!pytorchDtype)
      return rewriter.notifyMatchFailure(op,
                                         "dtype must be a constant integer");

    // Map PyTorch dtype enum -> ONNX dtype enum
    // PyTorch: {5=f16, 15=bf16, 6=f32, 7=f64, 1=i8, 2=i16, 3=i32, 4=i64,
    //           11=bool}
    // ONNX:   {10=f16, 16=bf16, 1=f32, 11=f64, 3=i8, 5=i16, 6=i32, 7=i64,
    //           9=bool}
    int64_t onnxDtype = 0;
    switch (*pytorchDtype) {
    case 5:
      onnxDtype = 10;
      break; // float16
    case 15:
      onnxDtype = 16;
      break; // bfloat16
    case 6:
      onnxDtype = 1;
      break; // float32
    case 7:
      onnxDtype = 11;
      break; // float64
    case 1:
      onnxDtype = 3;
      break; // int8
    case 2:
      onnxDtype = 5;
      break; // int16
    case 3:
      onnxDtype = 6;
      break; // int32
    case 4:
      onnxDtype = 7;
      break; // int64
    case 11:
      onnxDtype = 9;
      break; // bool
    default:
      return rewriter.notifyMatchFailure(op, "unsupported PyTorch dtype value");
    }

    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::Value init =
        createEmptyTensorForTorch(rewriter, loc, resultType, input);

    auto toAttr = rewriter.getI64IntegerAttr(onnxDtype);
    auto hipOp = mlir::hip::CastOp::create(rewriter, loc, resultType, context,
                                           input, init, toAttr);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateTorchCastConversionPatterns(mlir::RewritePatternSet &patterns,
                                         mlir::MLIRContext *ctx) {
  patterns.add<TorchToDtypeToHip>(ctx);
}

} // namespace hip
} // namespace mlir
