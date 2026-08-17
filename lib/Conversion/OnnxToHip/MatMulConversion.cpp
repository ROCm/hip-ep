/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"
#include "hip/Dialect/IR/HipShapeUtilsMatmulGemm.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.MatMul -> hip.matmul
///
/// Before:
///   %r = "onnx.MatMul"(%a, %b)
///       : (tensor<?x4xf16>, tensor<?x4x?xf16>) -> tensor<?x?x?xf16>
/// After:
///   %batch = tensor.dim %b, %c0
///   %m = tensor.dim %a, %c0
///   %n = tensor.dim %b, %c2
///   %init = tensor.empty(%batch, %m, %n) : tensor<?x?x?xf16>
///   %r = hip.matmul ... outs(%init : tensor<?x?x?xf16>)
struct MatMulToHip : public mlir::RewritePattern {
  MatMulToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.MatMul", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override;
};

mlir::LogicalResult
MatMulToHip::matchAndRewrite(mlir::Operation *op,
                             mlir::PatternRewriter &rewriter) const {
  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);

  auto aType = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
  auto bType = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
  auto resultType =
      mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
  if (!aType || !bType)
    return rewriter.notifyMatchFailure(op, "MatMul operands must be ranked");
  if (!resultType)
    return rewriter.notifyMatchFailure(op, "MatMul result must be ranked");

  // Complete every fallible static check before reification emits tensor.dim
  // or broadcast-selection IR.
  mlir::FailureOr<llvm::SmallVector<int64_t>> inferredShape =
      mlir::hip::inferMatmulShape(aType.getShape(), bType.getShape(),
                                  [&]() { return op->emitError(); });
  if (mlir::failed(inferredShape))
    return mlir::failure();
  if (mlir::failed(mlir::hip::verifyStridedBatchMatmul(
          aType.getShape(), bType.getShape(),
          [&]() { return op->emitError(); })))
    return mlir::failure();
  if (!isResultTypeCompatibleWithInferredShape(resultType, *inferredShape))
    return rewriter.notifyMatchFailure(
        op, "MatMul result type is incompatible with inferred shape");

  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> resultShape =
      mlir::hip::reifyMatmulResultShape(rewriter, loc, a, b,
                                        [&]() { return op->emitError(); });
  if (mlir::failed(resultShape))
    return mlir::failure();
  mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
      rewriter, loc, resultType, *resultShape);
  if (mlir::failed(init))
    return rewriter.notifyMatchFailure(
        op, "MatMul result type is incompatible with inferred shape");
  // The semantic inference family reads the result type from the DPS init.
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, context, a, b, *init);
  rewriter.replaceOp(op, hipOp->getResult(0));
  return mlir::success();
}

} // namespace

void populateMatMulConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<MatMulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
