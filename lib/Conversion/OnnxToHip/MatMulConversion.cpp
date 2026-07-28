/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.MatMul -> hip.hipblaslt.matmul
///
/// Before:
///   %batch = tensor.dim %A, %c0
///   %init = tensor.empty(%batch, ...) : tensor<?x?x?xf16>
/// After:
///   %shape = matmul_shape(%A, %B)
///   %init = tensor.empty(%shape...) : tensor<?x?x?xf16>
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
  auto ctxOrFailure = getContextArg(op, rewriter);
  if (mlir::failed(ctxOrFailure))
    return mlir::failure();
  mlir::Value context = *ctxOrFailure;

  mlir::Location loc = op->getLoc();
  mlir::Value a = op->getOperand(0);
  mlir::Value b = op->getOperand(1);
  auto resultType =
      mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

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
  // Inferred-type Op::create overload: result type is read from the typed
  // outs operand via the auto-emitted MatmulOp::inferReturnTypes (HipOps.td
  // base, autoInfer=1). Equivalent to passing `resultType` explicitly --
  // outs.getType() == resultType by construction here -- but keeps the DPS
  // contract `result_type == outs_operand_type` closed by ODS rather than
  // restated at the callsite.
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
