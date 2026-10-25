/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.MatMul -> hip.hipblaslt.matmul
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

  int64_t transA = 0;
  int64_t transB = 0;
  // TransposeMatMulFold stamps these with getI64IntegerAttr, i.e. signless i64,
  // so read through getValue(); getSInt() asserts on a non-signed type.
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("hipdnn.transA"))
    transA = attr.getValue().getSExtValue();
  if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("hipdnn.transB"))
    transB = attr.getValue().getSExtValue();

  auto init = createEmptyTensorFromReifiedShape(
      rewriter, loc, resultType, [&](OpBuilder &builder) {
        return reifyMatmulResultShape(
            builder, loc, a, b, [&] { return op->emitOpError(); }, transA,
            transB);
      });
  if (failed(init))
    return rewriter.notifyMatchFailure(op, "incompatible MatMul result shape");

  llvm::SmallVector<mlir::NamedAttribute> attrs;
  attrs.push_back(
      rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(transA)));
  attrs.push_back(
      rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(transB)));
  llvm::SmallVector<mlir::Value> operands = {context, a, b, *init};
  auto hipOp = mlir::hip::MatmulOp::create(rewriter, loc, operands, attrs);
  rewriter.replaceOp(op, hipOp.getResult(0));
  return mlir::success();
}

} // namespace

void populateMatMulConversionPatterns(RewritePatternSet &patterns,
                                      MLIRContext *ctx) {
  patterns.add<MatMulToHip>(ctx);
}

} // namespace hip
} // namespace mlir
