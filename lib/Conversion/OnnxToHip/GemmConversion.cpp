/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX Gemm -> HIP Gemm
//
// Before:
//   %init = tensor.empty(tensor.dim %A, 0, tensor.dim %A, 1)
// After:
//   %shape = gemm_shape(%A, %B, transA, transB)
//   %init = tensor.empty(%shape.M, %shape.N)
//===----------------------------------------------------------------------===//
struct GemmToHip : public mlir::RewritePattern {
  GemmToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Gemm", /*benefit=*/1, ctx) {}
  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    auto context = *ctxOrFailure;
    auto inputA = op->getOperand(0);
    auto inputB = op->getOperand(1);
    bool hasInputC = op->getNumOperands() > 2 &&
                     !mlir::isa<mlir::NoneType>(op->getOperand(2).getType());
    auto inputC = hasInputC ? op->getOperand(2) : nullptr;

    float alpha = 1.0f;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("alpha"))
      alpha = attr.getValueAsDouble();
    float beta = 1.0f;
    if (auto attr = op->getAttrOfType<mlir::FloatAttr>("beta"))
      beta = attr.getValueAsDouble();
    int64_t transA = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transA"))
      transA = attr.getSInt();
    int64_t transB = 0;
    if (auto attr = op->getAttrOfType<mlir::IntegerAttr>("transB"))
      transB = attr.getSInt();

    mlir::Location loc = op->getLoc();
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());
    mlir::FailureOr<llvm::SmallVector<mlir::OpFoldResult>> resultShape =
        mlir::hip::reifyGemmResultShape(rewriter, loc, inputA, inputB, inputC,
                                        transA, transB,
                                        [&]() { return op->emitError(); });
    if (mlir::failed(resultShape))
      return mlir::failure();
    mlir::FailureOr<mlir::Value> init = createEmptyTensorFromReifiedShape(
        rewriter, loc, resultType, *resultShape);
    if (mlir::failed(init))
      return rewriter.notifyMatchFailure(
          op, "Gemm result type is incompatible with inferred shape");

    // hip.gemm
    llvm::SmallVector<mlir::NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("alpha", rewriter.getF32FloatAttr(alpha)));
    attrs.push_back(
        rewriter.getNamedAttr("beta", rewriter.getF32FloatAttr(beta)));
    attrs.push_back(
        rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(transA)));
    attrs.push_back(
        rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(transB)));

    llvm::SmallVector<mlir::Value> operands = {context, inputA, inputB};
    if (hasInputC) {
      operands.push_back(inputC);
    }
    operands.push_back(*init);
    // Result type inferred from `init` via InferTypeOpInterface — DPS contract:
    // result type == outs operand type.
    auto hipOp = mlir::hip::GemmOp::create(rewriter, loc, operands, attrs);
    rewriter.replaceOp(op, hipOp.getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateGemmConversionPatterns(RewritePatternSet &patterns,
                                    MLIRContext *ctx) {
  patterns.add<GemmToHip>(ctx);
}

} // namespace hip
} // namespace mlir
