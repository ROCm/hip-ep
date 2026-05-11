//===- GemmConversion.cpp - ONNX-to-HIP Gemm conversion ------- *- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.  All rights reserved.
// Licensed under the MIT License.
//
//===----------------------------------------------------------------------===//
//
// Why this conversion exists
// --------------------------
// `onnx.Gemm` is a generalized matrix-multiply with optional alpha/beta
// scales, transposes (`transA`, `transB`), and a broadcast bias `C`.  We
// rewrite to `hip.gemm`, which is a thin wrapper around the hipBLASLt
// matmul descriptor and so directly maps each Gemm attribute onto a
// kernel-launch parameter.
//
// Why not reuse hip.matmul
// ------------------------
// `hip.matmul` is the lower-level op for the inner contraction only -- no
// alpha, no beta, no bias.  Folding bias and scaling into a separate
// elementwise epilogue chain after `hip.matmul` is correct but produces
// extra device traffic.  The dedicated `hip.gemm` op preserves the fused
// shape, letting hipBLASLt pick a kernel that does the multiply-add in a
// single launch.
//
// Non-obvious choices
// -------------------
// * `transA` / `transB` become row/col-major flags on the operand
//   descriptors; we never materialize a `tensor.transpose` op upstream of
//   the matmul, which would force an extra GPU copy.
// * Bias `C` is passed as `OptionalValueRange` so the lowering can pick
//   between "fused matmul + bias" and "matmul, then add bias" depending on
//   what hipBLASLt supports for the operand dtypes.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// ONNX Gemm -> HIP Gemm
//===----------------------------------------------------------------------===//
struct GemmToHip : public RewritePattern {
  GemmToHip(MLIRContext* ctx)
      : RewritePattern("onnx.Gemm", /*benefit=*/1, ctx) {}
  LogicalResult matchAndRewrite(Operation* op,
                                PatternRewriter& rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (failed(ctxOrFailure))
      return failure();
    auto context = *ctxOrFailure;
    auto inputA = op->getOperand(0);
    auto inputB = op->getOperand(1);
    bool hasInputC =
        op->getNumOperands() > 2 && !isa<NoneType>(op->getOperand(2).getType());
    auto inputC = hasInputC ? op->getOperand(2) : nullptr;

    float alpha = 1.0f;
    if (auto attr = op->getAttrOfType<FloatAttr>("alpha"))
      alpha = attr.getValueAsDouble();
    float beta = 1.0f;
    if (auto attr = op->getAttrOfType<FloatAttr>("beta"))
      beta = attr.getValueAsDouble();
    int64_t transA = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("transA"))
      transA = attr.getSInt();
    int64_t transB = 0;
    if (auto attr = op->getAttrOfType<IntegerAttr>("transB"))
      transB = attr.getSInt();

    Location loc = op->getLoc();
    auto resultType = cast<RankedTensorType>(op->getResult(0).getType());
    Value init = createEmptyTensor(rewriter, loc, resultType, inputA);

    // hip.gemm
    llvm::SmallVector<NamedAttribute> attrs;
    attrs.push_back(
        rewriter.getNamedAttr("alpha", rewriter.getF32FloatAttr(alpha)));
    attrs.push_back(
        rewriter.getNamedAttr("beta", rewriter.getF32FloatAttr(beta)));
    attrs.push_back(
        rewriter.getNamedAttr("transA", rewriter.getI64IntegerAttr(transA)));
    attrs.push_back(
        rewriter.getNamedAttr("transB", rewriter.getI64IntegerAttr(transB)));

    llvm::SmallVector<Value> operands = {context, inputA, inputB};
    if (hasInputC) {
      operands.push_back(inputC);
    }
    operands.push_back(init);
    auto hipOp = mlir::hip::GemmOp::create(rewriter, loc, TypeRange{resultType},
                                           operands, attrs);
    rewriter.replaceOp(op, hipOp.getResult(0));
    return success();
  }
};

} // namespace

void mlir::hip::populateGemmConversionPatterns(RewritePatternSet& patterns,
                                               MLIRContext* ctx) {
  patterns.add<GemmToHip>(ctx);
}

} // namespace hip
} // namespace mlir
