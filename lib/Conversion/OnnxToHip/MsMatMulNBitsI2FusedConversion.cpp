/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Fusion: com.microsoft.DequantizeLinear + com.microsoft.MatMulNBits(bits=2)
//         + com.microsoft.QuantizeLinear -> hip.ms_matmul_nbits_i2_fused
//
// This pre-lowering rewrite runs BEFORE the individual DQ/MatMul/Q converters
// and fuses the ORCA W2A8 activation-quantization pattern into a single op.
//
// Pattern recognized:
//   %dq_out = onnx.Custom(%act_u16, %dq_scale, %dq_zp)
//              {function_name="DequantizeLinear", domain="com.microsoft"}
//   %mat_out = onnx.Custom(%dq_out, %B, %w_scales, %w_zp, ...)
//              {function_name="MatMulNBits", domain="com.microsoft", bits=2}
//   %q_out   = onnx.Custom(%mat_out, %q_scale, %q_zp)
//              {function_name="QuantizeLinear", domain="com.microsoft"}
//
// Conditions for fusion:
//   - bits=2 (only for the 2-bit path; bits=4/8 go through separate paths)
//   - DQ output has exactly one use (the MatMulNBits)
//   - MatMulNBits output has exactly one use (the QuantizeLinear)
//   - All three ops are in com.microsoft domain
//===----------------------------------------------------------------------===//

struct MsMatMulNBitsI2FusePattern : public mlir::RewritePattern {
  MsMatMulNBitsI2FusePattern(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Custom", /*benefit=*/10, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    // Gate: must be MatMulNBits with bits=2
    auto funcName = op->getAttrOfType<mlir::StringAttr>("function_name");
    if (!funcName || funcName.getValue() != "MatMulNBits")
      return rewriter.notifyMatchFailure(op, "not MatMulNBits");

    auto domain = op->getAttrOfType<mlir::StringAttr>("domain_name");
    if (!domain || domain.getValue() != "com.microsoft")
      return rewriter.notifyMatchFailure(op, "not com.microsoft");

    auto bitsAttr = op->getAttrOfType<mlir::IntegerAttr>("bits");
    if (!bitsAttr || bitsAttr.getSInt() != 2)
      return rewriter.notifyMatchFailure(op, "not bits=2");

    if (op->getNumOperands() < 3 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "wrong operand/result count");

    // MatMulNBits inputs: query(0), B(1), w_scales(2), w_zp(3, optional)
    mlir::Value actIn   = op->getOperand(0);  // fp32 activation (from DQ)

    // Only fuse for prefill (M>1). For M=1 decode: the fused kernel overhead
    // (DQ/Q inside GEMV loop) costs more than running separate DQ+GEMV+Q.
    // Root cause: ORCA's 3-way shared DQ (q/k/v projections) means DQ cannot
    // be eliminated — only Q is saved, but the fused matmul is 20% slower.
    if (auto actType = mlir::dyn_cast<mlir::RankedTensorType>(actIn.getType())) {
      int64_t static_M = 1;
      for (int r = 0; r < actType.getRank() - 1; ++r) {
        int64_t d = actType.getDimSize(r);
        if (d == mlir::ShapedType::kDynamic) { static_M = -1; break; }
        static_M *= d;
      }
      if (static_M == 1)
        return rewriter.notifyMatchFailure(op, "M=1: fused overhead > savings");
    }
    mlir::Value B       = op->getOperand(1);
    mlir::Value wScales = op->getOperand(2);
    mlir::Value wZp;
    if (op->getNumOperands() > 3) {
      mlir::Value v = op->getOperand(3);
      if (v && !mlir::isa<mlir::NoneType>(v.getType()))
        wZp = v;
    }

    mlir::Value matOut = op->getResult(0);

    // Check: MatMulNBits output must have exactly one use (QuantizeLinear)
    if (!matOut.hasOneUse())
      return rewriter.notifyMatchFailure(op, "matmul output has multiple uses");

    mlir::Operation *qOp = *matOut.getUsers().begin();
    {
      auto qFn = qOp->getAttrOfType<mlir::StringAttr>("function_name");
      auto qDom = qOp->getAttrOfType<mlir::StringAttr>("domain_name");
      if (!qFn || qFn.getValue() != "QuantizeLinear" ||
          !qDom || qDom.getValue() != "com.microsoft")
        return rewriter.notifyMatchFailure(op, "downstream is not Q");
    }
    if (qOp->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "Q wrong result count");

    // Check: actIn must come from a DequantizeLinear
    mlir::Operation *dqOp = actIn.getDefiningOp();
    if (!dqOp)
      return rewriter.notifyMatchFailure(op, "actIn has no defining op");
    {
      auto dqFn  = dqOp->getAttrOfType<mlir::StringAttr>("function_name");
      auto dqDom = dqOp->getAttrOfType<mlir::StringAttr>("domain_name");
      if (!dqFn || dqFn.getValue() != "DequantizeLinear" ||
          !dqDom || dqDom.getValue() != "com.microsoft")
        return rewriter.notifyMatchFailure(op, "actIn not from DQ");
    }
    // Allow DQ output with multiple uses (e.g., one DQ feeding q/k/v projections).
    // We fuse this matmul's Q away; the DQ stays in the graph for other users.

    // Extract DQ inputs: in[0]=uint16 act, in[1]=dq_scale, in[2]=dq_zp
    if (dqOp->getNumOperands() < 2)
      return rewriter.notifyMatchFailure(op, "DQ has too few operands");
    mlir::Value actU16  = dqOp->getOperand(0);
    mlir::Value dqScale = dqOp->getOperand(1);
    mlir::Value dqZp;
    if (dqOp->getNumOperands() > 2) {
      mlir::Value v = dqOp->getOperand(2);
      if (v && !mlir::isa<mlir::NoneType>(v.getType()))
        dqZp = v;
    }

    // Extract Q inputs: in[0]=fp32 mat_out, in[1]=q_scale, in[2]=q_zp
    if (qOp->getNumOperands() < 2)
      return rewriter.notifyMatchFailure(op, "Q has too few operands");
    mlir::Value qScale = qOp->getOperand(1);
    mlir::Value qZp;
    if (qOp->getNumOperands() > 2) {
      mlir::Value v = qOp->getOperand(2);
      if (v && !mlir::isa<mlir::NoneType>(v.getType()))
        qZp = v;
    }

    // Get context arg
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return rewriter.notifyMatchFailure(op, "missing context arg");
    mlir::Value ctx = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    // Output type: from Q result (uint16)
    auto qResultType =
        mlir::cast<mlir::RankedTensorType>(qOp->getResult(0).getType());

    // Derive dynamic sizes from actU16 input (same spatial shape as output)
    auto actU16Type = mlir::cast<mlir::RankedTensorType>(actU16.getType());
    llvm::SmallVector<mlir::Value> dynSizes;
    for (auto i : llvm::seq<int64_t>(0, actU16Type.getRank() - 1)) {
      if (qResultType.isDynamicDim(static_cast<unsigned>(i)))
        dynSizes.push_back(mlir::tensor::DimOp::create(rewriter, loc, actU16, i));
    }
    // Last dim (N) is static from qResultType
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, qResultType.getShape(), qResultType.getElementType(),
        dynSizes);

    // Extract N, K, block_size attributes
    auto blockSizeAttr = op->getAttrOfType<mlir::IntegerAttr>("block_size");
    int64_t blockSize = blockSizeAttr ? blockSizeAttr.getSInt() : 64;

    // K from the last dim of actU16
    int64_t K = actU16Type.getDimSize(actU16Type.getRank() - 1);
    // N from the last dim of qResultType
    int64_t N = qResultType.getDimSize(qResultType.getRank() - 1);

    auto NAttr         = rewriter.getI64IntegerAttr(N);
    auto KAttr         = rewriter.getI64IntegerAttr(K);
    auto blockSizeI64  = rewriter.getI64IntegerAttr(blockSize);

    // Emit fused op
    auto fusedOp = mlir::hip::MsMatMulNBitsI2FusedOp::create(
        rewriter, loc, mlir::TypeRange{qResultType},
        ctx, actU16, B, wScales, wZp,
        dqScale, dqZp, qScale, qZp,
        init, NAttr, KAttr, blockSizeI64);

    // Replace the Q op's result with fused op result.
    rewriter.replaceOp(qOp, fusedOp->getResults());
    // MatMulNBits is now dead (Q was its only user). Erase it.
    rewriter.eraseOp(op);
    // Only erase DQ if it has no other users; otherwise leave it for DCE.
    if (dqOp->use_empty())
      rewriter.eraseOp(dqOp);

    return mlir::success();
  }
};

} // namespace

void populateMsMatMulNBitsI2FusedConversionPatterns(
    RewritePatternSet &patterns, MLIRContext *ctx) {
  patterns.add<MsMatMulNBitsI2FusePattern>(ctx);
}

} // namespace hip
} // namespace mlir
