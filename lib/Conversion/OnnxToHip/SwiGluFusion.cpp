/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- SwiGluFusion.cpp - Fuse exported SwiGLU graphs --------------------===//
//
// Llama-family exports spell SwiGLU as:
//
//   %sigmoid = onnx.Sigmoid(%gate)
//   %silu = onnx.Mul(%gate, %sigmoid)
//   %result = onnx.Mul(%silu, %up)
//
// Both Mul operations are commutative. Fuse only identical tensor types:
// hip.swiglu is a flat elementwise kernel and does not implement broadcasting.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

static bool isOp(Operation *op, StringRef name, unsigned operands = 0) {
  return op && op->getName().getStringRef() == name &&
         (!operands || op->getNumOperands() == operands) &&
         op->getNumResults() == 1;
}

static Value getOtherOperand(Operation *op, Value value) {
  if (!isOp(op, "onnx.Mul", 2))
    return {};
  if (op->getOperand(0) == value)
    return op->getOperand(1);
  if (op->getOperand(1) == value)
    return op->getOperand(0);
  return {};
}

struct SwiGluToHip : public RewritePattern {
  SwiGluToHip(MLIRContext *ctx)
      : RewritePattern("onnx.Mul", /*benefit=*/2, ctx) {}

  LogicalResult matchAndRewrite(Operation *outerMul,
                                PatternRewriter &rewriter) const override {
    if (!isOp(outerMul, "onnx.Mul", 2))
      return rewriter.notifyMatchFailure(outerMul, "expected binary Mul");

    Operation *siluMul = nullptr;
    Value up;
    for (unsigned i = 0; i < 2; ++i) {
      Operation *candidate = outerMul->getOperand(i).getDefiningOp();
      if (isOp(candidate, "onnx.Mul", 2)) {
        siluMul = candidate;
        up = outerMul->getOperand(1 - i);
        break;
      }
    }
    if (!siluMul)
      return rewriter.notifyMatchFailure(outerMul,
                                         "no inner Mul producing SiLU");

    Operation *sigmoid = nullptr;
    Value gate;
    for (unsigned i = 0; i < 2; ++i) {
      Operation *candidate = siluMul->getOperand(i).getDefiningOp();
      if (isOp(candidate, "onnx.Sigmoid", 1)) {
        Value candidateGate = candidate->getOperand(0);
        Value other = getOtherOperand(siluMul, candidate->getResult(0));
        if (other == candidateGate) {
          sigmoid = candidate;
          gate = candidateGate;
          break;
        }
      }
    }
    if (!sigmoid)
      return rewriter.notifyMatchFailure(
          outerMul, "inner Mul is not gate * Sigmoid(gate)");

    if (!sigmoid->getResult(0).hasOneUse() ||
        !siluMul->getResult(0).hasOneUse())
      return rewriter.notifyMatchFailure(
          outerMul, "SwiGLU intermediates must have one use");

    auto gateType = dyn_cast<RankedTensorType>(gate.getType());
    auto upType = dyn_cast<RankedTensorType>(up.getType());
    auto resultType =
        dyn_cast<RankedTensorType>(outerMul->getResult(0).getType());
    if (!gateType || !upType || !resultType)
      return rewriter.notifyMatchFailure(
          outerMul, "SwiGLU requires ranked tensor operands and result");
    if (gateType != upType || gateType != resultType)
      return rewriter.notifyMatchFailure(
          outerMul, "SwiGLU does not support broadcasting");

    Type elemType = gateType.getElementType();
    if (!elemType.isF16() && !elemType.isF32() && !elemType.isBF16() &&
        !elemType.isF64())
      return rewriter.notifyMatchFailure(
          outerMul, "SwiGLU supports only f16, f32, bf16, and f64");

    FailureOr<Value> context = getContextArg(outerMul, rewriter);
    if (failed(context))
      return failure();

    Location loc = outerMul->getLoc();
    Value init = createEmptyTensor(rewriter, loc, resultType, gate);
    auto swiglu = hip::SwigluOp::create(rewriter, loc, resultType, *context,
                                        gate, up, init);
    rewriter.replaceOp(outerMul, swiglu->getResult(0));
    rewriter.eraseOp(siluMul);
    rewriter.eraseOp(sigmoid);
    return success();
  }
};

} // namespace

void populateSwiGluFusionPatterns(RewritePatternSet &patterns,
                                  MLIRContext *ctx) {
  patterns.add<SwiGluToHip>(ctx);
}

} // namespace hip
} // namespace mlir
