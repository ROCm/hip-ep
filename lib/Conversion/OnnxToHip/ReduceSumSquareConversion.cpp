/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ReduceSumSquareConversion.cpp - Decompose onnx.ReduceSumSquare ----===//
//
// `onnx.ReduceSumSquare` is rewritten to the ONNX function body that defines
// it, a chain of ops that already have HIP converters:
//
//   %x_sq = onnx.Mul(%x, %x)
//   %y    = onnx.ReduceSum(%x_sq, %axes) {keepdims, noop_with_empty_axes}
//
// axes (operand or attribute), keepdims and noop_with_empty_axes pass through
// unchanged, so the ReduceSum converter owns the destination shape and the
// reduced-axes handling. x^2 is rounded to the element type before the sum,
// as in the function body; the fp16 reduce_sum kernel accumulates in float.
//
// Like LpNormalization, this lives in the convert-onnx-to-hip pre-lowering
// loop: the main compute-op driver is single-shot and would leave the
// freshly created onnx.Mul / onnx.ReduceSum unconverted.
//
//===----------------------------------------------------------------------===//

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {

namespace {

struct ReduceSumSquareDecompose : public mlir::RewritePattern {
  ReduceSumSquareDecompose(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.ReduceSumSquare", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() < 1 || op->getNumOperands() > 2 ||
        op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "reduce_sum_square.arity");

    mlir::Location loc = op->getLoc();
    mlir::Value x = op->getOperand(0);

    mlir::OperationState mulState(loc, "onnx.Mul");
    mulState.addOperands({x, x});
    mulState.addTypes(x.getType());
    mlir::Value xSq = rewriter.create(mulState)->getResult(0);

    mlir::OperationState reduceState(loc, "onnx.ReduceSum");
    reduceState.addOperands(xSq);
    if (op->getNumOperands() == 2)
      reduceState.addOperands(op->getOperand(1));
    for (llvm::StringRef name : {"axes", "keepdims", "noop_with_empty_axes"})
      if (mlir::Attribute attr = op->getAttr(name))
        reduceState.addAttribute(name, attr);
    reduceState.addTypes(op->getResult(0).getType());
    mlir::Value result = rewriter.create(reduceState)->getResult(0);

    rewriter.replaceOp(op, result);
    return mlir::success();
  }
};

} // namespace

void populateReduceSumSquareConversionPatterns(RewritePatternSet &patterns,
                                               MLIRContext *ctx) {
  patterns.add<ReduceSumSquareDecompose>(ctx);
}

} // namespace hip
} // namespace mlir
