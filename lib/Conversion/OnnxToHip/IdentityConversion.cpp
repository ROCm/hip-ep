/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// Identity -> no-op (direct value forwarding)
//===----------------------------------------------------------------------===//
//
// `onnx.Identity` is semantically `output = input`: a pure pass-through.
//
// The most efficient implementation is to forward the input SSA value to
// every user of the result.  This is equivalent to (and strictly cheaper
// than) lowering to a full-range `tensor.extract_slice` / `memref.subview`
// view, because there is no need to materialise even a zero-cost view op
// in the IR.  Downstream passes (bufferization, etc.) see the input value
// directly and behave exactly as if the Identity had never been written.
//
// Matches `ReshapeToStdTensor`'s same-type no-op shortcut, which also uses
// `rewriter.replaceOp(op, data)` to forward the SSA value.
//
// No HIP dialect op, no HipToLLVM lowering, and no runtime support are
// required.

struct IdentityForward : public mlir::RewritePattern {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(IdentityForward)
  IdentityForward(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Identity", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    mlir::Value input = op->getOperand(0);

    // Sanity check: the compiler only supports ranked tensors. ONNX's spec
    // lists many additional type categories (sequence, optional, complex,
    // sub-byte int/float types) which the pipeline never produces in
    // practice; bail out so the failure is visible if one ever appears.
    if (!mlir::isa<mlir::RankedTensorType>(input.getType()) ||
        !mlir::isa<mlir::RankedTensorType>(op->getResult(0).getType()))
      return rewriter.notifyMatchFailure(
          op, "expected ranked tensor input and output");

    // Input and output types must match for Identity — if they ever
    // diverge, that's a frontend bug rather than something we want to
    // paper over by inserting a cast here.
    if (input.getType() != op->getResult(0).getType())
      return rewriter.notifyMatchFailure(op,
                                         "Identity input/output type mismatch");

    rewriter.replaceOp(op, input);
    return mlir::success();
  }
};

} // namespace

void populateIdentityConversionPatterns(RewritePatternSet &patterns,
                                        MLIRContext *ctx) {
  patterns.add<IdentityForward>(ctx);
}

} // namespace hip
} // namespace mlir
