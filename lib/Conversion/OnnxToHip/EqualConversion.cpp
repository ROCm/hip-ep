/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

namespace mlir {
namespace hip {
namespace {

/// onnx.Equal -> hip.equal
struct EqualToHip : public mlir::RewritePattern {
  EqualToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.Equal", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();
    mlir::Value a = op->getOperand(0);
    mlir::Value b = op->getOperand(1);
    auto resultType =
        mlir::cast<mlir::RankedTensorType>(op->getResult(0).getType());

    // hip.equal's LLVM lowering reads only `num_elements` from the
    // result type and forwards it as the SHARED count for both operand
    // pointers (see `lib/Conversion/HipToLLVM/EqualLowering.cpp`).  The
    // kernel itself has no broadcast machinery -- both buffers must
    // already match the result shape verbatim.  ONNX, however, allows
    // NumPy-style broadcast; the canonical real-world trigger is
    // `Equal(input_ids, scalar_const)` in the Qwen-VL embedding graph,
    // where the rhs is a rank-0 i64 Constant.  Without broadcasting
    // here the kernel would read 128 elements out of the 1-element
    // constant buffer and produce garbage -- the downstream NonZero
    // then reports `N=0` for an input that should match.
    //
    // We materialise the broadcast as an explicit `hip.expand` on
    // whichever operand needs it; the existing `wrap_expand` path
    // handles every dtype the elementwise kernels accept (including
    // i64 for this case).
    //
    // Phase 2a: also covers the dynamic-result case (Qwen-VL embedding
    // with batch_size / sequence_length symbolic). The wide operand --
    // the one whose shape already matches resultType -- is used as the
    // `shapeSource` so broadcastToShape can materialise tensor.dim ops
    // at the user op's insertion point (preserving SSA dominance).
    auto aShapeT = mlir::dyn_cast<mlir::RankedTensorType>(a.getType());
    auto bShapeT = mlir::dyn_cast<mlir::RankedTensorType>(b.getType());
    bool aMatches = aShapeT && aShapeT.getShape() == resultType.getShape();
    bool bMatches = bShapeT && bShapeT.getShape() == resultType.getShape();

    // The "wide" operand is the one whose shape already matches the
    // result type -- exactly one side should match for a real broadcast,
    // or both for the trivial "no broadcast needed" case. If neither
    // matches the result, we can't proceed (this happens if both sides
    // need broadcasting, which ONNX rarely emits -- bail to let some
    // other pattern try).
    if (!aMatches && !bMatches)
      return rewriter.notifyMatchFailure(
          op, "neither operand of Equal matches result shape; "
              "broadcast source ambiguous");

    mlir::Value shapeSource = aMatches ? a : b;

    if (!aMatches)
      a = broadcastToShape(rewriter, loc, context, a, resultType, shapeSource);
    if (!bMatches)
      b = broadcastToShape(rewriter, loc, context, b, resultType, shapeSource);

    // After broadcast both operands have the result shape; either is a
    // fine source for createEmptyTensor's dynamic-dim probes.
    mlir::Value init = createEmptyTensor(rewriter, loc, resultType, a);

    auto hipOp = mlir::hip::EqualOp::create(rewriter, loc, resultType, context,
                                            a, b, init);
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateEqualConversionPatterns(RewritePatternSet &patterns,
                                     MLIRContext *ctx) {
  patterns.add<EqualToHip>(ctx);
}

} // namespace hip
} // namespace mlir
