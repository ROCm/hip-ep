/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "OnnxToHipUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

// Map MLIR element type -> HIPDNN_EP_DATATYPE_* enum used by the runtime
// stub. Only the subset that NonZero supports today is enumerated here;
// any other element type fails conversion explicitly so that adding a new
// type later surfaces the gap loudly instead of silently mis-classifying
// the input.
static int64_t getHipdnnInputDataType(mlir::Type elemType) {
  if (elemType.isF32())
    return 0; // HIPDNN_EP_DATATYPE_FLOAT
  if (elemType.isF16())
    return 1; // HIPDNN_EP_DATATYPE_HALF
  if (elemType.isInteger(32))
    return 3; // HIPDNN_EP_DATATYPE_INT32
  if (elemType.isInteger(64))
    return 4; // HIPDNN_EP_DATATYPE_INT64
  // ONNX bool tensors are marshalled as i1 in MLIR and as 1-byte uint8 by
  // the EP. Reuse the INT8 slot so the runtime sees a consistent 1-byte
  // element size; the stub does not interpret the bytes.
  if (elemType.isInteger(1))
    return 5; // HIPDNN_EP_DATATYPE_INT8 (bool, 1 byte)
  return -1;
}

// onnx.NonZero -> hip.nonzero
//
// Input  X: ranked tensor of shape [D0, ..., D{R-1}] (static shape required
//           by this EP, see CLAUDE.md "Models must have static shapes").
// Output Y: tensor<R x ? x i64>. The N dimension is data-dependent (number
//           of non-zero elements at runtime); the upper bound is numel(X)
//           and is materialised as the runtime `dynSize` of the
//           `tensor.empty` init operand so the buffer is large enough to
//           hold the worst case.
struct NonZeroToHip : public mlir::RewritePattern {
  NonZeroToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.NonZero", /*benefit=*/1, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");
    if (!inputType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "NonZero requires static input shape (this EP rejects dynamic "
              "shapes)");

    int64_t inputDataType = getHipdnnInputDataType(inputType.getElementType());
    if (inputDataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported NonZero input element type (allowed: f16, f32, "
              "i32, i64, bool/i1)");

    auto resultType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getResult(0).getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op, "result must be a ranked tensor");
    if (resultType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "NonZero result must be rank-2 (R, N)");
    if (!resultType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(
          op, "NonZero result element type must be i64");

    // First result dim must be the input rank (static); the N dim is
    // dynamic and the upper bound is numel(X).
    int64_t inputRank = inputType.getRank();
    if (resultType.getDimSize(0) != inputRank)
      return rewriter.notifyMatchFailure(
          op, "NonZero result first dim must equal input rank");

    int64_t numel = 1;
    for (int64_t d : inputType.getShape())
      numel *= d;

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    // Materialise the upper-bound size for the dynamic N dim. `numel` is a
    // compile-time constant under the static-shape invariant; the runtime
    // receives it as the worst-case capacity.
    mlir::Value upperBound =
        mlir::arith::ConstantIndexOp::create(rewriter, loc, numel);
    mlir::Value init = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        mlir::ValueRange{upperBound});

    auto hipOp = mlir::hip::NonZeroOp::create(
        rewriter, loc, resultType, context, op->getOperand(0), init,
        rewriter.getI64IntegerAttr(inputDataType));
    rewriter.replaceOp(op, hipOp->getResult(0));
    return mlir::success();
  }
};

} // namespace

void populateNonZeroConversionPatterns(RewritePatternSet &patterns,
                                       MLIRContext *ctx) {
  patterns.add<NonZeroToHip>(ctx);
}

} // namespace hip
} // namespace mlir
