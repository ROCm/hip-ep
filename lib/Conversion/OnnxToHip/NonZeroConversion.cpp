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
  if (elemType.isUnsignedInteger(8))
    return 7; // HIPDNN_EP_DATATYPE_UINT8 (ORT bool, ui8)
  if (elemType.isInteger(1) || elemType.isSignedInteger(8) ||
      elemType.isSignlessInteger(8))
    return 5; // HIPDNN_EP_DATATYPE_INT8 (bool/i1, signed/signless i8)
  return -1;
}

// onnx.NonZero -> hip.nonzero
//
// Input  X: ranked tensor of shape [D0, ..., D{R-1}]. Static dims are
//           used directly; dynamic dims are read via `tensor.dim` at
//           runtime and multiplied into the upper-bound capacity for the
//           output `N` dim.
// Output Y: tensor<R x ? x i64>. The N dim is data-dependent (number of
//           non-zero elements at runtime); the upper bound is numel(X)
//           and is materialised as the runtime `dynSize` of the
//           `tensor.empty` init operand so the buffer is large enough to
//           hold the worst case.
// Output count_buf: tensor<1xi32>. Receives the actual non-zero count at
//           runtime (written by the kernel via atomicAdd). The ONNX op has
//           a single result (the indices), so count_buf has no ONNX-level
//           consumer; downstream ScatterND backtraces its `indices` operand
//           to this op and reads result[1] to bound the rows it processes
//           (no D2H sync). Materialised as a second DPS init below.
//
// Benefit 2 (vs the default 1) so NonZero converts before ScatterND, letting
// ScatterND's backtrace find a `hip.nonzero` rather than deferring on an
// unconverted `onnx.NonZero`.
struct NonZeroToHip : public mlir::RewritePattern {
  NonZeroToHip(mlir::MLIRContext *ctx)
      : RewritePattern("onnx.NonZero", /*benefit=*/2, ctx) {}

  mlir::LogicalResult
  matchAndRewrite(mlir::Operation *op,
                  mlir::PatternRewriter &rewriter) const override {
    if (op->getNumOperands() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected 1 input and 1 output");

    auto inputType =
        mlir::dyn_cast<mlir::RankedTensorType>(op->getOperand(0).getType());
    if (!inputType)
      return rewriter.notifyMatchFailure(op, "input must be a ranked tensor");

    int64_t inputDataType = getHipdnnInputDataType(inputType.getElementType());
    if (inputDataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported NonZero input element type (allowed: f16, f32, "
              "i32, i64, i1, i8, ui8)");

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

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    // Materialise the upper-bound size for the dynamic N dim. When every
    // input dim is static we fold `numel` to a single `arith.constant`
    // (matches the legacy fast path so LIT tests stay intact). With at
    // least one dynamic dim we emit `tensor.dim` per dim and chain the
    // running product as `index` math. The resulting value is the worst
    // case `N` (every element non-zero) and becomes the dynsize operand
    // of the destination `tensor.empty`.
    mlir::Value upperBound;
    if (inputType.hasStaticShape()) {
      int64_t numel = 1;
      for (int64_t d : inputType.getShape())
        numel *= d;
      upperBound = mlir::arith::ConstantIndexOp::create(rewriter, loc, numel);
    } else {
      mlir::Value input = op->getOperand(0);
      upperBound = mlir::arith::ConstantIndexOp::create(rewriter, loc, 1);
      for (int64_t i = 0; i < inputRank; ++i) {
        mlir::Value dim;
        if (inputType.isDynamicDim(i)) {
          dim = mlir::tensor::DimOp::create(rewriter, loc, input, i);
        } else {
          dim = mlir::arith::ConstantIndexOp::create(rewriter, loc,
                                                     inputType.getDimSize(i));
        }
        upperBound =
            mlir::arith::MulIOp::create(rewriter, loc, upperBound, dim);
      }
    }
    mlir::Value indicesInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        mlir::ValueRange{upperBound});

    // Second DPS init: tensor<1xi32> count buffer (static, no dynsize).
    auto countType = mlir::RankedTensorType::get({1}, rewriter.getI32Type());
    mlir::Value countInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, countType.getShape(), countType.getElementType(),
        mlir::ValueRange{});

    // Two results: result[0] = indices (Y), result[1] = count_buf. Custom
    // NonZeroOp::inferReturnTypes (HipDialect.cpp) pushes both in this order.
    auto hipOp = mlir::hip::NonZeroOp::create(
        rewriter, loc, mlir::TypeRange{resultType, countType}, context,
        op->getOperand(0), indicesInit, countInit,
        rewriter.getI64IntegerAttr(inputDataType));
    // Replace the ONNX op (single result) with the indices result only;
    // count_buf stays live via the downstream ScatterND backtrace.
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
