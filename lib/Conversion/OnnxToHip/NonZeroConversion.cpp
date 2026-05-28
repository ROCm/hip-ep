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
// Input  X: ranked tensor of shape [D0, ..., D{R-1}].
// Output Y: tensor<R x ? x i64>. The N dim is data-dependent; upper bound
//           is numel(X).
// Output count_buf: tensor<1xi32>. Receives the actual nonzero count at
//           runtime (written by the kernel via atomicAdd). Downstream ops
//           (ScatterND) can backtrace to this value to limit processing.
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

    int64_t inputRank = inputType.getRank();
    if (resultType.getDimSize(0) != inputRank)
      return rewriter.notifyMatchFailure(
          op, "NonZero result first dim must equal input rank");

    auto ctxOrFailure = getContextArg(op, rewriter);
    if (mlir::failed(ctxOrFailure))
      return mlir::failure();
    mlir::Value context = *ctxOrFailure;

    mlir::Location loc = op->getLoc();

    // Compute upper-bound for the dynamic N dim.
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

    // Output indices buffer: [R, upper_bound]
    mlir::Value indicesInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        mlir::ValueRange{upperBound});

    // Count buffer: tensor<1xi32> — holds the actual nonzero count
    auto countType = mlir::RankedTensorType::get({1}, rewriter.getI32Type());
    mlir::Value countInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, countType.getShape(), countType.getElementType(),
        mlir::ValueRange{});

    auto hipOp = mlir::hip::NonZeroOp::create(
        rewriter, loc, mlir::TypeRange{resultType, countType}, context,
        op->getOperand(0), indicesInit, countInit,
        rewriter.getI64IntegerAttr(inputDataType));

    // The ONNX op has a single result (indices); replace it with result[0]
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
