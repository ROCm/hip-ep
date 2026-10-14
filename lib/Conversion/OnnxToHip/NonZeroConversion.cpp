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
    // Destination for the indices: [R, capacity] where capacity = numel(X)
    // is the worst-case (every element non-zero) upper bound.
    mlir::Value yInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, resultType.getShape(), resultType.getElementType(),
        mlir::ValueRange{upperBound});

    // Second destination: a device i32 scalar the kernel writes with the
    // actual number of non-zero elements (the data-dependent count N). The
    // capacity buffer above is over-allocated; only the first N columns are
    // meaningful. Reading the count back lets us slice `y` to its true extent
    // so downstream ops and the ORT-reported output shape use N, not capacity.
    auto countType =
        mlir::RankedTensorType::get(/*shape=*/{}, rewriter.getI32Type());
    mlir::Value countInit = mlir::tensor::EmptyOp::create(
        rewriter, loc, countType.getShape(), countType.getElementType());

    // hip.nonzero is multi-result and does not use the single-result inference
    // family, so pass explicit result types.
    auto hipOp = mlir::hip::NonZeroOp::create(
        rewriter, loc, mlir::TypeRange{resultType, countType}, context,
        op->getOperand(0), yInit, countInit,
        rewriter.getI64IntegerAttr(inputDataType));

    // Read the device count back to a host `index` (D2H + stream sync), then
    // slice the capacity buffer to [R, N] so the true extent flows onward.
    mlir::Value count =
        mlir::hip::ReadbackDimOp::create(rewriter, loc, rewriter.getIndexType(),
                                         context, hipOp.getResult(1))
            .getResult();

    mlir::SmallVector<mlir::OpFoldResult, 2> offsets = {
        rewriter.getIndexAttr(0), rewriter.getIndexAttr(0)};
    mlir::SmallVector<mlir::OpFoldResult, 2> sizes = {
        rewriter.getIndexAttr(inputRank), mlir::OpFoldResult(count)};
    mlir::SmallVector<mlir::OpFoldResult, 2> strides = {
        rewriter.getIndexAttr(1), rewriter.getIndexAttr(1)};
    mlir::Value trimmed = mlir::tensor::ExtractSliceOp::create(
        rewriter, loc, resultType, hipOp.getResult(0), offsets, sizes, strides);

    rewriter.replaceOp(op, trimmed);
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
