/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/Tosa/Utils/ConversionUtils.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/MLIRContext.h>
#include <mlir/IR/PatternMatch.h>
#include <mlir/Pass/Pass.h>
#include <mlir/Transforms/DialectConversion.h>
#include <mlir/Transforms/GreedyPatternRewriteDriver.h>

namespace mlir::hip {

#define GEN_PASS_DEF_CONVERTHIPTOTOSAPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

// TOSA broadcasts size-1 dimensions only and requires every operand to carry
// the result's rank, so hip's ONNX/NumPy rank-extending broadcast does not
// always survive a 1-1 mapping.
bool isTosaCompatibleOperand(Value operand, RankedTensorType resultType) {
  auto operandType = dyn_cast<RankedTensorType>(operand.getType());
  if (!operandType || !operandType.hasStaticShape())
    return false;
  if (operandType.getElementType() != resultType.getElementType())
    return false;
  if (operandType.getRank() != resultType.getRank())
    return false;

  ArrayRef<int64_t> shape = operandType.getShape();
  ArrayRef<int64_t> resultShape = resultType.getShape();
  for (int64_t i = 0, e = resultType.getRank(); i < e; ++i)
    if (shape[i] != resultShape[i] && shape[i] != 1)
      return false;
  return true;
}

// The hip context and the DPS `outs` buffer are both dropped: the result type
// already encodes the destination.
struct AddConverter final : public OpConversionPattern<hip::AddOp> {
  using OpConversionPattern<hip::AddOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::AddOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode (post-bufferization) has no SSA result to replace.
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");
    if (!isTosaCompatibleOperand(adaptor.getLhs(), resultType) ||
        !isTosaCompatibleOperand(adaptor.getRhs(), resultType))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    rewriter.replaceOpWithNewOp<tosa::AddOp>(op, resultType, adaptor.getLhs(),
                                             adaptor.getRhs());
    return success();
  }
};

// hip.max -> tosa.maximum.
// hip.max carries no NaN attribute. ONNX Max propagates NaNs, which matches
// TOSA's default PROPAGATE (MIGraphX instead uses IGNORE under fast-math).
struct MaxConverter final : public OpConversionPattern<hip::MaxOp> {
  using OpConversionPattern<hip::MaxOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::MaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape())
      return rewriter.notifyMatchFailure(op, "expected a static ranked tensor");

    // ONNX/NumPy rank-extend by prepending 1s, so a ReLU lowered from ONNX is
    // hip.max(tensor<1x64x112x112xf16>, tensor<f16>). EqualizeRanks reshapes
    // the scalar to 1x1x1x1; tosa.maximum then broadcasts the size-1 dims.
    Value lhs = adaptor.getLhs();
    Value rhs = adaptor.getRhs();
    if (failed(tosa::EqualizeRanks(rewriter, op.getLoc(), lhs, rhs)))
      return rewriter.notifyMatchFailure(op, "operand ranks not equalizable");
    if (!isTosaCompatibleOperand(lhs, resultType) ||
        !isTosaCompatibleOperand(rhs, resultType))
      return rewriter.notifyMatchFailure(op, "operands not tosa-broadcastable");

    rewriter.replaceOpWithNewOp<tosa::MaximumOp>(
        op, resultType, lhs, rhs, tosa::NanPropagationMode::PROPAGATE);
    return success();
  }
};

class HipToTosaPass : public impl::ConvertHipToTosaPassBase<HipToTosaPass> {
  void runOnOperation() override {
    auto funcOp = getOperation();
    if (!funcOp->hasAttr("rock.kernel"))
      return;

    MLIRContext *ctx = &getContext();

    ConversionTarget conversion(*ctx);
    conversion.addIllegalDialect<HipDialect>();
    conversion.addLegalDialect<tosa::TosaDialect, func::FuncDialect>();
    conversion.addLegalOp<ub::PoisonOp>();

    RewritePatternSet patterns(ctx);
    patterns.add<AddConverter, MaxConverter>(ctx);

    if (failed(applyFullConversion(funcOp, conversion, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::hip
