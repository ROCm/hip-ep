/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"

#include <mlir/Dialect/Func/IR/FuncOps.h>
#include <mlir/Dialect/Tosa/IR/TosaOps.h>
#include <mlir/Dialect/UB/IR/UBOps.h>
#include <mlir/IR/BuiltinTypes.h>
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

// tosa.matmul requires both operands to be rank-3 (batch, M/K, K/N). hip.matmul
// accepts NumPy-style broadcasting where B may be rank-2. Prepend a unit batch
// dimension to any rank-2 operand via tosa.reshape; a batch of 1 is broadcast
// against the other operand's batch by tosa.matmul.
static Value reshapeTo3D(Value input, ConversionPatternRewriter &rewriter) {
  auto type = dyn_cast<RankedTensorType>(input.getType());
  if (!type || type.getRank() != 2)
    return input;

  SmallVector<int64_t> shape(type.getShape());
  shape.insert(shape.begin(), 1);
  auto shapeConst = tosa::ConstShapeOp::create(
      rewriter, rewriter.getUnknownLoc(),
      tosa::shapeType::get(rewriter.getContext(), shape.size()),
      rewriter.getIndexTensorAttr(shape));
  return tosa::ReshapeOp::create(rewriter, rewriter.getUnknownLoc(),
                                 type.clone(shape), input, shapeConst);
}

// The hip context and the DPS `outs` buffer are both dropped: the result type
// already encodes the destination.
struct MatMulConverter final : public OpConversionPattern<hip::MatmulOp> {
  using OpConversionPattern<hip::MatmulOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hip::MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Memref mode (post-bufferization) has no SSA result to replace.
    if (op.getNumResults() != 1)
      return rewriter.notifyMatchFailure(op, "expected tensor mode");

    // tosa.matmul is a plain A @ B; transposes must have been folded away.
    if (op.getTransA() != 0 || op.getTransB() != 0)
      return rewriter.notifyMatchFailure(op, "transA/transB unsupported");

    auto resultType = dyn_cast<RankedTensorType>(op.getResult(0).getType());
    if (!resultType || !resultType.hasStaticShape() ||
        resultType.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "expected a static rank-3 tensor");

    Value a = reshapeTo3D(adaptor.getA(), rewriter);
    Value b = reshapeTo3D(adaptor.getB(), rewriter);
    auto aType = dyn_cast<RankedTensorType>(a.getType());
    auto bType = dyn_cast<RankedTensorType>(b.getType());
    if (!aType || aType.getRank() != 3 || !bType || bType.getRank() != 3)
      return rewriter.notifyMatchFailure(op, "operands not rank-3");

    // The quant-info builder appends the (zero) zero-point operands that
    // tosa.matmul requires for float inputs.
    rewriter.replaceOpWithNewOp<tosa::MatMulOp>(op, resultType, a, b);
    return success();
  }
};

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
    patterns.add<AddConverter, MatMulConverter>(ctx);

    if (failed(applyFullConversion(funcOp, conversion, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

} // namespace mlir::hip
