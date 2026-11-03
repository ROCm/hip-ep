/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.qswiglu(%ctx) ins(%gate, %up) outs(%y)
//   -> wrap_qswiglu(state, gate, up, y, numel,
//                   gate_scale, gate_zp, sigmoid_scale, sigmoid_zp,
//                   M_silu, silu_zp, up_scale, up_zp, M_mul, mul_zp,
//                   dq_scale, dq_zp)
//
// M_silu and M_mul are folded exactly as QElementwiseLowering folds a
// hip.qmul (lhs_scale * rhs_scale / output_scale in f32), which is what keeps
// the fused kernel bit-identical to the two qmul launches it replaces.
struct QSwigluOpLowering : public ConvertOpToLLVMPattern<QSwigluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QSwigluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto gateType = dyn_cast<MemRefType>(op.getGate().getType());
    auto upType = dyn_cast<MemRefType>(op.getUp().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!gateType || !upType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked memref operands");
    if (!gateType.getElementType().isF32() ||
        !upType.getElementType().isF32() ||
        !outputType.getElementType().isF32())
      return rewriter.notifyMatchFailure(op, "expected f32 operands");

    float siluScale = op.getSiluScale().convertToFloat();
    float mulScale = op.getMulScale().convertToFloat();
    if (siluScale == 0.0f || mulScale == 0.0f)
      return rewriter.notifyMatchFailure(op, "output scales must be non-zero");
    float gateScale = op.getGateScale().convertToFloat();
    float sigmoidScale = op.getSigmoidScale().convertToFloat();
    float upScale = op.getUpScale().convertToFloat();
    float mSilu = gateScale * sigmoidScale / siluScale;
    float mMul = siluScale * upScale / mulScale;

    auto i64c = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto f32c = [&](float v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(v));
    };

    SmallVector<Type, 17> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, i64Type, // state, ptrs, numel
        f32Type, i64Type, f32Type, i64Type,          // gate, sigmoid
        f32Type, i64Type, f32Type, i64Type,          // M_silu, up
        f32Type, i64Type, f32Type, i64Type};         // M_mul, dq
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQSwiglu, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 17> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getGate(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getUp(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc),
        f32c(gateScale),
        i64c(op.getGateZp()),
        f32c(sigmoidScale),
        i64c(op.getSigmoidZp()),
        f32c(mSilu),
        i64c(op.getSiluZp()),
        f32c(upScale),
        i64c(op.getUpZp()),
        f32c(mMul),
        i64c(op.getMulZp()),
        f32c(op.getDqScale().convertToFloat()),
        i64c(op.getDqZp())};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.fake_quantize(%ctx) ins(%x) outs(%y)
//   -> wrap_fake_quantize(state, x, y, numel, scale, zero_point, qmin, qmax)
struct FakeQuantizeOpLowering : public ConvertOpToLLVMPattern<FakeQuantizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FakeQuantizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !outputType || !inputType.getElementType().isF32() ||
        !outputType.getElementType().isF32())
      return rewriter.notifyMatchFailure(op, "expected f32 memref operands");
    float scale = op.getScale().convertToFloat();
    if (scale == 0.0f)
      return rewriter.notifyMatchFailure(op, "scale must be non-zero");

    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       f32Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapFakeQuantize, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    auto i64c = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    SmallVector<Value, 8> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc),
        LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                 rewriter.getF32FloatAttr(scale)),
        i64c(op.getZeroPoint()),
        i64c(op.getQmin()),
        i64c(op.getQmax())};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQSwigluLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.add<QSwigluOpLowering, FakeQuantizeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
