/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Shared lowering for quantized unary activation ops:
//   hip.qsigmoid (future family members: qtanh, qsoftplus, qgelu, ...)
//     -> wrap_qactivation(state, input, output, kind, num_elements, data_type,
//                         x_scale, x_zero_point, out_recip_scale, y_zero_point)
//
// Folding (compile time, mirrors QElementwiseLowering): unlike Add/Mul, the
// activation itself is non-linear, so only the output-side division folds
// into a reciprocal multiply -- there is no algebraic elimination across the
// activation.
//   x_fp = (X - x_zero_point) * x_scale
//   y_fp = activation(x_fp)                 // kind selects the device function
//   Y    = saturate(round(y_fp * out_recip_scale) + y_zero_point)
//   out_recip_scale = 1.0f / y_scale        // division folded here, once,
//                                           // at compile time
template <typename OpTy, HipdnnQActivationKind Kind>
struct QActivationLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();
    Type i32Type = rewriter.getI32Type();

    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    if (!outputType)
      return rewriter.notifyMatchFailure(
          op, "hip.qsigmoid lowering expects ranked memref outs operand");

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    float yScale = op.getYScale().convertToFloat();
    if (yScale == 0.0f)
      return rewriter.notifyMatchFailure(op, "y_scale must be non-zero");
    float outRecipScale = 1.0f / yScale;

    Value numElements =
        computeNumElements(outputType, adaptor.getY(), rewriter, loc);

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto createF32Const = [&](float v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(v));
    };

    // int wrap_qactivation(RuntimeState* state, void* input, void* output,
    //     int64_t kind, int64_t num_elements, int64_t data_type,
    //     float x_scale, int64_t x_zero_point,
    //     float out_recip_scale, int64_t y_zero_point)
    SmallVector<Type, 10> paramTypes = {
        ptrType, ptrType, ptrType, // state, input, output
        i64Type,                   // kind
        i64Type,                   // num_elements
        i64Type,                   // data_type
        f32Type, i64Type,          // x_scale, x_zero_point
        f32Type, i64Type};         // out_recip_scale, y_zero_point

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQActivation, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 10> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc),
        createI64Const(Kind),
        numElements,
        createI64Const(dataType),
        createF32Const(op.getXScale().convertToFloat()),
        createI64Const(op.getXZeroPoint()),
        createF32Const(outRecipScale),
        createI64Const(op.getYZeroPoint())};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct QSigmoidOpLowering
    : public QActivationLowering<QSigmoidOp,
                                 HipdnnQActivationKind::kQActivationSigmoid> {
  using QActivationLowering::QActivationLowering;
};

} // namespace

void populateQActivationLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<QSigmoidOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
