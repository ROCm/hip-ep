/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.qsigmoid(%ctx) ins(%x) outs(%y)
//   -> wrap_qsigmoid(state, x, y, numel, dtype,
//                    input_scale, input_zp, output_scale, output_zp)
struct QSigmoidOpLowering : public ConvertOpToLLVMPattern<QSigmoidOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QSigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected ranked memref operands");
    if (inputType.getElementType() != outputType.getElementType() ||
        !inputType.getElementType().isUnsignedInteger(16))
      return rewriter.notifyMatchFailure(op,
                                         "expected UINT16 input and output");

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto createF32Const = [&](float v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(v));
    };

    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, f32Type,
                                       i64Type, f32Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQSigmoid, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        numElements,
        createI64Const(dataType),
        createF32Const(op.getInputScale().convertToFloat()),
        createI64Const(op.getInputZp()),
        createF32Const(op.getOutputScale().convertToFloat()),
        createI64Const(op.getOutputZp()),
    };

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQSigmoidLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns) {
  patterns.add<QSigmoidOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
