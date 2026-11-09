/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.isinf(ctx, x, y)
//   -> wrap_isinf(state, input, output, num_elements, data_type,
//                 detect_negative, detect_positive)
//
// Input is float; output is bool (1 byte/element). data_type identifies the
// input operand type.
struct IsInfOpLowering : public ConvertOpToLLVMPattern<IsInfOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IsInfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    auto inputType = cast<MemRefType>(op.getX().getType());
    auto outputType = cast<MemRefType>(op.getY().getType());

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported input element type");

    Value numElements =
        computeNumElements(outputType, adaptor.getY(), rewriter, loc);

    // 7 params: state + 2 data ptrs + num_elements + data_type +
    // detect_negative + detect_positive
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapIsInf, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc),
        numElements,
        createI64Const(dataType),
        createI64Const(op.getDetectNegative()),
        createI64Const(op.getDetectPositive())};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateIsInfLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<IsInfOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
