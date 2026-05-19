/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct EqualOpLowering : public ConvertOpToLLVMPattern<EqualOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(EqualOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc);
    Value rhsPtr = extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported input element type");

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapEqual, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    lhsPtr,
                                  rhsPtr,      outputPtr,
                                  numElements, createI64Const(dataType)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateEqualLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<EqualOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
