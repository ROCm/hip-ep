/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.less(ctx, a, b, output)
//   -> wrap_less(state, a_ptr, b_ptr, out_ptr, num_elements, data_type)
//
// Output is bool (1 byte/element); data_type identifies the comparison
// operand type (lhs == rhs by ONNX type constraint T).
struct LessOpLowering : public ConvertOpToLLVMPattern<LessOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LessOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = extractContiguousMemRefPtr(adaptor.getLhs(), rewriter, loc);
    Value rhsPtr = extractContiguousMemRefPtr(adaptor.getRhs(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported input element type");

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapLess, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    lhsPtr,
                                  rhsPtr,      outPtr,
                                  numElements, createI64Const(dataType)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateLessLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<LessOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
