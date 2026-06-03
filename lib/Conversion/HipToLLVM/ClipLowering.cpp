/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.clip(ctx, input, min, max, output)
//   -> wrap_clip(state, in_ptr, min_ptr, max_ptr, out_ptr,
//                num_elements, data_type)
//
// `min` and `max` are 0-rank scalar device memrefs of the same dtype as
// `input`. Both pointers are always non-null (the OnnxToHip converter
// synthesizes dtype defaults for absent operands).
struct ClipOpLowering : public ConvertOpToLLVMPattern<ClipOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ClipOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value inPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value minPtr = extractContiguousMemRefPtr(adaptor.getMin(), rewriter, loc);
    Value maxPtr = extractContiguousMemRefPtr(adaptor.getMax(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    auto inType = cast<MemRefType>(op.getInput().getType());
    int64_t dataType = getHipdnnDataType(inType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported input element type");

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapClip, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,    inPtr,
                                  minPtr,      maxPtr,
                                  outPtr,      numElements,
                                  createI64Const(dataType)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateClipLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<ClipOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
