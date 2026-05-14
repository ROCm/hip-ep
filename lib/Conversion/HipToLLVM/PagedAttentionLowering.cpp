/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct PagedAttentionOpLowering
    : public ConvertOpToLLVMPattern<PagedAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PagedAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    Value statePtr = adaptor.getCtx();
    Value queryPtr =
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc);
    Value keyPtr =
        extractOptionalMemRefPtr(adaptor.getKey(), rewriter, loc);
    Value valuePtr =
        extractOptionalMemRefPtr(adaptor.getValue(), rewriter, loc);
    Value keyCachePtr =
        extractContiguousMemRefPtr(adaptor.getKeyCache(), rewriter, loc);
    Value valueCachePtr =
        extractContiguousMemRefPtr(adaptor.getValueCache(), rewriter, loc);
    Value cumSeqPtr = extractContiguousMemRefPtr(
        adaptor.getCumulativeSequenceLength(), rewriter, loc);
    Value pastSeqlensPtr =
        extractContiguousMemRefPtr(adaptor.getPastSeqlens(), rewriter, loc);
    Value blockTablePtr =
        extractContiguousMemRefPtr(adaptor.getBlockTable(), rewriter, loc);
    Value cosPtr =
        extractOptionalMemRefPtr(adaptor.getCosCache(), rewriter, loc);
    Value sinPtr =
        extractOptionalMemRefPtr(adaptor.getSinCache(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value keyCacheOutPtr =
        extractOptionalMemRefPtr(adaptor.getKeyCacheOut(), rewriter, loc);
    Value valueCacheOutPtr =
        extractOptionalMemRefPtr(adaptor.getValueCacheOut(), rewriter, loc);

    Value numHeads = createI64Const(op.getNumHeads());
    Value kvNumHeads = createI64Const(op.getKvNumHeads());
    Value doRotary = createI64Const(op.getDoRotary());
    Value rotaryInterleaved = createI64Const(op.getRotaryInterleaved());
    Value localWindowSize = createI64Const(op.getLocalWindowSize());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value softcap = createF32Const(op.getSoftcap().convertToFloat());

    auto queryTy = cast<MemRefType>(op.getQuery().getType());
    MemRefDescriptor qDesc(adaptor.getQuery());
    Value numTokens =
        queryTy.isDynamicDim(0)
            ? qDesc.size(rewriter, loc, 0)
            : createI64Const(queryTy.getDimSize(0));
    Value queryDim1 =
        queryTy.isDynamicDim(1)
            ? qDesc.size(rewriter, loc, 1)
            : createI64Const(queryTy.getDimSize(1));
    Value elemSizeBytes = createI64Const(
        queryTy.getElementType().getIntOrFloatBitWidth() / 8);

    SmallVector<Type, 32> paramTypes = {
        ptrType, // state
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType,
        ptrType, ptrType, // query..sin (10)
        ptrType, ptrType, ptrType, // outputs
        i64Type, i64Type, i64Type, i64Type, i64Type, // attrs i64
        f32Type, f32Type,                           // scale, softcap
        i64Type, i64Type, i64Type                   // shape
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapPagedAttention, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 32> args = {
        statePtr,
        queryPtr,
        keyPtr,
        valuePtr,
        keyCachePtr,
        valueCachePtr,
        cumSeqPtr,
        pastSeqlensPtr,
        blockTablePtr,
        cosPtr,
        sinPtr,
        outputPtr,
        keyCacheOutPtr,
        valueCacheOutPtr,
        numHeads,
        kvNumHeads,
        doRotary,
        rotaryInterleaved,
        localWindowSize,
        scale,
        softcap,
        numTokens,
        queryDim1,
        elemSizeBytes,
    };

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populatePagedAttentionLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<PagedAttentionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
