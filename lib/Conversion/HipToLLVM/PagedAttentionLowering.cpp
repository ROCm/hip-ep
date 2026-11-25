/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipToLLVMUtils.h"

namespace mlir::hip {
namespace {
struct PagedAttentionOpLowering : ConvertOpToLLVMPattern<PagedAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(PagedAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    Type ptr = getPtrType(), i64 = rewriter.getI64Type(),
         f32 = rewriter.getF32Type();
    auto c = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto data = [&](Value v) -> Value {
      return v ? extractContiguousMemRefPtr(v, rewriter, loc)
               : LLVM::ZeroOp::create(rewriter, loc, ptr).getResult();
    };
    auto dim = [&](Value original, Value converted, unsigned axis) -> Value {
      return getMemRefDimSize(cast<MemRefType>(original.getType()), axis,
                              converted, rewriter, loc);
    };
    SmallVector<Value> args = {
        adaptor.getCtx(),
        getOpStateSlotValue(op, rewriter, loc),
        data(adaptor.getQuery()),
        data(adaptor.getKey()),
        data(adaptor.getValue()),
        data(adaptor.getKeyPool()),
        data(adaptor.getValuePool()),
        data(adaptor.getCuSeqlens()),
        data(adaptor.getPastSeqlens()),
        data(adaptor.getBlockTable()),
        data(adaptor.getCosCache()),
        data(adaptor.getSinCache()),
        data(adaptor.getAttentionMetadata()),
        data(adaptor.getOutput()),
        dim(op.getQuery(), adaptor.getQuery(), 0),
        dim(op.getKey(), adaptor.getKey(), 0),
        dim(op.getValue(), adaptor.getValue(), 0),
        dim(op.getOutput(), adaptor.getOutput(), 0),
        dim(op.getPastSeqlens(), adaptor.getPastSeqlens(), 0),
        dim(op.getKeyPool(), adaptor.getKeyPool(), 0),
        dim(op.getValuePool(), adaptor.getValuePool(), 0),
        dim(op.getKeyPool(), adaptor.getKeyPool(), 1),
        dim(op.getValuePool(), adaptor.getValuePool(), 1),
        dim(op.getKeyPool(), adaptor.getKeyPool(), 3),
        dim(op.getBlockTable(), adaptor.getBlockTable(), 0),
        dim(op.getBlockTable(), adaptor.getBlockTable(), 1),
        dim(op.getCuSeqlens(), adaptor.getCuSeqlens(), 0),
        op.getAttentionMetadata()
            ? dim(op.getAttentionMetadata(), adaptor.getAttentionMetadata(), 0)
            : c(0),
        op.getCosCache() ? dim(op.getCosCache(), adaptor.getCosCache(), 0)
                         : c(0),
        op.getSinCache() ? dim(op.getSinCache(), adaptor.getSinCache(), 0)
                         : c(0),
        c(op.getNumHeads()),
        c(op.getKvNumHeads()),
        c(op.getLocalWindowSizeAttr().getInt()),
        c(op.getDoRotary()),
        c(op.getRotaryInterleaved()),
        LLVM::ConstantOp::create(
            rewriter, loc, f32,
            rewriter.getF32FloatAttr(op.getScale().convertToFloat())),
        LLVM::ConstantOp::create(
            rewriter, loc, f32,
            rewriter.getF32FloatAttr(op.getSoftcap().convertToFloat()))};
    SmallVector<Type> types{ptr, rewriter.getI32Type()};
    types.append(12, ptr);
    types.append(21, i64);
    types.append(2, f32);
    auto fn = LLVM::lookupOrCreateFn(rewriter, op->getParentOfType<ModuleOp>(),
                                     "wrap_paged_attention", types,
                                     rewriter.getI32Type());
    if (failed(fn))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *fn, args);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace
void populatePagedAttentionLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns) {
  patterns.add<PagedAttentionOpLowering>(converter);
}
} // namespace mlir::hip
