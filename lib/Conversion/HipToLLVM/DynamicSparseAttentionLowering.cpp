/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct DynamicSparseAttentionOpLowering
    : public ConvertOpToLLVMPattern<DynamicSparseAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DynamicSparseAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto pastType = cast<MemRefType>(op.getPastKey().getType());
    auto indicesType = cast<MemRefType>(op.getSelectedIndices().getType());

    SmallVector<Value, 21> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getKey(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getValue(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPastKey(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPastValue(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getSelectedIndices(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getSelectedCounts(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getSeqlensK(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPresentKey(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPresentValue(), rewriter, loc),
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc),
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc),
        createI64Const(op.getNumHeads()),
        createI64Const(op.getKvNumHeads()),
        createI64Const(pastType.getDimSize(3)),
        getMemRefDimSize(pastType, 2, adaptor.getPastKey(), rewriter, loc),
        getMemRefDimSize(indicesType, 1, adaptor.getSelectedIndices(), rewriter,
                         loc),
        LLVM::ConstantOp::create(
            rewriter, loc, f32Type,
            rewriter.getF32FloatAttr(op.getScale().convertToFloat()))};

    SmallVector<Type, 21> paramTypes(12, ptrType);
    paramTypes.append(7, i64Type);
    paramTypes.push_back(f32Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapDynamicSparseAttention, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateDynamicSparseAttentionLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<DynamicSparseAttentionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
