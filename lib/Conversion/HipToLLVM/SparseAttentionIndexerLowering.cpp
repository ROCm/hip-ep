/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct SparseAttentionIndexerOpLowering
    : public ConvertOpToLLVMPattern<SparseAttentionIndexerOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SparseAttentionIndexerOp op, OpAdaptor adaptor,
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
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto cosType = cast<MemRefType>(op.getCosCache().getType());
    auto pastType = cast<MemRefType>(op.getPastKey().getType());
    const int64_t cosRank = cosType.getRank();
    const int64_t headSize = pastType.getDimSize(2);
    const int64_t numHeads = queryType.getDimSize(2) / headSize - 1;

    Value maskPtr =
        adaptor.getMask()
            ? extractContiguousMemRefPtr(adaptor.getMask(), rewriter, loc)
            : LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();

    SmallVector<Value, 23> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getQueryNormWeight(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getKeyNormWeight(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getCosCache(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getSinCache(), rewriter, loc),
        maskPtr,
        extractContiguousMemRefPtr(adaptor.getPastKey(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getSelectedIndices(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getPresentKey(), rewriter, loc),
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc),
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc),
        getMemRefDimSize(pastType, 1, adaptor.getPastKey(), rewriter, loc),
        createI64Const(numHeads),
        createI64Const(headSize),
        createI64Const(cosType.getDimSize(cosRank - 1)),
        getMemRefDimSize(cosType, cosRank - 2, adaptor.getCosCache(), rewriter,
                         loc),
        createI64Const(cosRank == 3 ? 1 : 0),
        createI64Const(op.getCompressRatio()),
        createI64Const(op.getTokenBudget()),
        createI64Const(queryType.getElementTypeBitWidth() / 8),
        createF32Const(op.getEpsilon().convertToFloat()),
        createF32Const(op.getScale().convertToFloat())};

    SmallVector<Type, 23> paramTypes(10, ptrType);
    paramTypes.append(11, i64Type);
    paramTypes.append(2, f32Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSparseAttentionIndexer, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateSparseAttentionIndexerLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<SparseAttentionIndexerOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
