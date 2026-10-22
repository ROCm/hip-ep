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
    auto pastValueType = cast<MemRefType>(op.getPastValue().getType());
    auto presentKeyType = cast<MemRefType>(op.getPresentKey().getType());
    auto presentValueType = cast<MemRefType>(op.getPresentValue().getType());
    auto indicesType = cast<MemRefType>(op.getSelectedIndices().getType());
    auto countsType = cast<MemRefType>(op.getSelectedCounts().getType());
    auto seqlensType = cast<MemRefType>(op.getSeqlensK().getType());
    const int64_t headSize = queryType.getDimSize(2) / op.getNumHeads();

    SmallVector<Value, 35> args = {
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
        createI64Const(headSize),
        getMemRefDimSize(pastType, 2, adaptor.getPastKey(), rewriter, loc),
        getMemRefDimSize(indicesType, 1, adaptor.getSelectedIndices(), rewriter,
                         loc)};
    auto appendDims = [&](MemRefType type, Value memref,
                          ArrayRef<unsigned> dims) {
      for (unsigned dim : dims) {
        args.push_back(getMemRefDimSize(type, dim, memref, rewriter, loc));
      }
    };
    appendDims(pastType, adaptor.getPastKey(), {0, 1, 3});
    appendDims(pastValueType, adaptor.getPastValue(), {0, 1, 2, 3});
    appendDims(presentKeyType, adaptor.getPresentKey(), {2, 3});
    appendDims(presentValueType, adaptor.getPresentValue(), {2, 3});
    appendDims(indicesType, adaptor.getSelectedIndices(), {0});
    appendDims(countsType, adaptor.getSelectedCounts(), {0});
    appendDims(seqlensType, adaptor.getSeqlensK(), {0});
    args.push_back(LLVM::ConstantOp::create(
        rewriter, loc, f32Type,
        rewriter.getF32FloatAttr(op.getScale().convertToFloat())));

    SmallVector<Type, 35> paramTypes(12, ptrType);
    paramTypes.append(21, i64Type);
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
