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
    const int64_t queryHeads = op.getKey()
                                   ? op.getNumHeads()
                                   : op.getNumHeads() + 2 * op.getKvNumHeads();
    const int64_t headSize = queryType.getDimSize(2) / queryHeads;

    auto optionalPtr = [&](Value v) -> Value {
      return v ? extractContiguousMemRefPtr(v, rewriter, loc)
               : LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();
    };
    SmallVector<Value, 64> args = {adaptor.getCtx()};
    for (Value v :
         {adaptor.getQuery(), adaptor.getKey(), adaptor.getValue(),
          adaptor.getPastKey(), adaptor.getPastValue(),
          adaptor.getSelectedIndices(), adaptor.getSelectedCounts(),
          adaptor.getSeqlensK(), adaptor.getGate(), adaptor.getQNormWeight(),
          adaptor.getKNormWeight(), adaptor.getPositionIds(),
          adaptor.getCosCache(), adaptor.getSinCache(), adaptor.getOutput(),
          adaptor.getPresentKey(), adaptor.getPresentValue()}) {
      args.push_back(optionalPtr(v));
    }
    args.append(
        {getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc),
         getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc),
         createI64Const(op.getNumHeads()), createI64Const(op.getKvNumHeads()),
         createI64Const(headSize),
         getMemRefDimSize(pastType, 2, adaptor.getPastKey(), rewriter, loc),
         getMemRefDimSize(indicesType, 1, adaptor.getSelectedIndices(),
                          rewriter, loc)});
    // Dims of an absent operand are passed as 0; the runtime ignores them.
    auto appendDims = [&](Value typed, Value memref, ArrayRef<unsigned> dims) {
      for (unsigned dim : dims) {
        args.push_back(typed
                           ? getMemRefDimSize(cast<MemRefType>(typed.getType()),
                                              dim, memref, rewriter, loc)
                           : createI64Const(0));
      }
    };
    appendDims(op.getPastKey(), adaptor.getPastKey(), {0, 1, 3});
    appendDims(op.getPastValue(), adaptor.getPastValue(), {0, 1, 2, 3});
    appendDims(op.getPresentKey(), adaptor.getPresentKey(), {2, 3});
    appendDims(op.getPresentValue(), adaptor.getPresentValue(), {2, 3});
    appendDims(op.getSelectedIndices(), adaptor.getSelectedIndices(), {0});
    appendDims(op.getSelectedCounts(), adaptor.getSelectedCounts(), {0});
    appendDims(op.getSeqlensK(), adaptor.getSeqlensK(), {0});
    appendDims(op.getKey(), adaptor.getKey(), {0, 1, 2});
    appendDims(op.getValue(), adaptor.getValue(), {0, 1, 2});
    appendDims(op.getGate(), adaptor.getGate(), {0, 1, 2});
    appendDims(op.getQNormWeight(), adaptor.getQNormWeight(), {0});
    appendDims(op.getKNormWeight(), adaptor.getKNormWeight(), {0});
    appendDims(op.getPositionIds(), adaptor.getPositionIds(), {0, 1, 2});
    appendDims(op.getCosCache(), adaptor.getCosCache(), {0, 1});
    appendDims(op.getSinCache(), adaptor.getSinCache(), {0, 1});
    ArrayRef<int64_t> section =
        op.getMropeSection() ? *op.getMropeSection() : ArrayRef<int64_t>{};
    for (unsigned i = 0; i < 3; ++i) {
      args.push_back(createI64Const(i < section.size() ? section[i] : 0));
    }
    args.push_back(createI64Const(op.getMropeLayout().value_or(0)));
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };
    args.push_back(createF32Const(op.getScale().convertToFloat()));
    args.push_back(createF32Const(
        op.getNormEpsilon() ? op.getNormEpsilon()->convertToFloat() : 0.0f));
    args.push_back(optionalPtr(adaptor.getPastSequenceLength()));

    SmallVector<Type, 64> paramTypes(18, ptrType);
    paramTypes.append(43, i64Type);
    paramTypes.append(2, f32Type);
    paramTypes.push_back(ptrType);

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
