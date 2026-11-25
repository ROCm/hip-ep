/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipToLLVMUtils.h"

namespace mlir::hip {
namespace {
struct QwenQsaGqaLowering : ConvertOpToLLVMPattern<QwenQsaGqaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(QwenQsaGqaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Type> types{getPtrType(), rewriter.getI32Type()};
    SmallVector<Value> args{adaptor.getCtx(),
                            getOpStateSlotValue(op, rewriter, loc)};
    for (Value value : adaptor.getOperands().drop_front()) {
      args.push_back(extractContiguousMemRefPtr(value, rewriter, loc));
      types.push_back(getPtrType());
    }
    auto query = cast<MemRefType>(op.getQuery().getType());
    auto cache = cast<MemRefType>(op.getPastKey().getType());
    args.push_back(
        getMemRefDimSize(query, 0, adaptor.getQuery(), rewriter, loc));
    args.push_back(
        getMemRefDimSize(query, 1, adaptor.getQuery(), rewriter, loc));
    args.push_back(
        getMemRefDimSize(cache, 2, adaptor.getPastKey(), rewriter, loc));
    args.push_back(
        getMemRefDimSize(cache, 3, adaptor.getPastKey(), rewriter, loc));
    types.append(4, rewriter.getI64Type());
    auto function = LLVM::lookupOrCreateFn(
        rewriter, op->getParentOfType<ModuleOp>(),
        "wrap_qwen_qsa_group_query_attention", types, rewriter.getI32Type());
    if (failed(function))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *function, args);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace
void populateQwenQsaGqaLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns) {
  patterns.add<QwenQsaGqaLowering>(converter);
}
} // namespace mlir::hip
