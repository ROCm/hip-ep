/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
#include "HipToLLVMUtils.h"

namespace mlir::hip {
namespace {
struct QsaGqaLowering : ConvertOpToLLVMPattern<QsaGqaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(QsaGqaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op.getLoc();
    SmallVector<Type> types{getPtrType(), rewriter.getI32Type()};
    SmallVector<Value> args{adaptor.getCtx(),
                            getOpStateSlotValue(op, rewriter, loc)};
    // Thirteen inputs and three DPS outputs, in the ONNX/GQA contract order.
    for (Value v : adaptor.getOperands().drop_front()) {
      args.push_back(extractContiguousMemRefPtr(v, rewriter, loc));
      types.push_back(getPtrType());
    }
    auto q = cast<MemRefType>(op.getQuery().getType());
    auto k = cast<MemRefType>(op.getPastKey().getType());
    args.push_back(getMemRefDimSize(q, 0, adaptor.getQuery(), rewriter, loc));
    args.push_back(getMemRefDimSize(q, 1, adaptor.getQuery(), rewriter, loc));
    args.push_back(getMemRefDimSize(k, 2, adaptor.getPastKey(), rewriter, loc));
    types.append(3, rewriter.getI64Type());
    auto fn = LLVM::lookupOrCreateFn(rewriter, op->getParentOfType<ModuleOp>(),
                                     "wrap_qsa_group_query_attention", types,
                                     rewriter.getI32Type());
    if (failed(fn))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *fn, args);
    rewriter.eraseOp(op);
    return success();
  }
};
} // namespace
void populateQsaGqaLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<QsaGqaLowering>(converter);
}
} // namespace mlir::hip
