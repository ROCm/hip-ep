/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.kv_attention(%ctx) ins(%q, %k, %v) [kv_seqlen(..)] outs(%out) {attrs}
// -> wrap_kv_attention(state, q, k, v, kv_seqlen, out, num_heads,
//                      kv_num_heads, scale, batch, seq_len_q, kv_rows,
//                      key_dim, value_dim)
//
// The runtime is the fused fp16 attention, which needs one head dim for keys
// and values; anything else is a compile error here rather than a failed
// call later.
struct KvAttentionOpLowering : public ConvertOpToLLVMPattern<KvAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(KvAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto keyType = cast<MemRefType>(op.getKey().getType());
    auto valueType = cast<MemRefType>(op.getValue().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    for (MemRefType t : {queryType, keyType, valueType, outputType})
      if (!t.getElementType().isF16())
        return op.emitOpError("runtime supports fp16 only");
    const int64_t keyDim = keyType.getDimSize(3);
    const int64_t valueDim = valueType.getDimSize(3);
    if (keyDim != valueDim)
      return op.emitOpError("runtime needs one head dim for key and value");

    auto i64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    auto ptrOrNull = [&](Value memref) -> Value {
      return memref ? extractContiguousMemRefPtr(memref, rewriter, loc)
                    : nullPtr;
    };

    SmallVector<Value> args = {
        adaptor.getCtx(),
        ptrOrNull(adaptor.getQuery()),
        ptrOrNull(adaptor.getKey()),
        ptrOrNull(adaptor.getValue()),
        ptrOrNull(adaptor.getKvSeqlen()),
        ptrOrNull(adaptor.getOutput()),
        i64Const(op.getNumHeads()),
        i64Const(op.getKvNumHeads()),
        LLVM::ConstantOp::create(
            rewriter, loc, f32Type,
            rewriter.getF32FloatAttr(op.getScale().convertToFloat())),
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc),
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc),
        getMemRefDimSize(keyType, 2, adaptor.getKey(), rewriter, loc),
        i64Const(keyDim),
        i64Const(valueDim)};

    SmallVector<Type> paramTypes = {ptrType};
    paramTypes.append(5, ptrType);
    paramTypes.append(2, i64Type);
    paramTypes.push_back(f32Type);
    paramTypes.append(5, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapKvAttention, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateKvAttentionLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<KvAttentionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
