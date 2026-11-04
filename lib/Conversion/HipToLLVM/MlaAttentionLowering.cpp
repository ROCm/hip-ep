/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.mla_attention(%ctx) ins(%q, %cache, %k_b, %v_b) [k_b_scales(..)]
//     [v_b_scales(..)] [kv_seqlen(..)] outs(%out) {attrs...}
// -> wrap_mla_attention(state, slot, q, cache, k_b, v_b, k_b_scales,
//                       v_b_scales, kv_seqlen, out, num_heads, kv_lora_rank,
//                       qk_nope_dim, qk_rope_dim, v_dim, scale, block_size,
//                       batch, seq_len_q, cache_seq)
struct MlaAttentionOpLowering : public ConvertOpToLLVMPattern<MlaAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MlaAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto i64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    auto ptrOrNull = [&](Value memref) -> Value {
      return memref ? extractContiguousMemRefPtr(memref, rewriter, loc)
                    : nullPtr;
    };

    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto cacheType = cast<MemRefType>(op.getCache().getType());
    Value batch =
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc);
    Value seqLenQ =
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc);
    Value cacheSeq =
        getMemRefDimSize(cacheType, 2, adaptor.getCache(), rewriter, loc);

    SmallVector<Value> args = {
        adaptor.getCtx(),
        getOpStateSlotValue(op, rewriter, loc),
        ptrOrNull(adaptor.getQuery()),
        ptrOrNull(adaptor.getCache()),
        ptrOrNull(adaptor.getKB()),
        ptrOrNull(adaptor.getVB()),
        ptrOrNull(adaptor.getKBScales()),
        ptrOrNull(adaptor.getVBScales()),
        ptrOrNull(adaptor.getKvSeqlen()),
        ptrOrNull(adaptor.getOutput()),
        i64Const(op.getNumHeads()),
        i64Const(op.getKvLoraRank()),
        i64Const(op.getQkNopeDim()),
        i64Const(op.getQkRopeDim()),
        i64Const(op.getVDim()),
        LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                 rewriter.getF32FloatAttr(
                                     op.getScale().convertToFloat())),
        i64Const(op.getBlockSize()),
        batch,
        seqLenQ,
        cacheSeq};

    SmallVector<Type> paramTypes = {ptrType, i32Type};
    paramTypes.append(8, ptrType);
    paramTypes.append(5, i64Type);
    paramTypes.push_back(f32Type);
    paramTypes.append(4, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMlaAttention, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMlaAttentionLoweringPatterns(const LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns) {
  patterns.add<MlaAttentionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
