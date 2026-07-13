/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

//===- PagedAttentionLowering.cpp - hip.paged_attention → wrap_paged_attention

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Before (IR):
//   hip.paged_attention(ctx)
//       ins(%query, %key_cache, %value_cache, %block_table, %slot_mapping,
//           %seq_lens, [%key], [%value], [%cos_cache], [%sin_cache])
//       outs(%output)
//       {num_heads=32, kv_num_heads=8, scale=0.088, ...}
//
// After:
//   llvm.call @wrap_paged_attention(state, query, key, value, key_cache,
//       value_cache, block_table, slot_mapping, seq_lens, output,
//       num_heads, kv_num_heads, scale, do_rotary, cos_cache, sin_cache,
//       num_tokens, batch_size, head_dim, elem_size, block_size,
//       max_blocks_per_seq)

inline constexpr const char *kWrapPagedAttention = "wrap_paged_attention";

struct PagedAttentionOpLowering
    : public ConvertOpToLLVMPattern<PagedAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PagedAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();
    (void)i32Type;

    auto createI64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto createF32Const = [&](float v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(v));
    };
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    auto ptrOrNull = [&](Value v) -> Value {
      if (!v)
        return nullPtr;
      return extractContiguousMemRefPtr(v, rewriter, loc);
    };

    // State pointer.
    Value statePtr = adaptor.getCtx();

    // Required tensor pointers.
    Value queryPtr = extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc);
    Value keyCachePtr = extractContiguousMemRefPtr(adaptor.getKeyCache(), rewriter, loc);
    Value valueCachePtr = extractContiguousMemRefPtr(adaptor.getValueCache(), rewriter, loc);
    Value blockTablePtr = extractContiguousMemRefPtr(adaptor.getBlockTable(), rewriter, loc);
    Value slotMappingPtr = extractContiguousMemRefPtr(adaptor.getSlotMapping(), rewriter, loc);
    Value seqLensPtr = extractContiguousMemRefPtr(adaptor.getSequenceLengths(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Optional tensor pointers.
    Value keyPtr = ptrOrNull(adaptor.getKey());
    Value valuePtr = ptrOrNull(adaptor.getValue());
    Value cosCachePtr = ptrOrNull(adaptor.getCosCache());
    Value sinCachePtr = ptrOrNull(adaptor.getSinCache());

    // Scalar attributes.
    Value numHeads = createI64Const(op.getNumHeads());
    Value kvNumHeads = createI64Const(op.getKvNumHeads());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value doRotary = createI64Const(op.getDoRotary());
    Value numTokens = createI64Const(op.getNumTokens());
    Value batchSize = createI64Const(op.getBatchSize());
    Value headDim = createI64Const(op.getHeadDim());
    Value elemSize = createI64Const(op.getElementSizeBytes());
    Value blockSize = createI64Const(op.getBlockSize());
    Value maxBlocks = createI64Const(op.getMaxBlocksPerSeq());

    // Declare wrap_paged_attention.
    // Signature: (ptr × 11, i64 × 2, f32, i64, ptr, ptr, i64 × 6) -> i32
    SmallVector<Type> argTypes = {
        ptrType, // state
        ptrType, // query
        ptrType, // key
        ptrType, // value
        ptrType, // key_cache
        ptrType, // value_cache
        ptrType, // block_table
        ptrType, // slot_mapping
        ptrType, // sequence_lengths
        ptrType, // output
        i64Type, // num_heads
        i64Type, // kv_num_heads
        f32Type, // scale
        i64Type, // do_rotary
        ptrType, // cos_cache
        ptrType, // sin_cache
        i64Type, // num_tokens
        i64Type, // batch_size
        i64Type, // head_dim
        i64Type, // element_size_bytes
        i64Type, // block_size
        i64Type, // max_blocks_per_seq
    };
    declareRuntimeFunction(module, rewriter, kWrapPagedAttention, argTypes,
                           rewriter.getI32Type());

    // Call.
    SmallVector<Value> args = {
        statePtr, queryPtr, keyPtr,      valuePtr,    keyCachePtr,
        valueCachePtr, blockTablePtr, slotMappingPtr, seqLensPtr, outputPtr,
        numHeads, kvNumHeads, scale,     doRotary,    cosCachePtr,
        sinCachePtr,  numTokens,    batchSize,   headDim,     elemSize,
        blockSize,    maxBlocks,
    };
    auto callOp = rewriter.create<LLVM::CallOp>(
        loc, rewriter.getI32Type(), kWrapPagedAttention, args);

    rewriter.eraseOp(op);
    (void)callOp;
    return success();
  }
};

} // namespace

void populatePagedAttentionLoweringPatterns(LLVMTypeConverter &typeConverter,
                                            RewritePatternSet &patterns) {
  patterns.add<PagedAttentionOpLowering>(typeConverter);
}

} // namespace hip
} // namespace mlir
