/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.multi_head_attention(ctx,
//     query, [key], [value], [bias], [key_padding_mask], [attention_bias],
//     [past_key], [past_value], [past_sequence_length], [cache_indirection],
//     output, [present_key], [present_value], [qk]) {attributes...}
struct MultiHeadAttentionOpLowering
    : public ConvertOpToLLVMPattern<MultiHeadAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MultiHeadAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // These are properties of the runtime wrapper, not of the HIP operation:
    // the TOSA backend also implements packed layouts, biases, and KV caches.
    if (!op.getKey() || !op.getValue())
      return op.emitOpError(
          "default runtime requires separate query, key, and value inputs");
    if (op.getBias() || op.getKeyPaddingMask() || op.getAttentionBias() ||
        op.getPastKey() || op.getPastValue() || op.getPastSequenceLength() ||
        op.getCacheIndirection())
      return op.emitOpError(
          "default runtime does not support bias, masks, past/cache inputs, "
          "or cache indirection");
    if (op.getPresentKey() || op.getPresentValue() || op.getQk())
      return op.emitOpError("default runtime does not support present_key, "
                            "present_value, or qk outputs");
    auto queryShape = cast<ShapedType>(op.getQuery().getType());
    auto keyShape = cast<ShapedType>(op.getKey().getType());
    auto valueShape = cast<ShapedType>(op.getValue().getType());
    if (queryShape.getRank() != 3 || keyShape.getRank() != 3 ||
        valueShape.getRank() != 3)
      return op.emitOpError("default runtime requires rank-3 Q/K/V");
    if (!queryShape.getElementType().isF16())
      return op.emitOpError("default runtime requires fp16 Q/K/V and output");
    if (!queryShape.isDynamicDim(2) && !valueShape.isDynamicDim(2) &&
        queryShape.getDimSize(2) != valueShape.getDimSize(2))
      return op.emitOpError(
          "default runtime requires equal Q/V hidden extents");
    if (op.getMaskFilterValue().convertToFloat() != -10000.0f)
      return op.emitOpError(
          "default runtime supports only mask_filter_value = -10000");

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

    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref)
        return nullPtr;
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };

    // === Pointers ===
    Value statePtr = adaptor.getCtx();

    Value queryPtr =
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc);
    Value keyPtr = getMemRefPtrOrNull(adaptor.getKey());
    Value valuePtr = getMemRefPtrOrNull(adaptor.getValue());
    Value biasPtr = getMemRefPtrOrNull(adaptor.getBias());
    Value keyPaddingMaskPtr = getMemRefPtrOrNull(adaptor.getKeyPaddingMask());
    Value attentionBiasPtr = getMemRefPtrOrNull(adaptor.getAttentionBias());
    Value pastKeyPtr = getMemRefPtrOrNull(adaptor.getPastKey());
    Value pastValuePtr = getMemRefPtrOrNull(adaptor.getPastValue());
    Value pastSequenceLengthPtr =
        getMemRefPtrOrNull(adaptor.getPastSequenceLength());
    Value cacheIndirectionPtr =
        getMemRefPtrOrNull(adaptor.getCacheIndirection());

    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value presentKeyPtr = getMemRefPtrOrNull(adaptor.getPresentKey());
    Value presentValuePtr = getMemRefPtrOrNull(adaptor.getPresentValue());
    Value qkPtr = getMemRefPtrOrNull(adaptor.getQk());

    // === Attributes ===
    Value numHeads = createI64Const(op.getNumHeads());
    Value maskFilterValue =
        createF32Const(op.getMaskFilterValue().convertToFloat());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value unidirectional = createI64Const(op.getUnidirectional());

    // === Shape info from query: [batch, seq_q, hidden] (standard) or
    //                            [batch, seq_kv, num_heads, 3, head_size]
    //                            (packed QKV).
    // Runtime distinguishes by query rank (3 vs 5) and key/value presence.
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    int64_t queryRank = queryType.getRank();

    // Batch size = query dim 0 (always).
    Value batchSize = getMemRefDimSize(queryType, /*dimIdx=*/0,
                                       adaptor.getQuery(), rewriter, loc);

    // Sequence length of query = query dim 1.
    Value seqLenQ = getMemRefDimSize(queryType, /*dimIdx=*/1,
                                     adaptor.getQuery(), rewriter, loc);

    // Hidden / packed-QKV-related fields are extracted from the right dim.
    // Rank 3 standard: query[..., 2] = hidden_size.
    // Rank 5 packed:  query[..., 2] = num_heads, [..., 4] = head_size,
    //                                 hidden_size = num_heads * head_size.
    Value queryHidden;
    if (queryRank >= 3) {
      queryHidden = getMemRefDimSize(queryType, /*dimIdx=*/2,
                                     adaptor.getQuery(), rewriter, loc);
    } else {
      queryHidden = createI64Const(0);
    }

    // head_size derivation:
    //   * packed QKV (rank 5): head_size = query.shape[-1] (dim 4)
    //   * standard (rank 3, separate K/V): head_size = hidden / num_heads
    //   * else (rank 3, packed QKV in hidden): head_size = hidden / (3*H)
    // We compute this in runtime via num_heads + hidden_size to keep static
    // and dynamic shape paths uniform; the runtime stub today only needs the
    // raw values forwarded.
    Value headSize;
    bool packedQkvRank5 = (queryRank == 5);
    if (packedQkvRank5) {
      headSize = getMemRefDimSize(queryType, /*dimIdx=*/4, adaptor.getQuery(),
                                  rewriter, loc);
    } else {
      // Sentinel 0 -- runtime derives head_size from hidden / num_heads.
      headSize = createI64Const(0);
    }

    // KV sequence length: if key memref is present and rank-3
    // [B, seq_kv, hidden], use key.shape[1]; otherwise fall back to 0
    // (runtime treats 0 as "self-attention, seq_kv == seq_q"). For past_key
    // present, runtime adds past_seq.
    Value seqLenKV;
    if (op.getKey()) {
      auto keyType = cast<MemRefType>(op.getKey().getType());
      if (keyType.getRank() >= 2)
        seqLenKV = getMemRefDimSize(keyType, /*dimIdx=*/1, adaptor.getKey(),
                                    rewriter, loc);
      else
        seqLenKV = createI64Const(0);
    } else {
      seqLenKV = createI64Const(0);
    }

    // v_hidden derived from value (rank-3 [B, seq_kv, v_hidden]) when present.
    Value vHidden;
    if (op.getValue()) {
      auto valueType = cast<MemRefType>(op.getValue().getType());
      if (valueType.getRank() >= 3)
        vHidden = getMemRefDimSize(valueType, /*dimIdx=*/2, adaptor.getValue(),
                                   rewriter, loc);
      else
        vHidden = createI64Const(0);
    } else {
      vHidden = createI64Const(0);
    }

    unsigned elementSizeBytes =
        queryType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);
    Value queryRankVal = createI64Const(queryRank);

    // === Build call ===
    // wrap_multi_head_attention signature (declared in hipdnn_ep_runtime.h).
    SmallVector<Type> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        // Inputs (10 pointers - some may be nullptr)
        ptrType, // query
        ptrType, // key
        ptrType, // value
        ptrType, // bias
        ptrType, // key_padding_mask
        ptrType, // attention_bias
        ptrType, // past_key
        ptrType, // past_value
        ptrType, // past_sequence_length
        ptrType, // cache_indirection
        // Outputs (4 pointers - last 3 may be nullptr)
        ptrType, // output
        ptrType, // present_key
        ptrType, // present_value
        ptrType, // qk
        // Attributes (4 values)
        i64Type, // num_heads
        f32Type, // mask_filter_value
        f32Type, // scale
        i64Type, // unidirectional
        // Shape info (7 values)
        i64Type, // batch_size
        i64Type, // seq_len_q
        i64Type, // seq_len_kv
        i64Type, // query_hidden
        i64Type, // v_hidden
        i64Type, // head_size
        i64Type, // query_rank
        i64Type  // element_size_bytes
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMultiHeadAttention, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,
                               // Per-instance op-state slot (threaded by
                               // --assign-op-state-slots)
                               getOpStateSlotValue(op, rewriter, loc),
                               // Inputs (10)
                               queryPtr, keyPtr, valuePtr, biasPtr,
                               keyPaddingMaskPtr, attentionBiasPtr, pastKeyPtr,
                               pastValuePtr, pastSequenceLengthPtr,
                               cacheIndirectionPtr,
                               // Outputs (4)
                               outputPtr, presentKeyPtr, presentValuePtr, qkPtr,
                               // Attributes (4)
                               numHeads, maskFilterValue, scale, unidirectional,
                               // Shape info (7)
                               batchSize, seqLenQ, seqLenKV, queryHidden,
                               vHidden, headSize, queryRankVal, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMultiHeadAttentionLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MultiHeadAttentionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
