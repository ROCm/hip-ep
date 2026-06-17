/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.gqa(ctx, query, key, value, past_key, past_value, seqlens_k,
// total_seq_len,
//         output, present_key, present_value) {attributes...}
struct GqaOpLowering : public ConvertOpToLLVMPattern<GqaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GqaOp op, OpAdaptor adaptor,
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

    // Helper: create nullptr
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    // Helper: extract memref pointer or nullptr for optional operands
    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref)
        return nullPtr;
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };

    // === Extract all inputs (required + optional) ===

    Value statePtr = adaptor.getCtx();

    // Required inputs
    Value queryPtr =
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc);
    Value seqlensKPtr =
        extractContiguousMemRefPtr(adaptor.getSeqlensK(), rewriter, loc);
    Value totalSeqLenPtr =
        extractContiguousMemRefPtr(adaptor.getTotalSeqLen(), rewriter, loc);

    // Optional inputs (may be nullptr)
    Value keyPtr = getMemRefPtrOrNull(adaptor.getKey());
    Value valuePtr = getMemRefPtrOrNull(adaptor.getValue());
    Value pastKeyPtr = getMemRefPtrOrNull(adaptor.getPastKey());
    Value pastValuePtr = getMemRefPtrOrNull(adaptor.getPastValue());
    Value cosCachePtr = getMemRefPtrOrNull(adaptor.getCosCache());
    Value sinCachePtr = getMemRefPtrOrNull(adaptor.getSinCache());
    Value positionIdsPtr = getMemRefPtrOrNull(adaptor.getPositionIds());
    Value attentionBiasPtr = getMemRefPtrOrNull(adaptor.getAttentionBias());
    Value headSinkPtr = getMemRefPtrOrNull(adaptor.getHeadSink());
    Value kScalePtr = getMemRefPtrOrNull(adaptor.getKScale());
    Value vScalePtr = getMemRefPtrOrNull(adaptor.getVScale());

    // Output pointers
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value presentKeyPtr =
        extractContiguousMemRefPtr(adaptor.getPresentKey(), rewriter, loc);
    Value presentValuePtr =
        extractContiguousMemRefPtr(adaptor.getPresentValue(), rewriter, loc);
    Value outputQkPtr = getMemRefPtrOrNull(adaptor.getOutputQk());

    // === Extract attributes ===

    Value numHeads = createI64Const(op.getNumHeads());
    Value kvNumHeads = createI64Const(op.getKvNumHeads());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value doRotary = createI64Const(op.getDoRotary());
    Value rotaryInterleaved = createI64Const(op.getRotaryInterleaved());
    Value softcap = createF32Const(op.getSoftcap().convertToFloat());
    Value localWindowSize = createI64Const(op.getLocalWindowSize());
    Value smoothSoftmax = createI64Const(op.getSmoothSoftmax());
    Value qkOutput = createI64Const(op.getQkOutput());
    Value kvCacheBitWidth = createI64Const(op.getKvCacheBitWidth());

    // String attributes need to be converted to enum integers
    // "NONE"=0, "PER_TENSOR"=1, "PER_CHANNEL"=2
    auto quantTypeToEnum = [](llvm::StringRef str) -> int64_t {
      if (str == "NONE")
        return 0;
      if (str == "PER_TENSOR")
        return 1;
      if (str == "PER_CHANNEL")
        return 2;
      return 0; // default to NONE
    };
    Value kQuantType = createI64Const(quantTypeToEnum(op.getKQuantType()));
    Value vQuantType = createI64Const(quantTypeToEnum(op.getVQuantType()));

    // Extract shape info from query memref: [batch, seq_q, num_heads *
    // head_dim]
    // NOTE: Currently only supports static shapes. Dynamic shape support would
    // require extracting dimensions at runtime using MemRefDescriptor::size()
    // and computing headDim dynamically.
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto queryShape = queryType.getShape();
    int64_t batchSize = queryShape[0];
    int64_t seqLenQ = queryShape[1];
    int64_t queryHidden = queryShape[2];
    // Packed QKV: query shape is [B, S, (H + 2*G)*d] instead of [B, S, H*d].
    // Derive head_dim accordingly: d = hidden / (H + 2*G) vs hidden / H.
    bool packedQKV = !op.getKey();
    int64_t headDim =
        packedQKV ? queryHidden / (op.getNumHeads() + 2 * op.getKvNumHeads())
                  : queryHidden / op.getNumHeads();
    unsigned elementSizeBytes =
        queryType.getElementType().getIntOrFloatBitWidth() / 8;

    // Extract seq_len_kv from present_key shape.
    // ONNX GQA uses BNSD layout: [batch, kv_num_heads, total_seq, head_dim]
    auto presentKeyType = cast<MemRefType>(op.getPresentKey().getType());
    auto pkShape = presentKeyType.getShape();
    int64_t seqLenKV = (pkShape.size() == 4) ? pkShape[2] : pkShape[1];

    // past_buf_seq: buffer dimension of past_key (may be max_length for
    // pre-allocated caches, which is larger than actual valid past tokens).
    // Needed so gqa_forward can distinguish buffer stride from valid length.
    int64_t pastBufSeq = 0;
    if (op.getPastKey()) {
      auto pastKeyType = cast<MemRefType>(op.getPastKey().getType());
      auto pastShape = pastKeyType.getShape();
      pastBufSeq = (pastShape.size() == 4) ? pastShape[2] : pastShape[1];
    }

    Value batchSizeVal = createI64Const(batchSize);
    Value seqLenQVal = createI64Const(seqLenQ);
    Value seqLenKVVal = createI64Const(seqLenKV);
    Value pastBufSeqVal = createI64Const(pastBufSeq);
    Value headDimVal = createI64Const(headDim);
    Value elemSizeVal = createI64Const(elementSizeBytes);

    // Function signature matches wrap_group_query_attention() in gqa.cpp
    SmallVector<Type, 38> paramTypes = {
        ptrType, // state
        // Inputs (14 pointers - some may be nullptr)
        ptrType, // query
        ptrType, // key
        ptrType, // value
        ptrType, // past_key
        ptrType, // past_value
        ptrType, // seqlens_k
        ptrType, // total_seq_len
        ptrType, // cos_cache
        ptrType, // sin_cache
        ptrType, // position_ids
        ptrType, // attention_bias
        ptrType, // head_sink
        ptrType, // k_scale
        ptrType, // v_scale
        // Outputs (4 pointers - output_qk may be nullptr)
        ptrType, // output
        ptrType, // present_key
        ptrType, // present_value
        ptrType, // output_qk
        // Attributes (12 values)
        i64Type, // num_heads
        i64Type, // kv_num_heads
        f32Type, // scale
        i64Type, // do_rotary
        i64Type, // rotary_interleaved
        f32Type, // softcap
        i64Type, // local_window_size
        i64Type, // smooth_softmax
        i64Type, // qk_output
        i64Type, // k_quant_type
        i64Type, // v_quant_type
        i64Type, // kv_cache_bit_width
        // Shape info (6 values)
        i64Type, // batch_size
        i64Type, // seq_len_q
        i64Type, // seq_len_kv
        i64Type, // past_buf_seq
        i64Type, // head_dim
        i64Type  // element_size_bytes
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapGQA, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 38> args = {
        statePtr,
        // Inputs (14 pointers)
        queryPtr, keyPtr, valuePtr, pastKeyPtr, pastValuePtr, seqlensKPtr,
        totalSeqLenPtr, cosCachePtr, sinCachePtr, positionIdsPtr,
        attentionBiasPtr, headSinkPtr, kScalePtr, vScalePtr,
        // Outputs (4 pointers)
        outputPtr, presentKeyPtr, presentValuePtr, outputQkPtr,
        // Attributes (12 values)
        numHeads, kvNumHeads, scale, doRotary, rotaryInterleaved, softcap,
        localWindowSize, smoothSoftmax, qkOutput, kQuantType, vQuantType,
        kvCacheBitWidth,
        // Shape info (6 values)
        batchSizeVal, seqLenQVal, seqLenKVVal, pastBufSeqVal, headDimVal,
        elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGqaLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns) {
  patterns.add<GqaOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
