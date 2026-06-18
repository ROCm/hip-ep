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

    // no_causal: emit as i32 (matches wrap_group_query_attention signature).
    Value noCausal = LLVM::ConstantOp::create(
        rewriter, loc, i32Type,
        rewriter.getI32IntegerAttr(op.getNoCausal() ? 1 : 0));

    // Extract shape info from query memref: [batch, seq_q, num_heads *
    // head_dim]. Uses getMemRefDimSize() to handle both static and dynamic
    // dimensions — static dims become LLVM constants, dynamic dims are
    // extracted from the memref descriptor at runtime.
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    Value batchSizeVal =
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc);
    Value seqLenQVal =
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc);
    Value queryHiddenVal =
        getMemRefDimSize(queryType, 2, adaptor.getQuery(), rewriter, loc);

    // Packed QKV: query shape is [B, S, (H + 2*G)*d] instead of [B, S, H*d].
    // Derive head_dim accordingly: d = hidden / (H + 2*G) vs hidden / H.
    bool packedQKV = !op.getKey();
    int64_t headDimDivisor = packedQKV
                                 ? (op.getNumHeads() + 2 * op.getKvNumHeads())
                                 : op.getNumHeads();
    // Compile-time sanity: when query_hidden is static, it must be evenly
    // divisible by the divisor. A mismatch means num_heads/kv_num_heads is
    // wrong for this model — fail loudly here rather than silently producing
    // a truncated head_dim at runtime via integer SDiv.
    //
    // When query_hidden is DYNAMIC, divisibility cannot be checked here.
    // The lowering proceeds with SDiv as before, but emits a one-time
    // build-time advisory so the warning shows up in the compile log;
    // a runtime mismatch (model with wrong num_heads/kv_num_heads) would
    // silently truncate head_dim via integer division — there is no
    // runtime helper to abort cleanly, and the locked design choice for
    // this PR is "warn and continue" rather than "assert and abort".
    if (!queryType.isDynamicDim(2)) {
      int64_t queryHidden = queryType.getDimSize(2);
      if (headDimDivisor <= 0 || queryHidden % headDimDivisor != 0)
        return op.emitOpError()
               << "query_hidden (" << queryHidden << ") not divisible by "
               << (packedQKV ? "(H + 2*G)" : "H") << " = " << headDimDivisor
               << " (num_heads=" << op.getNumHeads()
               << ", kv_num_heads=" << op.getKvNumHeads()
               << (packedQKV ? ", packed QKV)" : ")");
    } else {
      op.emitWarning()
          << "hip.gqa: dynamic query_hidden — divisibility by "
          << (packedQKV ? "(H + 2*G)" : "H") << " = " << headDimDivisor
          << " cannot be verified at compile time; runtime SDiv will "
          << "silently truncate on a model with mismatched num_heads "
          << "(num_heads=" << op.getNumHeads()
          << ", kv_num_heads=" << op.getKvNumHeads()
          << (packedQKV ? ", packed QKV)" : ")");
    }
    Value headDimVal = LLVM::SDivOp::create(rewriter, loc, queryHiddenVal,
                                            createI64Const(headDimDivisor));
    unsigned elementSizeBytes =
        queryType.getElementType().getIntOrFloatBitWidth() / 8;

    // Extract seq_len_kv from present_key shape.
    // ONNX GQA uses BNSD layout: [batch, kv_num_heads, total_seq, head_dim]
    auto presentKeyType = cast<MemRefType>(op.getPresentKey().getType());
    unsigned pkSeqDim = (presentKeyType.getRank() == 4) ? 2 : 1;
    Value seqLenKVVal = getMemRefDimSize(
        presentKeyType, pkSeqDim, adaptor.getPresentKey(), rewriter, loc);

    // past_buf_seq: buffer dimension of past_key (may be max_length for
    // pre-allocated caches, which is larger than actual valid past tokens).
    // Needed so gqa_forward can distinguish buffer stride from valid length.
    Value pastBufSeqVal = createI64Const(0);
    if (op.getPastKey()) {
      auto pastKeyType = cast<MemRefType>(op.getPastKey().getType());
      unsigned pastSeqDim = (pastKeyType.getRank() == 4) ? 2 : 1;
      pastBufSeqVal = getMemRefDimSize(pastKeyType, pastSeqDim,
                                       adaptor.getPastKey(), rewriter, loc);
    }

    Value elemSizeVal = createI64Const(elementSizeBytes);

    // Function signature matches wrap_group_query_attention() in gqa.cpp
    SmallVector<Type, 39> paramTypes = {
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
        i32Type, // no_causal
        // Shape info (6 values)
        i64Type, // batch_size
        i64Type, // seq_len_q
        i64Type, // seq_len_kv
        i64Type, // past_buf_seq
        i64Type, // head_dim
        i64Type, // element_size_bytes
        i32Type  // op_state_slot
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapGQA, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 39> args = {
        statePtr,
        // Inputs (14 pointers)
        queryPtr, keyPtr, valuePtr, pastKeyPtr, pastValuePtr, seqlensKPtr,
        totalSeqLenPtr, cosCachePtr, sinCachePtr, positionIdsPtr,
        attentionBiasPtr, headSinkPtr, kScalePtr, vScalePtr,
        // Outputs (4 pointers)
        outputPtr, presentKeyPtr, presentValuePtr, outputQkPtr,
        // Attributes (13 values)
        numHeads, kvNumHeads, scale, doRotary, rotaryInterleaved, softcap,
        localWindowSize, smoothSoftmax, qkOutput, kQuantType, vQuantType,
        kvCacheBitWidth, noCausal,
        // Shape info (6 values)
        batchSizeVal, seqLenQVal, seqLenKVVal, pastBufSeqVal, headDimVal,
        elemSizeVal,
        // Per-instance op-state slot (threaded by --assign-op-state-slots)
        getOpStateSlotValue(op, rewriter, loc)};

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
