/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct LinearAttentionOpLowering
    : public ConvertOpToLLVMPattern<LinearAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LinearAttentionOp op, OpAdaptor adaptor,
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

    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref)
        return nullPtr;
      return extractMemRefPtr(memref, rewriter, loc);
    };

    // === Extract all inputs ===
    Value statePtr = adaptor.getCtx();

    Value queryPtr = extractMemRefPtr(adaptor.getQuery(), rewriter, loc);
    Value keyPtr = extractMemRefPtr(adaptor.getKey(), rewriter, loc);
    Value valuePtr = extractMemRefPtr(adaptor.getValue(), rewriter, loc);

    Value pastStatePtr = getMemRefPtrOrNull(adaptor.getPastState());
    Value decayPtr = getMemRefPtrOrNull(adaptor.getDecay());
    Value betaPtr = getMemRefPtrOrNull(adaptor.getBeta());

    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value presentStatePtr =
        extractMemRefPtr(adaptor.getPresentState(), rewriter, loc);

    // === Extract attributes ===
    Value qNumHeads = createI64Const(op.getQNumHeads());
    Value kvNumHeads = createI64Const(op.getKvNumHeads());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value chunkSize = createI64Const(op.getChunkSize());

    // Convert update_rule string to integer enum:
    // "linear"=0, "gated"=1, "delta"=2, "gated_delta"=3
    auto updateRuleToEnum = [](llvm::StringRef str) -> int64_t {
      if (str == "linear")
        return 0;
      if (str == "gated")
        return 1;
      if (str == "delta")
        return 2;
      if (str == "gated_delta")
        return 3;
      return 3; // default to gated_delta
    };
    Value updateRule = createI64Const(updateRuleToEnum(op.getUpdateRule()));

    // === Extract shape info from query: [B, T, H_q * d_k] ===
    // Use getMemRefDimSize so we pick up compile-time constants for static
    // dims and runtime MemRefDescriptor::size for dynamic dims -- otherwise
    // a dynamic shape (typical LLM decode/prefill) would feed
    // ShapedType::kDynamic (a large negative sentinel) into the runtime
    // argument.
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    Value batchSizeVal =
        getMemRefDimSize(queryType, 0, adaptor.getQuery(), rewriter, loc);
    Value seqLenVal =
        getMemRefDimSize(queryType, 1, adaptor.getQuery(), rewriter, loc);

    // head_dim_k = query.dim[2] / q_num_heads.
    // q_num_heads is always a compile-time attribute, so when query.dim[2]
    // is static we fold the division; when it's dynamic we emit a runtime
    // llvm.udiv so the computed head_dim_k is valid for any shape.
    Value queryHiddenVal =
        getMemRefDimSize(queryType, 2, adaptor.getQuery(), rewriter, loc);
    Value qNumHeadsForDiv = createI64Const(op.getQNumHeads());
    Value headDimKVal =
        LLVM::UDivOp::create(rewriter, loc, queryHiddenVal, qNumHeadsForDiv);

    // n_k_heads = key.dim[2] / head_dim_k.  The key tensor is [B, T, n_k*d_k]
    // and n_k may differ from kv_num_heads (it only has to divide it).  Use
    // getMemRefDimSize so dynamic key shapes fall back to a runtime extract.
    auto keyType = cast<MemRefType>(op.getKey().getType());
    Value keyHiddenVal =
        getMemRefDimSize(keyType, 2, adaptor.getKey(), rewriter, loc);
    Value nKHeadsVal =
        LLVM::UDivOp::create(rewriter, loc, keyHiddenVal, headDimKVal);

    // Derive d_v from present_state shape: [B, H_kv, d_k, d_v].
    // present_state is always 4D per spec, so index 3 is safe.
    auto presentStateType = cast<MemRefType>(op.getPresentState().getType());
    Value headDimVVal = getMemRefDimSize(presentStateType, 3,
                                         adaptor.getPresentState(), rewriter,
                                         loc);

    unsigned elementSizeBytes =
        queryType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    // Function signature:
    //   state + 6 inputs + 2 outputs (9 ptrs)
    //   + 3 head-count attrs (q_num_heads, kv_num_heads, n_k_heads)
    //   + 3 scalar attrs (scale, chunk_size, update_rule)
    //   + 5 shape params (batch_size, seq_len, head_dim_k, head_dim_v,
    //                     element_size_bytes)
    // = 20 parameters total.
    SmallVector<Type, 20> paramTypes = {
        ptrType, // state
        ptrType, // query
        ptrType, // key
        ptrType, // value
        ptrType, // past_state (nullable)
        ptrType, // decay (nullable)
        ptrType, // beta (nullable)
        ptrType, // output
        ptrType, // present_state
        i64Type, // q_num_heads
        i64Type, // kv_num_heads
        i64Type, // n_k_heads  (derived from key.dim[2] / head_dim_k)
        f32Type, // scale
        i64Type, // chunk_size
        i64Type, // update_rule
        i64Type, // batch_size
        i64Type, // seq_len
        i64Type, // head_dim_k
        i64Type, // head_dim_v
        i64Type  // element_size_bytes
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapLinearAttention, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 20> args = {
        statePtr,   queryPtr,    keyPtr,      valuePtr,        pastStatePtr,
        decayPtr,   betaPtr,     outputPtr,   presentStatePtr, qNumHeads,
        kvNumHeads, nKHeadsVal,  scale,       chunkSize,       updateRule,
        batchSizeVal, seqLenVal, headDimKVal, headDimVVal,     elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateLinearAttentionLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<LinearAttentionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
