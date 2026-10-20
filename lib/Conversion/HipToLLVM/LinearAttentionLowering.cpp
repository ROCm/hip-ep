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
      return extractContiguousMemRefPtr(memref, rewriter, loc);
    };

    // === Extract all inputs ===
    Value statePtr = adaptor.getCtx();

    Value queryPtr =
        extractContiguousMemRefPtr(adaptor.getQuery(), rewriter, loc);
    // Null key and value select the packed QKV layout in query.
    Value keyPtr = getMemRefPtrOrNull(adaptor.getKey());
    Value valuePtr = getMemRefPtrOrNull(adaptor.getValue());

    Value pastStatePtr = getMemRefPtrOrNull(adaptor.getPastState());
    Value decayPtr = getMemRefPtrOrNull(adaptor.getDecay());
    Value betaPtr = getMemRefPtrOrNull(adaptor.getBeta());
    Value decayBiasPtr = getMemRefPtrOrNull(adaptor.getDecayBias());
    Value decayScalePtr = getMemRefPtrOrNull(adaptor.getDecayScale());
    Value qkNormWeightPtr = getMemRefPtrOrNull(adaptor.getQkNormWeight());

    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value presentStatePtr =
        extractContiguousMemRefPtr(adaptor.getPresentState(), rewriter, loc);

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

    // Derive d_v from present_state shape: [B, H_kv, d_k, d_v].
    // present_state is always 4D per spec, so index 3 is safe.
    auto presentStateType = cast<MemRefType>(op.getPresentState().getType());
    auto getHeadDimV = [&] {
      return getMemRefDimSize(presentStateType, 3, adaptor.getPresentState(),
                              rewriter, loc);
    };

    Value headDimKVal;
    Value headDimVVal;
    Value nKHeadsVal;
    if (op.isPackedQKV()) {
      // query = [B, T, H_q*d_k + n_k*d_k + H_kv*d_v]: d_k from present_state,
      // n_k from what q and v leave over.
      headDimVVal = getHeadDimV();
      headDimKVal = getMemRefDimSize(presentStateType, 2,
                                     adaptor.getPresentState(), rewriter, loc);
      Value qWidth = LLVM::MulOp::create(rewriter, loc, qNumHeads, headDimKVal);
      Value vWidth = LLVM::MulOp::create(rewriter, loc, kvNumHeads, headDimVVal);
      Value kWidth = LLVM::SubOp::create(
          rewriter, loc,
          LLVM::SubOp::create(rewriter, loc, queryHiddenVal, qWidth), vWidth);
      nKHeadsVal = LLVM::UDivOp::create(rewriter, loc, kWidth, headDimKVal);
    } else {
      Value qNumHeadsForDiv = createI64Const(op.getQNumHeads());
      headDimKVal =
          LLVM::UDivOp::create(rewriter, loc, queryHiddenVal, qNumHeadsForDiv);

      // n_k_heads = key.dim[2] / head_dim_k.  The key tensor is
      // [B, T, n_k*d_k] and n_k may differ from kv_num_heads (it only has to
      // divide it).  Use getMemRefDimSize so dynamic key shapes fall back to
      // a runtime extract.
      auto keyType = cast<MemRefType>(op.getKey().getType());
      Value keyHiddenVal =
          getMemRefDimSize(keyType, 2, adaptor.getKey(), rewriter, loc);
      nKHeadsVal =
          LLVM::UDivOp::create(rewriter, loc, keyHiddenVal, headDimKVal);
      headDimVVal = getHeadDimV();
    }

    // Derive layout flags for the optional decay / beta tensors so the
    // runtime can pick the correct per-token stride:
    //   decay_per_key_dim = 1 when decay last dim == H_kv * d_k (GLA/RWKV-6)
    //                     = 0 when decay last dim == H_kv (DeltaNet/RetNet)
    //   beta_per_head     = 1 when beta  last dim == H_kv
    //                     = 0 when beta  last dim == 1          (broadcast)
    // Values are meaningless when the corresponding operand is absent, in
    // which case we pass 0.
    auto deriveLayoutFlagEqualsOne = [&](Value operand, MemRefType type,
                                         Value refVal,
                                         int64_t refConst) -> Value {
      // Returns 1 when operand's last dim differs from the reference value
      // (for decay: ref = H_kv -> last == H_kv means per-head layout (0);
      //  for beta : ref = 1    -> last == 1    means broadcast layout (0)).
      int dim = type.getRank() - 1;
      if (!type.isDynamicDim(dim)) {
        int64_t last = type.getDimSize(dim);
        return createI64Const(last == refConst ? 0 : 1);
      }
      Value lastDim = getMemRefDimSize(type, dim, operand, rewriter, loc);
      Value cmp = LLVM::ICmpOp::create(rewriter, loc, LLVM::ICmpPredicate::ne,
                                       lastDim, refVal);
      return LLVM::ZExtOp::create(rewriter, loc, i64Type, cmp);
    };

    Value decayPerKeyDimVal = createI64Const(0);
    if (adaptor.getDecay()) {
      auto decayType = cast<MemRefType>(op.getDecay().getType());
      decayPerKeyDimVal = deriveLayoutFlagEqualsOne(
          adaptor.getDecay(), decayType, kvNumHeads, op.getKvNumHeads());
    }

    Value betaPerHeadVal = createI64Const(0);
    if (adaptor.getBeta()) {
      auto betaType = cast<MemRefType>(op.getBeta().getType());
      Value oneConst = createI64Const(1);
      betaPerHeadVal = deriveLayoutFlagEqualsOne(adaptor.getBeta(), betaType,
                                                 oneConst, /*refConst=*/1);
    }

    int64_t elemTypeEnum = getHipdnnDataType(queryType.getElementType());
    if (elemTypeEnum < 0 || elemTypeEnum > 2) {
      return rewriter.notifyMatchFailure(
          op, "hip.linear_attention requires f32, f16, or bf16 element type");
    }
    Value typeVal = createI64Const(elemTypeEnum);

    Value qkNormEps = createF32Const(op.getQkNormEpsilon().convertToFloat());
    Value qScale = createF32Const(op.getQScale().convertToFloat());
    Value betaSigmoid = createI64Const(op.getBetaSigmoid() ? 1 : 0);

    // Function signature:
    //   state + 9 inputs + 2 outputs (12 ptrs)
    //   + 3 head-count attrs (Hq, Hkv, Nk)
    //   + 2 optional-input layout flags (decay_per_key_dim, beta_per_head)
    //   + 3 scalar attrs (scale, chunk_size, update_rule)
    //   + 5 shape params (B, seq_len, dk, dv, type)
    //   + 3 input-preparation attrs (qk_norm_epsilon, q_scale, beta_sigmoid)
    // = 28 parameters total.
    SmallVector<Type, 28> paramTypes = {
        ptrType, // state
        ptrType, // query
        ptrType, // key (null: packed QKV)
        ptrType, // value (null: packed QKV)
        ptrType, // past_state (nullable)
        ptrType, // decay (nullable)
        ptrType, // beta (nullable)
        ptrType, // decay_bias (nullable)
        ptrType, // decay_scale (nullable)
        ptrType, // qk_norm_weight (nullable)
        ptrType, // output
        ptrType, // present_state
        i64Type, // Hq
        i64Type, // Hkv
        i64Type, // Nk (derived from key.dim[2] / head_dim_k)
        i64Type, // decay_per_key_dim (0/1; 0 when decay absent)
        i64Type, // beta_per_head     (0/1; 0 when beta absent)
        f32Type, // scale
        i64Type, // chunk_size
        i64Type, // update_rule
        i64Type, // B
        i64Type, // seq_len
        i64Type, // dk
        i64Type, // dv
        i64Type, // type (HIPDNN_EP_DATATYPE_FLOAT/HALF/BFLOAT16)
        f32Type, // qk_norm_epsilon
        f32Type, // q_scale
        i64Type  // beta_sigmoid (0/1)
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapLinearAttention, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 28> args = {
        statePtr,       queryPtr,        keyPtr,          valuePtr,
        pastStatePtr,   decayPtr,        betaPtr,         decayBiasPtr,
        decayScalePtr,  qkNormWeightPtr, outputPtr,       presentStatePtr,
        qNumHeads,      kvNumHeads,      nKHeadsVal,      decayPerKeyDimVal,
        betaPerHeadVal, scale,           chunkSize,       updateRule,
        batchSizeVal,   seqLenVal,       headDimKVal,     headDimVVal,
        typeVal,        qkNormEps,       qScale,          betaSigmoid};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateLinearAttentionLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.add<LinearAttentionOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
