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
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto queryShape = queryType.getShape();
    int64_t batchSize = queryShape[0];
    int64_t seqLen = queryShape[1];
    int64_t queryHidden = queryShape[2];
    int64_t headDimK = queryHidden / op.getQNumHeads();

    // Derive d_v from present_state shape: [B, H_kv, d_k, d_v]
    auto presentStateType = cast<MemRefType>(op.getPresentState().getType());
    auto psShape = presentStateType.getShape();
    int64_t headDimV = psShape[3];

    unsigned elementSizeBytes =
        queryType.getElementType().getIntOrFloatBitWidth() / 8;

    Value batchSizeVal = createI64Const(batchSize);
    Value seqLenVal = createI64Const(seqLen);
    Value headDimKVal = createI64Const(headDimK);
    Value headDimVVal = createI64Const(headDimV);
    Value elemSizeVal = createI64Const(elementSizeBytes);

    // Function signature: state + 6 inputs + 2 outputs + 5 attrs + 5 shape = 19
    SmallVector<Type, 19> paramTypes = {
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

    SmallVector<Value, 19> args = {
        statePtr,
        queryPtr, keyPtr, valuePtr,
        pastStatePtr, decayPtr, betaPtr,
        outputPtr, presentStatePtr,
        qNumHeads, kvNumHeads, scale, chunkSize, updateRule,
        batchSizeVal, seqLenVal, headDimKVal, headDimVVal, elemSizeVal};

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
