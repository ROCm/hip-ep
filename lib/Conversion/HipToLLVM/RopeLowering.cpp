/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.miopen.rope(handle, q, k, cos_cache, sin_cache, start_pos)
struct RopeOpLowering : public ConvertOpToLLVMPattern<RopeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RopeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value posIdsPtr =
        extractContiguousMemRefPtr(adaptor.getPositionIds(), rewriter, loc);
    Value cosCachePtr =
        extractContiguousMemRefPtr(adaptor.getCosCache(), rewriter, loc);
    Value sinCachePtr =
        extractContiguousMemRefPtr(adaptor.getSinCache(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t inputRank = inputType.getRank();
    if (inputRank != 3 && inputRank != 4)
      return rewriter.notifyMatchFailure(
          op, "hip.rope: input rank must be 3 (BSH) or 4 (BSNH/BNSH)");

    int64_t rotaryDimAttr = op.getRotaryEmbeddingDim();
    int64_t numHeadsAttr = op.getNumHeads();

    // Derive (batch, seq_len, num_heads, head_dim, is_bnsh) from input shape
    // and op attributes.
    //
    // 3D input: [batch, seq_len, num_heads * head_dim] (BSNH-equivalent)
    // 4D input: [batch, num_heads, seq_len, head_dim] (BNSH; ONNX default)
    //
    // batch and seq_len may be dynamic (symbolic dimensions in a dynamic-shape
    // model). num_heads and head_dim must be static (architecture constants).
    // Use getMemRefDimSize so dynamic dims read from the memref descriptor at
    // runtime instead of failing with a compile-time sentinel.
    int64_t numHeadsVal = numHeadsAttr;
    int64_t headDimVal = 0;
    int64_t isBnshVal = 0;
    auto inputShape = inputType.getShape();

    if (inputRank == 3) {
      isBnshVal = 0;
      int64_t hidden = inputShape[2];
      if (numHeadsVal <= 0) {
        if (rotaryDimAttr <= 0)
          return rewriter.notifyMatchFailure(
              op, "hip.rope: cannot infer num_heads without rotary_dim "
                  "attribute on 3D input");
        if (hidden == ShapedType::kDynamic)
          return rewriter.notifyMatchFailure(
              op, "hip.rope: cannot infer num_heads from dynamic hidden dim");
        numHeadsVal = hidden / rotaryDimAttr;
      }
      if (numHeadsVal <= 0)
        return rewriter.notifyMatchFailure(op,
                                           "hip.rope: invalid num_heads on 3D");
      if (hidden == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "hip.rope: cannot derive head_dim from dynamic hidden dim");
      headDimVal = hidden / numHeadsVal;
    } else {
      // 4D BNSH layout: [B, num_heads, S, head_dim]
      isBnshVal = 1;
      int64_t shapeNumHeads = inputShape[1];
      headDimVal = inputShape[3];
      if (numHeadsVal <= 0)
        numHeadsVal = shapeNumHeads;
      else if (shapeNumHeads != ShapedType::kDynamic &&
               shapeNumHeads != numHeadsVal)
        return rewriter.notifyMatchFailure(
            op, "hip.rope: num_heads attribute disagrees with 4D input shape");
    }

    if (headDimVal == ShapedType::kDynamic ||
        numHeadsVal == ShapedType::kDynamic)
      return rewriter.notifyMatchFailure(
          op, "hip.rope: dynamic num_heads/head_dim not supported");

    if (rotaryDimAttr <= 0)
      rotaryDimAttr = headDimVal;
    if (rotaryDimAttr > headDimVal)
      return rewriter.notifyMatchFailure(
          op, "hip.rope: rotary_embedding_dim must be <= head_dim");

    // batch and seq_len: use getMemRefDimSize so dynamic dims read from the
    // runtime memref descriptor rather than emitting a broken compile-time
    // kDynamic sentinel.
    Value batchSize =
        getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value seqLen =
        (inputRank == 3)
            ? getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc)
            : getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);

    // Build i64 constants for the static dims and attributes.
    Value interleaved = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getInterleaved()));
    Value numHeads = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(numHeadsVal));
    Value headDim = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(headDimVal));
    Value rotaryDim = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(rotaryDimAttr));
    Value isBnsh = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(isBnshVal));

    auto cosCacheType = cast<MemRefType>(op.getCosCache().getType());
    Value cosCacheNumElements =
        computeNumElements(cosCacheType, adaptor.getCosCache(), rewriter, loc);

    // Compute element_size_bytes
    unsigned elementSizeBits =
        inputType.getElementType().getIntOrFloatBitWidth();
    Value elemSizeBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(elementSizeBits / 8));

    // Function signature: wrap_rotary_embedding(
    //     RuntimeState* state, void* input, void* position_ids,
    //     void* cos_cache, void* sin_cache, void* output,
    //     int64_t interleaved, int64_t batch_size, int64_t seq_len,
    //     int64_t num_heads, int64_t head_dim, int64_t rotary_dim,
    //     int64_t cos_cache_num_elements, int64_t element_size_bytes,
    //     int64_t is_bnsh)
    SmallVector<Type, 15> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRotaryEmbedding, paramTypes, i32Type);

    if (failed(funcOp))
      return failure();

    SmallVector<Value, 15> args = {
        statePtr,  inputPtr,    posIdsPtr,           cosCachePtr,   sinCachePtr,
        outputPtr, interleaved, batchSize,           seqLen,        numHeads,
        headDim,   rotaryDim,   cosCacheNumElements, elemSizeBytes, isBnsh};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateRopeLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<RopeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
