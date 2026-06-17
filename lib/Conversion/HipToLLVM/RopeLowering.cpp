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

    // Extract attributes as constants
    Value interleaved = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getInterleaved()));
    Value numHeads = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getNumHeads()));
    Value rotaryDim = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getRotaryEmbeddingDim()));

    // Compute num_elements (supports dynamic shapes)
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

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
    //     int64_t interleaved, int64_t num_heads, int64_t rotary_dim,
    //     int64_t input_num_elements, int64_t cos_cache_num_elements,
    //     int64_t element_size_bytes)
    SmallVector<Type, 12> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        ptrType, ptrType, i64Type, i64Type,
                                        i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRotaryEmbedding, paramTypes, i32Type);

    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr,    inputPtr,         posIdsPtr,           cosCachePtr,
        sinCachePtr, outputPtr,        interleaved,         numHeads,
        rotaryDim,   inputNumElements, cosCacheNumElements, elemSizeBytes};

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
