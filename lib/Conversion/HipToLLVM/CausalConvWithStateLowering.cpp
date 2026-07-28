/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.causal_conv_with_state(%ctx)
//   ins(%input, %weight, %bias, %past_state)
//   outs(%output, %present_state)
//   {activation = "silu", ndim = 1}
// -> wrap_causal_conv_with_state(state, input, weight, bias, past_state,
//                                 output, present_state,
//                                 batch_size, channels, seq_len,
//                                 kernel_size, ndim, activation,
//                                 element_size_bytes)
struct CausalConvWithStateOpLowering
    : public ConvertOpToLLVMPattern<CausalConvWithStateOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CausalConvWithStateOpLowering)
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CausalConvWithStateOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value weightPtr =
        extractContiguousMemRefPtr(adaptor.getWeight(), rewriter, loc);
    Value biasPtr =
        adaptor.getBias()
            ? extractContiguousMemRefPtr(adaptor.getBias(), rewriter, loc)
            : nullPtr;
    Value pastStatePtr =
        adaptor.getPastState()
            ? extractContiguousMemRefPtr(adaptor.getPastState(), rewriter, loc)
            : nullPtr;
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value presentStatePtr =
        extractContiguousMemRefPtr(adaptor.getPresentState(), rewriter, loc);

    // Extract shape info from input memref: (batch, channels, L) for ndim=1
    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto inputShape = inputType.getShape();
    int64_t rank = inputType.getRank();

    // Input layout: (batch, channels, spatial_dims...)
    // For ndim=1: rank=3, shape = [batch, channels, L]
    // For ndim=2: rank=4, shape = [batch, channels, H, W]
    // For ndim=3: rank=5, shape = [batch, channels, D, H, W]
    if (rank < 3)
      return op.emitError("input must be at least rank-3");

    Value batchSize =
        getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value channels =
        getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);

    // seq_len = product of spatial dimensions (last ndim dims)
    // For ndim=1: just the last dim
    Value seqLen =
        getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);
    for (int64_t i = 3; i < rank; ++i) {
      Value dim =
          getMemRefDimSize(inputType, i, adaptor.getInput(), rewriter, loc);
      seqLen = LLVM::MulOp::create(rewriter, loc, seqLen, dim);
    }

    // Extract kernel size from weight shape: (channels, 1, k_1, ...)
    // kernel_size = product of spatial kernel dims.
    // Use getMemRefDimSize so we pick up compile-time constants for static dims
    // and runtime MemRefDescriptor::size for dynamic dims -- otherwise a
    // dynamic weight shape would multiply ShapedType::kDynamic (a large
    // negative sentinel) into the runtime argument.
    auto weightType = cast<MemRefType>(op.getWeight().getType());
    int64_t weightRank = weightType.getRank();
    Value kernelSizeVal = createI64Const(1);
    for (int64_t i = 2; i < weightRank; ++i) {
      Value dim =
          getMemRefDimSize(weightType, i, adaptor.getWeight(), rewriter, loc);
      kernelSizeVal = LLVM::MulOp::create(rewriter, loc, kernelSizeVal, dim);
    }

    // Attributes
    Value ndimVal = createI64Const(op.getNdim());

    // Convert activation string to enum: "none"=0, "silu"/"swish"=1
    llvm::StringRef activationStr = op.getActivation();
    int64_t activationEnum = 0;
    if (activationStr == "silu" || activationStr == "swish")
      activationEnum = 1;
    Value activationVal = createI64Const(activationEnum);

    unsigned elemSizeBytes =
        inputType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elemSizeBytes);

    // Build function signature
    SmallVector<Type, 15> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // input
        ptrType, // weight
        ptrType, // bias (nullable)
        ptrType, // past_state (nullable)
        ptrType, // output
        ptrType, // present_state
        i64Type, // batch_size
        i64Type, // channels
        i64Type, // seq_len
        i64Type, // kernel_size
        i64Type, // ndim
        i64Type, // activation
        i64Type  // element_size_bytes
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCausalConvWithState, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 15> args = {
        statePtr,   getOpStateSlotValue(op, rewriter, loc),
        inputPtr,   weightPtr,
        biasPtr,    pastStatePtr,
        outputPtr,  presentStatePtr,
        batchSize,  channels,
        seqLen,     kernelSizeVal,
        ndimVal,    activationVal,
        elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateCausalConvWithStateLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<CausalConvWithStateOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
