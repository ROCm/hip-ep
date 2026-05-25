/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.cast(ctx, input, output)
//   -> wrap_cast(state, input, output, num_elements,
//                src_data_type, dst_data_type)
// Supports both static and dynamic shapes (computes num_elements at runtime).
struct CastOpLowering : public ConvertOpToLLVMPattern<CastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract pointers using alignedPtr (respects memref.view offsets)

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractContiguousMemRefPtrWithSlot(
        op, /*operandIdx=*/1, adaptor.getInput(), statePtr, rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes). For Cat-C-downstream
    // casts, the input descriptor's dynamic dim may itself be slot-bound
    // (the immediate producer is a Cat-C op or another translucent
    // propagator). Use `getMemRefDimSizeWithSlot` on the INPUT operand
    // (operand index 1 = `input`; 0 is `!hip.context`) so the loop
    // count matches the runtime-published extent rather than the
    // upper-bound DPS init buffer size encoded in the descriptor.
    constexpr unsigned kInputOperandIdx = 1;
    Value numElements = createI64Const(1);
    for (auto dimIdx : llvm::seq<int64_t>(inputType.getRank())) {
      Value dimSize = getMemRefDimSizeWithSlot(
          op, kInputOperandIdx, inputType, static_cast<unsigned>(dimIdx),
          adaptor.getInput(), statePtr, rewriter, loc);
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // Phase 2.4: if `hip-reserve-propagator-slots` reserved output
    // slot ids for this cast, publish the (1:1) output dim values.
    // Cast is rank- and shape-preserving so the input dim values
    // computed above are exactly the output dim values.
    auto dimSizeProvider = [&](unsigned resultIdx, unsigned outDim) -> Value {
      if (resultIdx != 0)
        return Value();
      return getMemRefDimSizeWithSlot(op, kInputOperandIdx, inputType, outDim,
                                      adaptor.getInput(), statePtr, rewriter,
                                      loc);
    };
    emitPropagatorSlotPublishes(op, statePtr, dimSizeProvider, rewriter, loc);

    // Get source and destination data type enums
    int64_t srcDataType = getHipdnnDataType(inputType.getElementType());
    int64_t dstDataType = getHipdnnDataType(outputType.getElementType());

    if (srcDataType < 0 || dstDataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.cast");

    Value srcDataTypeVal = createI64Const(srcDataType);
    Value dstDataTypeVal = createI64Const(dstDataType);

    // int wrap_cast(RuntimeState* state, void* input, void* output,
    //     int64_t num_elements, int64_t src_data_type, int64_t dst_data_type)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCast, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    inputPtr,       outputPtr,
                                  numElements, srcDataTypeVal, dstDataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateCastLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<CastOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
