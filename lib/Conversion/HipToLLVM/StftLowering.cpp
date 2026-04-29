/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.stft -> wrap_stft.
//
// wrap_stft signature (matches lib/Runtime/real/stft.cpp):
//   int wrap_stft(state, signal, window, output,
//                 batch, signal_len,
//                 frame_step, frame_length,
//                 n_frames, n_freqs,
//                 onesided, data_type);
//
// We pull batch/signal_len from the signal memref descriptor at runtime
// and compute n_frames from signal_len, supporting dynamic signal lengths.
// The window pointer is null when the optional operand is absent.

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {

// Local fallback declarations.  These are also expected to live in
// HipToLLVMUtils.h, but this file ships them inline so we don't break when
// a parallel coverage commit reformats that header before ours lands.
#ifndef HIP_STFT_KWRAP_DEFINED
#define HIP_STFT_KWRAP_DEFINED
inline constexpr const char *kWrapStftLocal = "wrap_stft";
void populateStftLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
#endif

namespace {

struct StftLowering : public ConvertOpToLLVMPattern<StftOp> {
  using ConvertOpToLLVMPattern<StftOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(StftOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value signalPtr = extractMemRefPtr(adaptor.getSignal(), rewriter, loc);
    Value windowPtr = extractOptionalMemRefPtr(adaptor.getWindow(), rewriter,
                                                loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto signalType = dyn_cast<MemRefType>(op.getSignal().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!signalType || !outputType)
      return rewriter.notifyMatchFailure(
          op, "hip.stft lowering requires memref operands");
    if (signalType.getRank() != 2)
      return rewriter.notifyMatchFailure(
          op, "hip.stft lowering requires a rank-2 signal "
              "(batch, signal_length)");
    if (outputType.getRank() != 4)
      return rewriter.notifyMatchFailure(
          op, "hip.stft lowering requires a rank-4 output "
              "(batch, n_frames, n_freqs, 2)");
    if ((!signalType.getElementType().isF32() &&
         !signalType.getElementType().isF16()) ||
        (!outputType.getElementType().isF32() &&
         !outputType.getElementType().isF16()))
      return rewriter.notifyMatchFailure(
          op, "hip.stft lowering currently supports f32/f16 only");

    int64_t frameStep = op.getFrameStep();
    int64_t frameLength = op.getFrameLength();
    int64_t onesided = op.getOnesided();
    int64_t nFreqs = onesided ? frameLength / 2 + 1 : frameLength;

    int64_t dataType = getHipdnnDataType(signalType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.stft unsupported element type");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value batchVal = getMemRefDimSize(signalType, 0, adaptor.getSignal(),
                                      rewriter, loc);
    Value signalLenVal = getMemRefDimSize(signalType, 1, adaptor.getSignal(),
                                          rewriter, loc);
    Value frameStepVal = i64Const(frameStep);
    Value frameLengthVal = i64Const(frameLength);
    // nFrames = (signalLen - frameLength) / frameStep + 1
    Value nFramesVal = LLVM::AddOp::create(
        rewriter, loc,
        LLVM::SDivOp::create(
            rewriter, loc,
            LLVM::SubOp::create(rewriter, loc, signalLenVal, frameLengthVal),
            frameStepVal),
        i64Const(1));
    Value nFreqsVal = i64Const(nFreqs);
    Value onesidedVal = i64Const(onesided);
    Value dtypeVal = i64Const(dataType);

    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType,
        i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapStftLocal, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr,        signalPtr,    windowPtr,   outputPtr,
        batchVal,        signalLenVal, frameStepVal, frameLengthVal,
        nFramesVal,      nFreqsVal,    onesidedVal, dtypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateStftLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<StftLowering>(converter);
}

} // namespace hip
} // namespace mlir
