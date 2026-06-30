/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// Shared body for hip.memcpy_{h2d,d2h}_async. Both lower to a state-based
// wrapper that takes the RuntimeState* (ctx) and resolves the stream itself,
// mirroring wrap_hipMemcpyAsync:
//   wrap_hipMemcpy{H2D,D2H}Async(state, dst, src, size_bytes) : i32
//
// Before: hip.memcpy_d2h_async(%ctx, %hostDst, %devSrc : ...)
// After:  llvm.call @wrap_hipMemcpyD2HAsync(%state, %dstPtr, %srcPtr, %size)
template <typename OpTy>
static LogicalResult lowerMemcpy(OpTy op, Value dstDesc, MemRefType dstTy,
                                 Value srcDesc, MemRefType srcTy, Value statePtr,
                                 const char *wrapSym,
                                 const TypeConverter *typeConverter,
                                 ConversionPatternRewriter &rewriter) {
  Location loc = op.getLoc();
  ModuleOp module = op->template getParentOfType<ModuleOp>();
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();

  // Offset-aware data pointers: a memcpy operand may carry a non-zero
  // descriptor offset (e.g. a memref.view/subview into a larger buffer), so we
  // must honor the offset (extractMemRefDataPtr), not just alignedPtr.
  Value dstPtr =
      extractMemRefDataPtr(dstDesc, dstTy, typeConverter, rewriter, loc);
  Value srcPtr =
      extractMemRefDataPtr(srcDesc, srcTy, typeConverter, rewriter, loc);
  if (!dstPtr || !srcPtr)
    return rewriter.notifyMatchFailure(op, "failed to compute data pointers");

  // Byte count = numElements * bytes-per-element (src == dst per the verifier).
  // computeNumElements handles dynamic dims off the descriptor; per-element
  // bytes use the codebase idiom `(bitwidth + 7) / 8` (transfer operands are
  // int/float, never index).
  Value numElems = computeNumElements(dstTy, dstDesc, rewriter, loc);
  int64_t elemBytes = (dstTy.getElementTypeBitWidth() + 7) / 8;
  Value elemBytesVal = LLVM::ConstantOp::create(
      rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elemBytes));
  Value size = LLVM::MulOp::create(rewriter, loc, numElems, elemBytesVal);

  FailureOr<LLVM::LLVMFuncOp> memcpyFn = LLVM::lookupOrCreateFn(
      rewriter, module, wrapSym, {ptrType, ptrType, ptrType, i64Type}, i32Type);
  if (failed(memcpyFn))
    return failure();

  LLVM::CallOp::create(rewriter, loc, *memcpyFn,
                       ValueRange{statePtr, dstPtr, srcPtr, size});
  rewriter.eraseOp(op);
  return success();
}

struct MemcpyH2DAsyncOpLowering
    : public ConvertOpToLLVMPattern<MemcpyH2DAsyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MemcpyH2DAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstTy = cast<MemRefType>(op.getDst().getType());
    auto srcTy = cast<MemRefType>(op.getSrc().getType());
    return lowerMemcpy(op, adaptor.getDst(), dstTy, adaptor.getSrc(), srcTy,
                       adaptor.getCtx(), kWrapHipMemcpyH2DAsync,
                       getTypeConverter(), rewriter);
  }
};

struct MemcpyD2HAsyncOpLowering
    : public ConvertOpToLLVMPattern<MemcpyD2HAsyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(MemcpyD2HAsyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto dstTy = cast<MemRefType>(op.getDst().getType());
    auto srcTy = cast<MemRefType>(op.getSrc().getType());
    return lowerMemcpy(op, adaptor.getDst(), dstTy, adaptor.getSrc(), srcTy,
                       adaptor.getCtx(), kWrapHipMemcpyD2HAsync,
                       getTypeConverter(), rewriter);
  }
};

// hip.stream_sync(%ctx) -> wrap_hipStreamSynchronize(state). Plain blocking
// sync (NOT hipdnn_ep_stream_sync, which also records PERF events). The wrapper
// takes ctx and resolves the stream itself.
//
// Before: hip.stream_sync(%ctx)
// After:  llvm.call @wrap_hipStreamSynchronize(%state)
struct StreamSyncOpLowering : public ConvertOpToLLVMPattern<StreamSyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(StreamSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();

    FailureOr<LLVM::LLVMFuncOp> syncFn = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipStreamSynchronize, {ptrType}, i32Type);
    if (failed(syncFn))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *syncFn, ValueRange{adaptor.getCtx()});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMemcpyLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<MemcpyH2DAsyncOpLowering, MemcpyD2HAsyncOpLowering,
               StreamSyncOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
