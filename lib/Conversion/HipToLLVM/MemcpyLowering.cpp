/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "hip/Dialect/IR/HipShapeUtils.h"

namespace mlir {
namespace hip {
namespace {

// Resolve the runtime stream pointer from the RuntimeState* (ctx). The async
// memcpy / stream-sync wrappers take the STREAM, not the state, so every
// lowering below first calls hipdnn_ep_state_get_stream(state).
static Value getStream(Value statePtr, ModuleOp module,
                       ConversionPatternRewriter &rewriter, Location loc) {
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  FailureOr<LLVM::LLVMFuncOp> streamFn =
      LLVM::lookupOrCreateFn(rewriter, module, kHipGetStream, {ptrType},
                             ptrType);
  if (failed(streamFn))
    return Value();
  return LLVM::CallOp::create(rewriter, loc, *streamFn, ValueRange{statePtr})
      .getResult();
}

// Total byte count of a (static-or-dynamic) memref as an i64 Value:
// numElements(descriptor) * elemBytes.
static Value byteCount(MemRefType type, Value descriptor,
                       ConversionPatternRewriter &rewriter, Location loc) {
  Type i64Type = rewriter.getI64Type();
  int64_t elemBytes = elementByteSize(type.getElementType());
  Value numElems = computeNumElements(type, descriptor, rewriter, loc);
  Value elemBytesVal = LLVM::ConstantOp::create(
      rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elemBytes));
  return LLVM::MulOp::create(rewriter, loc, numElems, elemBytesVal);
}

// Shared body for the two async memcpy ops. They differ only in the wrapper
// symbol (H2D vs D2H) and operand roles; the wrapper ABI is identical:
//   wrap_hipMemcpy{H2D,D2H}(dst_ptr, src_ptr, size_bytes, stream) : i32
//
// Before:
//   hip.memcpy_d2h_async(%ctx, %hostDst, %devSrc
//       : memref<8xi64, #hip.mem<host>>, memref<8xi64>)
// After:
//   %stream = llvm.call @hipdnn_ep_state_get_stream(%state)
//   %sz     = llvm.mul %nelems, %elemBytes
//   llvm.call @wrap_hipMemcpyD2H(%dstPtr, %srcPtr, %sz, %stream)
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

  Value stream = getStream(statePtr, module, rewriter, loc);
  if (!stream)
    return failure();

  // Byte count derives from the dst memref (verifier guarantees src == dst
  // element type + count).
  Value size = byteCount(dstTy, dstDesc, rewriter, loc);

  FailureOr<LLVM::LLVMFuncOp> memcpyFn = LLVM::lookupOrCreateFn(
      rewriter, module, wrapSym, {ptrType, ptrType, i64Type, ptrType}, i32Type);
  if (failed(memcpyFn))
    return failure();

  LLVM::CallOp::create(rewriter, loc, *memcpyFn,
                       ValueRange{dstPtr, srcPtr, size, stream});
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
                       adaptor.getCtx(), kWrapHipMemcpyH2D, getTypeConverter(),
                       rewriter);
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
                       adaptor.getCtx(), kWrapHipMemcpyD2H, getTypeConverter(),
                       rewriter);
  }
};

// hip.stream_sync(%ctx) -> wrap_hipStreamSynchronize(stream). A plain blocking
// stream sync (NOT hipdnn_ep_stream_sync, which also flushes error state).
//
// Before: hip.stream_sync(%ctx)
// After:  %stream = llvm.call @hipdnn_ep_state_get_stream(%state)
//         llvm.call @wrap_hipStreamSynchronize(%stream)
struct StreamSyncOpLowering : public ConvertOpToLLVMPattern<StreamSyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(StreamSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();

    Value stream = getStream(adaptor.getCtx(), module, rewriter, loc);
    if (!stream)
      return failure();

    FailureOr<LLVM::LLVMFuncOp> syncFn = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipStreamSynchronize, {ptrType}, i32Type);
    if (failed(syncFn))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *syncFn, ValueRange{stream});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTransferLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns) {
  patterns.add<MemcpyH2DAsyncOpLowering, MemcpyD2HAsyncOpLowering,
               StreamSyncOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
