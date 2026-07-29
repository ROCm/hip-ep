/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.readback_scalar(ctx, scalar : memref<T>) : T
//   -> %slot = llvm.alloca : !llvm.ptr                        // host scalar
//   slot
//      llvm.call @hipdnn_ep_readback_scalar(state, %slot,     // D2H + stream
//                                           scalar_ptr, nbytes)//   sync
//      %v = llvm.load %slot : T                               // host value
//
// General-purpose host-readback for a single device scalar of arbitrary element
// type (i64 / f32 / f16 / i32 / ...). The runtime helper synchronizes the
// stream (so the producing kernel has finished) and copies `nbytes` from the
// device buffer into the stack slot; we then load the value back. The result
// feeds host-side scalar arithmetic — e.g. the trip-count computation of a
// data-dependent onnx.Range whose limit/start/delta are GPU-computed.
//
// This differs from hip.readback_dim, which is specialised to a non-negative
// i32 extent and always returns `index`: readback_scalar preserves the
// operand's element type and sign.
struct ReadbackScalarOpLowering
    : public ConvertOpToLLVMPattern<ReadbackScalarOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ReadbackScalarOpLowering)
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReadbackScalarOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();

    Type resultType = op.getValue().getType();
    // The host slot stores the same bit pattern we copy from the device.
    // `index` lowers to i64; everything else keeps its own LLVM type.
    Type slotType = isa<IndexType>(resultType)
                        ? i64Type
                        : typeConverter->convertType(resultType);
    if (!slotType)
      return rewriter.notifyMatchFailure(op,
                                         "unsupported readback element type");

    unsigned bits;
    if (isa<IndexType>(resultType))
      bits = 64;
    else if (auto intTy = dyn_cast<IntegerType>(resultType))
      bits = intTy.getWidth();
    else if (auto fTy = dyn_cast<FloatType>(resultType))
      bits = fTy.getWidth();
    else
      return rewriter.notifyMatchFailure(op,
                                         "unsupported readback element type");
    int64_t numBytes = (bits + 7) / 8;

    Value statePtr = adaptor.getCtx();
    // The scalar may be a `memref.view` into the host-scratch buffer with a
    // non-zero descriptor offset (MaterializeHostScalars packs several scalars
    // into one allocation). Use the offset-aware data-ptr helper, NOT
    // extractContiguousMemRefPtr (which returns the base alignedPtr and would
    // read the wrong scratch slot — yielding garbage / cross-scalar aliasing).
    auto scalarMemRefTy = dyn_cast<MemRefType>(op.getScalar().getType());
    if (!scalarMemRefTy)
      return rewriter.notifyMatchFailure(op, "scalar operand must be a memref");
    Value scalarPtr = extractMemRefDataPtr(adaptor.getScalar(), scalarMemRefTy,
                                           typeConverter, rewriter, loc);
    if (!scalarPtr)
      return rewriter.notifyMatchFailure(op, "failed to compute scalar ptr");

    // Host stack slot for the read-back value.
    Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                         rewriter.getI64IntegerAttr(1));
    Value hostSlot =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, slotType, one,
                               /*alignment=*/static_cast<unsigned>(numBytes));

    // void hipdnn_ep_readback_scalar(RuntimeState*, void* host_dst,
    //                                const void* device_scalar, int64_t nbytes)
    Type voidTy = LLVM::LLVMVoidType::get(rewriter.getContext());
    SmallVector<Type, 4> paramTypes = {ptrType, ptrType, ptrType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipReadbackScalar, paramTypes, voidTy);
    if (failed(funcOp))
      return failure();

    Value nBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(numBytes));
    LLVM::CallOp::create(rewriter, loc, *funcOp,
                         ValueRange{statePtr, hostSlot, scalarPtr, nBytesVal});

    Value loaded = LLVM::LoadOp::create(rewriter, loc, slotType, hostSlot);
    // `index` results: the load produced i64, which is the index ABI.
    rewriter.replaceOp(op, loaded);
    return success();
  }
};

} // namespace

void populateReadbackScalarLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns) {
  patterns.add<ReadbackScalarOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
