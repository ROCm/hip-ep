/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.readback_dim(ctx, scalar) : index
//   -> %c = hipdnn_ep_readback_i32(state, scalar_ptr)   // D2H + stream sync
//      zext %c : i32 to i64                              // index ABI
//
// Generic host-readback primitive for data-dependent dynamic shapes (NOT
// specific to any one operator): `scalar` is a device i32 buffer that some
// kernel computed (e.g. a count / valid-length / selected-element tally). The
// runtime helper synchronizes the stream (so the producing kernel has
// finished) and copies the 4-byte scalar back to the host. The result feeds
// host-side index arithmetic (tensor.extract_slice size, tensor.empty
// dynsize, hip.alloc_output extent, ...).
struct ReadbackDimOpLowering : public ConvertOpToLLVMPattern<ReadbackDimOp> {
  ReadbackDimOpLowering(const LLVMTypeConverter &converter,
                        const DenseMap<Operation *, Value> &hostScalars)
      : ConvertOpToLLVMPattern<ReadbackDimOp>(converter),
        hostScalars(hostScalars) {}

  DenseMap<Operation *, Value> hostScalars;

  LogicalResult
  matchAndRewrite(ReadbackDimOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    if (Value original = hostScalars.lookup(op)) {
      Value value = rewriter.getRemappedValue(original);
      if (!value)
        return rewriter.notifyMatchFailure(op, "cannot convert host extent");
      rewriter.replaceOpWithNewOp<LLVM::ZExtOp>(op, i64Type, value);
      return success();
    }

    Value statePtr = adaptor.getCtx();
    // The scalar may be a `memref.view` into the host-scratch buffer with a
    // non-zero descriptor offset (MaterializeHostScalars packs several small
    // integer scalars into one allocation). Use the offset-aware data-ptr
    // helper, NOT extractContiguousMemRefPtr (which returns the base alignedPtr
    // and would read the wrong scratch slot).
    Value scalarPtr;
    if (auto scalarMemRefTy = dyn_cast<MemRefType>(op.getScalar().getType()))
      scalarPtr = extractMemRefDataPtr(adaptor.getScalar(), scalarMemRefTy,
                                       typeConverter, rewriter, loc);
    else
      scalarPtr =
          extractContiguousMemRefPtr(adaptor.getScalar(), rewriter, loc);
    if (!scalarPtr)
      return rewriter.notifyMatchFailure(op, "failed to compute scalar ptr");

    // int32_t hipdnn_ep_readback_i32(RuntimeState* state, const void* scalar)
    SmallVector<Type, 2> paramTypes = {ptrType, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipReadbackI32, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    auto call = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                     ValueRange{statePtr, scalarPtr});
    // The readback scalar is a non-negative extent; zero-extend to the i64
    // that `index` lowers to.
    Value widened =
        LLVM::ZExtOp::create(rewriter, loc, i64Type, call.getResult());
    rewriter.replaceOp(op, widened);
    return success();
  }
};

} // namespace

void populateReadbackDimLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns,
    const DenseMap<Operation *, Value> &hostScalars) {
  patterns.add<ReadbackDimOpLowering>(converter, hostScalars);
}

} // namespace hip
} // namespace mlir
