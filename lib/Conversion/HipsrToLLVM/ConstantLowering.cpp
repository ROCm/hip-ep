/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ConstantLowering.cpp - Lower hipsr.constant to LLVM ---------------===//
//
// Mirrors the hip.get_constant lowering (HipToLLVM/MemoryLowering.cpp): an
// externalized hipsr.constant becomes a @wrap_get_global(ctx, offset, size)
// runtime call whose returned device pointer is wrapped in a memref
// descriptor. hipsr.constant carries no operands, so ctx is the enclosing
// llvm.func's arg 0 (the runtime state pointer, per the EP ABI -- same
// convention as MemRefCopyOpLowering) and offset/size are read off the op.
// Registered via HipsrDialect's ConvertToLLVMPatternInterface, so it is a
// stateless pattern discovered by the standard LLVM conversion driver.
//
//===----------------------------------------------------------------------===//

#include "HipsrToLLVMUtils.h"

namespace mlir {
namespace hipsr {
namespace {

struct ConstantLowering : public ConvertOpToLLVMPattern<ConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.isExternalized()) {
      return lowerExternalized(op, rewriter);
    }
    if (!op.shouldExternalize()) {
      return lowerInline(op, rewriter);
    }
    // shouldExternalize() == true but no offset yet: the externalization pass
    // must run first.
    op.emitError(
        "hipsr.constant reached LLVM lowering without externalization; "
        "run -hipsr-externalize-constants first");
    return failure();
  }

private:
  // offset/size were stamped on the op by hipsr-externalize-constants; wrap the
  // AS 0 pointer @wrap_get_global returns in a memref descriptor.
  LogicalResult lowerExternalized(ConstantOp op,
                                  ConversionPatternRewriter &rewriter) const {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type i64Type = IntegerType::get(ctx, 64);

    auto memRefType = dyn_cast<MemRefType>(op.getResult().getType());
    if (!memRefType) {
      op.emitError("hipsr.constant: externalized constant must have a memref "
                   "result to lower to a runtime call");
      return failure();
    }

    // ctx = the enclosing function's arg 0. Bail until the parent is llvm.func
    // so the arg is already an !llvm.ptr (the conversion driver retries).
    auto llvmFn = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFn) {
      return rewriter.notifyMatchFailure(op, "expected enclosing llvm.func");
    }
    if (llvmFn.getNumArguments() == 0) {
      op.emitError("hipsr.constant: enclosing function has no arguments; "
                   "expected the runtime context at arg 0");
      return failure();
    }
    Value ctxArg = llvmFn.getArgument(0);

    Value offsetVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getOffsetAttr().getInt()));
    Value sizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getSizeAttr().getInt()));

    SmallVector<Type, 3> paramTypes = {ptrType, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipsrWrapGetGlobal, paramTypes, ptrType);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 3> args = {ctxArg, offsetVal, sizeVal};
    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // The runtime always returns a generic pointer (AS 0). Cast to the memref's
    // address space (e.g. AS 1 = AMDGPU global memory) if needed.
    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace)) {
      return failure();
    }

    Value dataPtr = callOp.getResult();
    if (*addrSpace != 0) {
      dataPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc, LLVM::LLVMPointerType::get(ctx, *addrSpace), dataPtr);
    }

    // Row-major sizes / strides for a static, identity-layout memref.
    auto shape = memRefType.getShape();
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    for (int64_t dim : shape) {
      sizes.push_back(LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(dim)));
    }
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      strides.insert(strides.begin(), LLVM::ConstantOp::create(
                                          rewriter, loc, i64Type,
                                          rewriter.getI64IntegerAttr(stride)));
      stride *= shape[i];
    }

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, dataPtr, dataPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }

  // Inline (disableExternalize): a compile-time constant. Only the ranked
  // tensor result is lowered here (to arith.constant); a device memref result
  // in this state is not yet supported (needs a memref.global) and is a
  // follow-up.
  LogicalResult lowerInline(ConstantOp op,
                            ConversionPatternRewriter &rewriter) const {
    auto tensorType = dyn_cast<RankedTensorType>(op.getResult().getType());
    auto dense = dyn_cast_or_null<DenseElementsAttr>(op.getValueAttr());
    if (!tensorType || !dense) {
      op.emitError("hipsr.constant: inline (disableExternalize) lowering only "
                   "supports a ranked-tensor result with a dense value; memref "
                   "results are a follow-up");
      return failure();
    }
    rewriter.replaceOpWithNewOp<arith::ConstantOp>(op, tensorType, dense);
    return success();
  }
};

} // namespace

void populateHipsrConstantLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<ConstantLowering>(converter);
}

} // namespace hipsr
} // namespace mlir
