/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.size(ctx, x, y)
//   -> wrap_size(state, output_ptr, num_elements)
//
// `num_elements` is computed in IR via `computeNumElements`, which uses
// compile-time constants for static dims and reads the MemRef descriptor
// `sizes[]` for dynamic dims. The runtime stores that i64 into the
// rank-0 i64 output buffer (a single GPU memcpy of 8 bytes).
//
// We do NOT pass the input pointer to the runtime: Size's value depends
// only on the shape, not on the bytes, so the runtime ABI needs only the
// element count and output pointer.
struct SizeOpLowering : public ConvertOpToLLVMPattern<SizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SizeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = dyn_cast<MemRefType>(op.getX().getType());
    auto outputType = dyn_cast<MemRefType>(op.getY().getType());
    if (!inputType || !outputType)
      return rewriter.notifyMatchFailure(
          op, "hip.size lowering expects ranked memref operands");
    if (outputType.getRank() != 0)
      return rewriter.notifyMatchFailure(
          op, "hip.size result memref must be rank-0");
    if (!outputType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(
          op, "hip.size result element type must be i64");

    Value statePtr = adaptor.getCtx();
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);

    // prod(input.shape) -- handles any mix of static and dynamic dims.
    // For a fully static input this is a single LLVM constant after
    // canonicalisation; for dynamic dims it's a chain of llvm.mul over
    // descriptor.sizes[i].
    Value numElements =
        computeNumElements(inputType, adaptor.getX(), rewriter, loc);

    // int wrap_size(RuntimeState* state, void* output, int64_t num_elements)
    SmallVector<Type, 3> paramTypes = {ptrType, ptrType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSize, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 3> args = {statePtr, outputPtr, numElements};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateSizeLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<SizeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
