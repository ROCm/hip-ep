/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.range(ctx, start, limit, delta, output)
struct RangeOpLowering : public ConvertOpToLLVMPattern<RangeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RangeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value startPtr = extractMemRefPtr(adaptor.getStart(), rewriter, loc);
    Value limitPtr = extractMemRefPtr(adaptor.getLimit(), rewriter, loc);
    Value deltaPtr = extractMemRefPtr(adaptor.getDelta(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Get element size in bytes
    auto startType = cast<MemRefType>(op.getStart().getType());
    unsigned elementSizeBytes =
        startType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // int wrap_range(RuntimeState* state, void* start, void* limit,
    //                void* delta, void* output, int64_t element_size_bytes)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, ptrType, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRange, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr, startPtr,  limitPtr,
                               deltaPtr, outputPtr, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateRangeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<RangeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
