/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.split(ctx, data, output) {axis, offset}
struct SplitOpLowering : public ConvertOpToLLVMPattern<SplitOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SplitOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    const int64_t rank = dataType.getRank();
    const int64_t axis = op.getAxis();

    if (rank <= 0 || axis < 0 || axis >= rank)
      return op->emitOpError("axis out of range for input rank");

    Value statePtr = adaptor.getCtx();
    Value dataPtr = extractMemRefPtr(adaptor.getData(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value inputAxisDim =
        getMemRefDimSize(dataType, static_cast<unsigned>(axis), adaptor.getData(),
                         rewriter, loc);
    Value outputAxisDim = getMemRefDimSize(
        outputType, static_cast<unsigned>(axis), adaptor.getOutput(), rewriter,
        loc);

    auto createI64 = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value innerSize = createI64(1);
    for (int64_t d = axis + 1; d < rank; ++d) {
      innerSize =
          LLVM::MulOp::create(rewriter, loc, innerSize,
                              getMemRefDimSize(dataType, static_cast<unsigned>(d),
                                               adaptor.getData(), rewriter, loc));
    }

    Value outputNumElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    Value axisVal = createI64(axis);
    Value offsetVal = createI64(op.getOffset());
    Value elemSizeVal = createI64(elementSizeBytes);

    // int wrap_split(RuntimeState* state, void* data, void* output,
    //                int64_t axis, int64_t offset, int64_t input_axis_dim,
    //                int64_t output_axis_dim, int64_t inner_size,
    //                int64_t output_num_elements,
    //                int64_t element_size_bytes)
    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSplit, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 10> args = {statePtr,       dataPtr, outputPtr, axisVal,
                                   offsetVal,      inputAxisDim, outputAxisDim,
                                   innerSize,      outputNumElements,
                                   elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateSplitLoweringPatterns(const LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns) {
  patterns.add<SplitOpLowering>(converter);
}

} // namespace hip
} // namespace mlir

