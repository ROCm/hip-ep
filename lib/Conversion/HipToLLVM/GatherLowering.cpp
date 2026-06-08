/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.gather(handle, indices, table, output)
struct GatherOpLowering : public ConvertOpToLLVMPattern<GatherOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Compute data_num_elements
    auto dataType = cast<MemRefType>(op.getData().getType());
    Value dataNumElementsVal =
        computeNumElements(dataType, adaptor.getData(), rewriter, loc);

    // Compute indices_num_elements
    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    Value indicesNumElementsVal =
        computeNumElements(indicesType, adaptor.getIndices(), rewriter, loc);

    // Compute output_num_elements
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value outputNumElementsVal =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    // element_size_bytes
    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // axis attribute
    int64_t axisAttr = op.getAxis();
    Value axisVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(axisAttr));

    // axis_size = data.shape[axis]; inner_size = product(data.shape[axis+1:]).
    int64_t dataRank = dataType.getRank();
    if (axisAttr < 0)
      axisAttr += dataRank;
    Value axisSizeVal =
        getMemRefDimSize(dataType, static_cast<unsigned>(axisAttr),
                         adaptor.getData(), rewriter, loc);
    Value innerSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
    for (int64_t d : llvm::seq<int64_t>(axisAttr + 1, dataRank)) {
      Value ds = getMemRefDimSize(dataType, static_cast<unsigned>(d),
                                  adaptor.getData(), rewriter, loc);
      innerSizeVal = LLVM::MulOp::create(rewriter, loc, innerSizeVal, ds);
    }

    // int wrap_gather(RuntimeState* state, void* data, void* indices,
    //                 void* output, int64_t axis, int64_t data_num_elements,
    //                 int64_t indices_num_elements,
    //                 int64_t output_num_elements,
    //                 int64_t axis_size, int64_t inner_size,
    //                 int64_t element_size_bytes)
    SmallVector<Type, 11> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGather, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,
                               dataPtr,
                               indicesPtr,
                               outputPtr,
                               axisVal,
                               dataNumElementsVal,
                               indicesNumElementsVal,
                               outputNumElementsVal,
                               axisSizeVal,
                               innerSizeVal,
                               elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGatherLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<GatherOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
