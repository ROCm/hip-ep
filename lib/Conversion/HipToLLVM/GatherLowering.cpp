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
    Value dataPtr = extractMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr = extractMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

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
    int64_t axis = op.getAxis();
    int64_t dataRank = dataType.getRank();
    if (axis < 0) axis += dataRank;
    Value axisVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(axis));

    // pre_axis_size = product of data dims [0..axis-1] at runtime.
    auto buildDimProduct = [&](int64_t lo, int64_t hi) -> Value {
      Value prod = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(1));
      MemRefDescriptor desc(adaptor.getData());
      for (int64_t i = lo; i < hi; ++i) {
        Value d;
        if (dataType.isDynamicDim(i))
          d = desc.size(rewriter, loc, i);
        else
          d = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                        rewriter.getI64IntegerAttr(
                                            dataType.getDimSize(i)));
        prod = LLVM::MulOp::create(rewriter, loc, prod, d);
      }
      return prod;
    };
    Value preAxisVal = buildDimProduct(0, axis);

    // int wrap_gather(RuntimeState* state, void* data, void* indices,
    //                 void* output, int64_t axis, int64_t data_num,
    //                 int64_t indices_num, int64_t output_num,
    //                 int64_t element_size_bytes, int64_t pre_axis_size)
    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                         i64Type, i64Type, i64Type, i64Type,
                                         i64Type, i64Type};

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
                               elemSizeVal,
                               preAxisVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateGatherLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GatherOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
