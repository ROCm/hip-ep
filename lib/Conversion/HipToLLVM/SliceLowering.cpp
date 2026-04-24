/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.slice(ctx)
//   ins(data, starts, ends [, axes] [, steps])
//   outs(output)
// -> wrap_slice(state, data, starts, ends, axes (nullable), steps (nullable),
//               output, input_shape_ptr, output_shape_ptr,
//               rank, num_slice_entries, output_num_elements,
//               element_size_bytes, data_type)
//
// `input_shape_ptr` / `output_shape_ptr` are stack-allocated i64 arrays of
// length `rank`, populated from the memref descriptor (static dims become
// compile-time constants; dynamic dims are read from the descriptor).
//===----------------------------------------------------------------------===//

struct SliceOpLowering : public ConvertOpToLLVMPattern<SliceOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value dataPtr = extractMemRefPtr(adaptor.getData(), rewriter, loc);
    Value startsPtr = extractMemRefPtr(adaptor.getStarts(), rewriter, loc);
    Value endsPtr = extractMemRefPtr(adaptor.getEnds(), rewriter, loc);
    Value axesPtr =
        extractOptionalMemRefPtr(adaptor.getAxes(), rewriter, loc);
    Value stepsPtr =
        extractOptionalMemRefPtr(adaptor.getSteps(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Sanity-check types
    auto dataType = cast<MemRefType>(op.getData().getType());
    auto startsType = cast<MemRefType>(op.getStarts().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t rank = dataType.getRank();
    if (outputType.getRank() != rank)
      return op.emitError(
          "hip.slice input and output must have the same rank");

    // Number of slicing entries = number of elements in starts.
    Value numSliceEntries =
        computeNumElements(startsType, adaptor.getStarts(), rewriter, loc);

    // Rank/num_elements metadata
    Value rankVal = createI64Const(rank);
    Value outputNumElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    // Allocate stack buffers for input/output shape (i64 x rank).
    Value arraySize = createI64Const(rank);
    Value inputShapeBuf = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, i64Type, arraySize, /*alignment=*/8);
    Value outputShapeBuf = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, i64Type, arraySize, /*alignment=*/8);

    auto storeDim = [&](Value buf, int64_t dimIdx, Value dimVal) {
      Value idx = createI64Const(dimIdx);
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, buf,
                                       ValueRange{idx});
      LLVM::StoreOp::create(rewriter, loc, dimVal, slot);
    };

    for (int64_t i = 0; i < rank; ++i) {
      Value inDim =
          getMemRefDimSize(dataType, i, adaptor.getData(), rewriter, loc);
      Value outDim =
          getMemRefDimSize(outputType, i, adaptor.getOutput(), rewriter, loc);
      storeDim(inputShapeBuf, i, inDim);
      storeDim(outputShapeBuf, i, outDim);
    }

    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);
    Value dataTypeVal =
        createI64Const(getHipdnnDataType(dataType.getElementType()));

    // int wrap_slice(RuntimeState* state,
    //                void* data, void* starts, void* ends,
    //                void* axes, void* steps, void* output,
    //                const int64_t* input_shape,
    //                const int64_t* output_shape,
    //                int64_t rank, int64_t num_slice_entries,
    //                int64_t output_num_elements,
    //                int64_t element_size_bytes, int64_t data_type)
    SmallVector<Type, 14> paramTypes = {
        ptrType, // state
        ptrType, // data
        ptrType, // starts
        ptrType, // ends
        ptrType, // axes (nullable)
        ptrType, // steps (nullable)
        ptrType, // output
        ptrType, // input_shape
        ptrType, // output_shape
        i64Type, // rank
        i64Type, // num_slice_entries
        i64Type, // output_num_elements
        i64Type, // element_size_bytes
        i64Type  // data_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSlice, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 14> args = {statePtr,          dataPtr,
                                   startsPtr,         endsPtr,
                                   axesPtr,           stepsPtr,
                                   outputPtr,         inputShapeBuf,
                                   outputShapeBuf,    rankVal,
                                   numSliceEntries,   outputNumElements,
                                   elemSizeVal,       dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateSliceLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<SliceOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
