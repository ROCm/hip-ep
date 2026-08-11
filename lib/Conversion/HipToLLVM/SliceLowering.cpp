/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.slice(ctx, data, starts, ends, [axes], [steps], output)
//   -> wrap_slice(state, data_ptr, starts_ptr, ends_ptr,
//                 axes_ptr (or null), steps_ptr (or null), out_ptr,
//                 data_shape_ptr, data_rank,
//                 output_shape_ptr, output_rank,
//                 starts_num_elements, axes_num_elements,
//                 steps_num_elements, data_type)
//
// Today the runtime function is a no-op stub that only logs its parameters
// (see lib/Runtime/real/slice.cpp). This lowering exists to keep the IR
// pipeline (bufferize -> hip-to-llvm -> generate-interface) functional even
// when a Slice cannot be folded to tensor.extract_slice by the OnnxToHip
// decompose pattern.
struct SliceOpLowering : public ConvertOpToLLVMPattern<SliceOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(SliceOpLowering)
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto startsType = cast<MemRefType>(op.getStarts().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t hipDtype = getHipdnnDataType(dataType.getElementType());
    if (hipDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value startsPtr =
        extractContiguousMemRefPtr(adaptor.getStarts(), rewriter, loc);
    Value endsPtr =
        extractContiguousMemRefPtr(adaptor.getEnds(), rewriter, loc);
    Value axesPtr = extractOptionalMemRefPtr(adaptor.getAxes(), rewriter, loc);
    Value stepsPtr =
        extractOptionalMemRefPtr(adaptor.getSteps(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);
    auto emitShapeArray = [&](MemRefType type, Value descriptor) -> Value {
      int rank = type.getRank();
      int arrLen = std::max(rank, 1);
      auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (int i = 0; i < rank; ++i) {
        Value dim = getMemRefDimSize(type, i, descriptor, rewriter, loc);
        Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                             rewriter.getI32IntegerAttr(i));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
      }
      return arr;
    };

    Value dataShape = emitShapeArray(dataType, adaptor.getData());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    Value startsNum =
        computeNumElements(startsType, adaptor.getStarts(), rewriter, loc);
    Value axesNum;
    if (op.getAxes()) {
      auto axesT = cast<MemRefType>(op.getAxes().getType());
      axesNum = computeNumElements(axesT, adaptor.getAxes(), rewriter, loc);
    } else {
      axesNum = createI64Const(0);
    }
    Value stepsNum;
    if (op.getSteps()) {
      auto stepsT = cast<MemRefType>(op.getSteps().getType());
      stepsNum = computeNumElements(stepsT, adaptor.getSteps(), rewriter, loc);
    } else {
      stepsNum = createI64Const(0);
    }

    Value dataRank = createI64Const(dataType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value dataTypeVal = createI64Const(hipDtype);

    SmallVector<Type, 16> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, data, starts, ends
        ptrType, ptrType, ptrType,          // axes, steps, output
        ptrType, i64Type,                   // data_shape, data_rank
        ptrType, i64Type,                   // out_shape,  out_rank
        i64Type, i64Type, i64Type,          // starts_num, axes_num,
                                            // steps_num
        i64Type};                           // data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSlice, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 16> args = {statePtr, dataPtr,  startsPtr,  endsPtr,
                                   axesPtr,  stepsPtr, outPtr,     dataShape,
                                   dataRank, outShape, outRank,    startsNum,
                                   axesNum,  stepsNum, dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateSliceLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<SliceOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
