/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.slice(ctx, data, starts, ends, [axes], [steps], output)
//   -> wrap_slice_typed(state, data_ptr, starts_ptr, ends_ptr,
//                       axes_ptr (or null), steps_ptr (or null), out_ptr,
//                       data_shape_ptr, data_rank,
//                       output_shape_ptr, output_rank,
//                       starts_num_elements, axes_num_elements,
//                       steps_num_elements, data_type, index_type)
//
// With `static_starts`/`static_steps` the call is instead
//   wrap_slice_static(state, data_ptr, out_ptr, data_shape_ptr,
//                     output_shape_ptr, starts_ptr, steps_ptr, rank,
//                     data_type)
// with the resolved per-axis starts and steps in host stack arrays, so the
// runtime never reads the index tensors back or syncs the stream.
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

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto startsType = cast<MemRefType>(op.getStarts().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t hipDtype = getHipdnnDataType(dataType.getElementType());
    if (hipDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");

    // starts, ends and the optional axes and steps share one element type.
    auto indexTypeOf = [](MemRefType type) -> int64_t {
      if (type.getElementType().isInteger(32))
        return HIPDNN_EP_DATATYPE_INT32;
      if (type.getElementType().isInteger(64))
        return HIPDNN_EP_DATATYPE_INT64;
      return HIPDNN_EP_DATATYPE_UNSUPPORTED;
    };
    int64_t indexType = indexTypeOf(startsType);
    auto sameIndexType = [&](MemRefType type) {
      return indexTypeOf(type) == indexType;
    };
    if (indexType < 0 ||
        !sameIndexType(cast<MemRefType>(op.getEnds().getType())) ||
        (op.getAxes() &&
         !sameIndexType(cast<MemRefType>(op.getAxes().getType()))) ||
        (op.getSteps() &&
         !sameIndexType(cast<MemRefType>(op.getSteps().getType()))))
      return rewriter.notifyMatchFailure(
          op, "slice index tensors must share one i32 or i64 element type");

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

    auto staticStarts = op.getStaticStartsAttr();
    auto staticSteps = op.getStaticStepsAttr();
    if (staticStarts && staticSteps) {
      int64_t rank = dataType.getRank();
      if (static_cast<int64_t>(staticStarts.size()) != rank ||
          static_cast<int64_t>(staticSteps.size()) != rank)
        return rewriter.notifyMatchFailure(op, "static slice params rank");
      auto emitConstArray = [&](ArrayRef<int64_t> values) -> Value {
        auto arrType = LLVM::LLVMArrayType::get(i64Type, values.size());
        Value arr =
            LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
        for (auto [i, v] : llvm::enumerate(values)) {
          Value idx = LLVM::ConstantOp::create(
              rewriter, loc, i32Type,
              rewriter.getI32IntegerAttr(static_cast<int32_t>(i)));
          Value elemPtr =
              LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
          LLVM::StoreOp::create(rewriter, loc, createI64Const(v), elemPtr);
        }
        return arr;
      };
      SmallVector<Type, 9> staticTypes = {ptrType, ptrType, ptrType,
                                          ptrType, ptrType, ptrType,
                                          ptrType, i64Type, i64Type};
      FailureOr<LLVM::LLVMFuncOp> staticFn = LLVM::lookupOrCreateFn(
          rewriter, module, kWrapSliceStatic, staticTypes, i32Type);
      if (failed(staticFn))
        return failure();
      SmallVector<Value, 9> staticArgs = {
          statePtr,
          dataPtr,
          outPtr,
          dataShape,
          outShape,
          emitConstArray(staticStarts.asArrayRef()),
          emitConstArray(staticSteps.asArrayRef()),
          createI64Const(rank),
          createI64Const(hipDtype)};
      LLVM::CallOp::create(rewriter, loc, *staticFn, staticArgs);
      rewriter.eraseOp(op);
      return success();
    }

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
    Value indexTypeVal = createI64Const(indexType);

    SmallVector<Type, 17> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, data, starts, ends
        ptrType, ptrType, ptrType,          // axes, steps, output
        ptrType, i64Type,                   // data_shape, data_rank
        ptrType, i64Type,                   // out_shape,  out_rank
        i64Type, i64Type, i64Type,          // starts_num, axes_num,
                                            // steps_num
        i64Type, i64Type};                  // data_type, index_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSlice, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 16> args = {
        statePtr, dataPtr,   startsPtr,   endsPtr,     axesPtr, stepsPtr,
        outPtr,   dataShape, dataRank,    outShape,    outRank, startsNum,
        axesNum,  stepsNum,  dataTypeVal, indexTypeVal};

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
