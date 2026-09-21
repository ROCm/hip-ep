/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.slice(ctx, data, starts, ends, [axes], [steps], output, host attrs)
//   -> wrap_slice(state, data_ptr, starts_device, starts_host,
//                 ends_device, ends_host, axes_device, axes_host,
//                 steps_device, steps_host, out_ptr,
//                 data_shape_ptr, data_rank,
//                 output_shape_ptr, output_rank,
//                 starts_num_elements, axes_num_elements,
//                 steps_num_elements, data_type)
//
// Constant arrays are materialized in host stack storage. Missing attributes
// use null host pointers, selecting the runtime's device-readback fallback.
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
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value one = createI64Const(1);
    auto emitI64Array = [&](mlir::DenseI64ArrayAttr attr) -> Value {
      if (!attr)
        return nullPtr;
      int64_t count = static_cast<int64_t>(attr.size());
      auto arrType =
          LLVM::LLVMArrayType::get(i64Type, std::max(count, int64_t{1}));
      Value arr =
          LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
      for (auto [i, value] : llvm::enumerate(attr.asArrayRef())) {
        Value idx = LLVM::ConstantOp::create(
            rewriter, loc, i32Type,
            rewriter.getI32IntegerAttr(static_cast<int32_t>(i)));
        Value elemPtr =
            LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, arr, idx);
        LLVM::StoreOp::create(rewriter, loc, createI64Const(value), elemPtr);
      }
      return arr;
    };
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

    auto startsAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>("starts_attr");
    auto endsAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>("ends_attr");
    auto axesAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>("axes_attr");
    auto stepsAttr = op->getAttrOfType<mlir::DenseI64ArrayAttr>("steps_attr");
    Value startsHost = emitI64Array(startsAttr);
    Value endsHost = emitI64Array(endsAttr);
    Value axesHost = emitI64Array(axesAttr);
    Value stepsHost = emitI64Array(stepsAttr);

    Value startsNum =
        startsAttr ? createI64Const(static_cast<int64_t>(startsAttr.size()))
                   : computeNumElements(startsType, adaptor.getStarts(),
                                        rewriter, loc);
    Value axesNum;
    if (axesAttr) {
      axesNum = createI64Const(static_cast<int64_t>(axesAttr.size()));
    } else if (op.getAxes()) {
      auto axesT = cast<MemRefType>(op.getAxes().getType());
      axesNum = computeNumElements(axesT, adaptor.getAxes(), rewriter, loc);
    } else {
      axesNum = createI64Const(0);
    }
    Value stepsNum;
    if (stepsAttr) {
      stepsNum = createI64Const(static_cast<int64_t>(stepsAttr.size()));
    } else if (op.getSteps()) {
      auto stepsT = cast<MemRefType>(op.getSteps().getType());
      stepsNum = computeNumElements(stepsT, adaptor.getSteps(), rewriter, loc);
    } else {
      stepsNum = createI64Const(0);
    }

    Value dataRank = createI64Const(dataType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value dataTypeVal = createI64Const(hipDtype);

    SmallVector<Type, 19> paramTypes = {
        ptrType, ptrType,                   // state, data
        ptrType, ptrType, ptrType, ptrType, // starts/ends device + host
        ptrType, ptrType, ptrType, ptrType, // axes/steps device + host
        ptrType,                            // output
        ptrType, i64Type,                   // data_shape, data_rank
        ptrType, i64Type,                   // out_shape,  out_rank
        i64Type, i64Type, i64Type,          // starts_num, axes_num,
                                            // steps_num
        i64Type};                           // data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSlice, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 19> args = {statePtr, dataPtr,   startsPtr,  startsHost,
                                   endsPtr,  endsHost,  axesPtr,    axesHost,
                                   stepsPtr, stepsHost, outPtr,     dataShape,
                                   dataRank, outShape,  outRank,    startsNum,
                                   axesNum,  stepsNum,  dataTypeVal};

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
