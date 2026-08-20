/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.slice(ctx, data, valid, starts[R], steps[R], extents[R], output)
//   -> wrap_slice(state, data, output, data_shape, output_shape,
//                 starts, steps, extents, rank, dtype, valid)
//
// Every control array is host SSA materialized by conversion. The runtime does
// no D2H and no ONNX normalization.
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
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t rank = dataType.getRank();
    if (outputType.getRank() != rank ||
        static_cast<int64_t>(op.getStarts().size()) != rank ||
        static_cast<int64_t>(op.getSteps().size()) != rank ||
        static_cast<int64_t>(op.getExtents().size()) != rank)
      return rewriter.notifyMatchFailure(op, "invalid exact-rank contract");

    int64_t hipDtype = getHipdnnDataType(dataType.getElementType());
    if (hipDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");

    auto i64Constant = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    Value one = i64Constant(1);
    Value arrayCount = i64Constant(std::max<int64_t>(rank, 1));

    auto allocateI64Array = [&]() {
      return LLVM::AllocaOp::create(rewriter, loc, ptrType, i64Type, arrayCount,
                                    /*alignment=*/8)
          .getResult();
    };
    auto storeArrayElement = [&](Value array, int64_t index, Value value) {
      Value indexValue = i64Constant(index);
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, array,
                                       indexValue);
      LLVM::StoreOp::create(rewriter, loc, value, slot);
    };
    auto emitShapeArray = [&](MemRefType type, Value descriptor) {
      Value array = allocateI64Array();
      for (int64_t dim : llvm::seq<int64_t>(rank)) {
        storeArrayElement(
            array, dim, getMemRefDimSize(type, dim, descriptor, rewriter, loc));
      }
      return array;
    };

    Value dataShape = emitShapeArray(dataType, adaptor.getData());
    Value outputShape = emitShapeArray(outputType, adaptor.getOutput());
    Value starts = allocateI64Array();
    Value steps = allocateI64Array();
    Value extents = allocateI64Array();
    for (int64_t dim : llvm::seq<int64_t>(rank)) {
      storeArrayElement(starts, dim, adaptor.getStarts()[dim]);
      storeArrayElement(steps, dim, adaptor.getSteps()[dim]);
      storeArrayElement(extents, dim, adaptor.getExtents()[dim]);
    }

    Value dataPtr = extractMemRefDataPtr(adaptor.getData(), dataType,
                                         typeConverter, rewriter, loc);
    Value outputPtr = extractMemRefDataPtr(adaptor.getOutput(), outputType,
                                           typeConverter, rewriter, loc);
    if (!dataPtr || !outputPtr)
      return rewriter.notifyMatchFailure(op, "failed to compute data pointers");

    SmallVector<Type> parameterTypes = {ptrType,
                                        ptrType,
                                        ptrType,
                                        ptrType,
                                        ptrType,
                                        ptrType,
                                        ptrType,
                                        ptrType,
                                        i64Type,
                                        i64Type,
                                        rewriter.getI1Type()};
    FailureOr<LLVM::LLVMFuncOp> function = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSlice, parameterTypes, i32Type);
    if (failed(function))
      return failure();

    LLVM::CallOp::create(
        rewriter, loc, *function,
        ValueRange{adaptor.getCtx(), dataPtr, outputPtr, dataShape, outputShape,
                   starts, steps, extents, i64Constant(rank),
                   i64Constant(hipDtype), adaptor.getParamsValid()});
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
