/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.gather_nd(ctx, data, indices, output) {batch_dims}
//   -> wrap_gather_nd(state, data_ptr, indices_ptr, out_ptr,
//                     data_shape_ptr, data_rank,
//                     indices_shape_ptr, indices_rank,
//                     output_shape_ptr, output_rank,
//                     batch_dims, data_type)
struct GatherNDOpLowering : public ConvertOpToLLVMPattern<GatherNDOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherNDOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t hipDtype = getHipdnnDataType(dataType.getElementType());
    if (hipDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
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
    Value indicesShape = emitShapeArray(indicesType, adaptor.getIndices());
    Value outShape = emitShapeArray(outputType, adaptor.getOutput());

    Value dataRank = createI64Const(dataType.getRank());
    Value indicesRank = createI64Const(indicesType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value batchDims = createI64Const(op.getBatchDims());
    Value dataTypeVal = createI64Const(hipDtype);

    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, data, indices, out
        ptrType, i64Type,                   // data_shape, data_rank
        ptrType, i64Type,                   // indices_shape, indices_rank
        ptrType, i64Type,                   // out_shape, out_rank
        i64Type, i64Type};                  // batch_dims, data_type

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGatherND, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr,     dataPtr,     indicesPtr, outPtr,  dataShape, dataRank,
        indicesShape, indicesRank, outShape,   outRank, batchDims, dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGatherNDLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns) {
  patterns.add<GatherNDOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
