/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.cumsum(ctx, x, [axis], y, axis_attr, exclusive, reverse)
//   -> wrap_cumsum(state, x_ptr, axis_device_ptr, y_ptr,
//                  x_shape_ptr, x_rank,
//                  num_elements, data_type, axis_dtype, axis_host,
//                  exclusive, reverse)
//
// A compile-time axis has a null device pointer and its value in `axis_host`.
// A dynamic axis retains its GPU pointer and dtype for the runtime D2H
// fallback.
struct CumSumOpLowering : public ConvertOpToLLVMPattern<CumSumOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CumSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value xPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
    Value axisPtr =
        adaptor.getAxis()
            ? extractContiguousMemRefPtr(adaptor.getAxis(), rewriter, loc)
            : LLVM::ZeroOp::create(rewriter, loc, ptrType);
    Value yPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);

    auto xType = cast<MemRefType>(op.getX().getType());
    int64_t dataType = getHipdnnDataType(xType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");
    int64_t axisDtype = HIPDNN_EP_DATATYPE_INT64;
    if (op.getAxis()) {
      auto axisType = cast<MemRefType>(op.getAxis().getType());
      axisDtype = getHipdnnDataType(axisType.getElementType());
      if (axisDtype < 0)
        return rewriter.notifyMatchFailure(op, "unsupported axis element type");
    }

    Value numElements =
        computeNumElements(xType, adaptor.getX(), rewriter, loc);

    // Stack-alloc x_shape array (max(rank,1) so rank-0 still has a buffer).
    Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                         rewriter.getI64IntegerAttr(1));
    int rank = xType.getRank();
    int arrLen = std::max(rank, 1);
    auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
    Value shapeArr =
        LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
    for (int i = 0; i < rank; ++i) {
      Value dim = getMemRefDimSize(xType, i, adaptor.getX(), rewriter, loc);
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                           rewriter.getI32IntegerAttr(i));
      Value elemPtr =
          LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, shapeArr, idx);
      LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
    }

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value rankVal = createI64Const(rank);
    Value dataTypeVal = createI64Const(dataType);
    Value axisDtypeVal = createI64Const(axisDtype);
    Value axisHostVal = createI64Const(op.getAxisAttr().value_or(0));
    Value exclusiveVal = createI64Const(op.getExclusive());
    Value reverseVal = createI64Const(op.getReverse());

    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, x, axis, y
        ptrType, i64Type,                   // x_shape, x_rank
        i64Type, i64Type, i64Type, i64Type, // num_elements, data_type,
                                            // axis_dtype, axis_host
        i64Type, i64Type};                  // exclusive, reverse

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCumSum, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {statePtr,    xPtr,         axisPtr,
                                   yPtr,        shapeArr,     rankVal,
                                   numElements, dataTypeVal,  axisDtypeVal,
                                   axisHostVal, exclusiveVal, reverseVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateCumSumLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<CumSumOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
