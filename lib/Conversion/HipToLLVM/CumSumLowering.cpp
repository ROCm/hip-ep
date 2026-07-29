/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.cumsum(ctx, x, axis, y, exclusive, reverse)
//   -> wrap_cumsum(state, x_ptr, axis_ptr, y_ptr,
//                  x_shape_ptr, x_rank,
//                  num_elements, data_type, axis_dtype,
//                  exclusive, reverse)
//
// `axis` is a rank-0 (scalar) GPU tensor of i32/i64 selecting the reduction
// axis. The runtime is responsible for reading it -- we only forward the
// pointer plus the axis dtype enum so the runtime knows whether to treat
// the byte buffer as int32 or int64.
struct CumSumOpLowering : public ConvertOpToLLVMPattern<CumSumOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(CumSumOpLowering)
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
        extractContiguousMemRefPtr(adaptor.getAxis(), rewriter, loc);
    Value yPtr = extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc);

    auto xType = cast<MemRefType>(op.getX().getType());
    auto axisType = cast<MemRefType>(op.getAxis().getType());

    int64_t dataType = getHipdnnDataType(xType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");
    int64_t axisDtype = getHipdnnDataType(axisType.getElementType());
    if (axisDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported axis element type");

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
    Value exclusiveVal = createI64Const(op.getExclusive());
    Value reverseVal = createI64Const(op.getReverse());

    SmallVector<Type, 11> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, x, axis, y
        ptrType, i64Type,                   // x_shape, x_rank
        i64Type, i64Type, i64Type, // num_elements, data_type, axis_dtype
        i64Type, i64Type};         // exclusive, reverse

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCumSum, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 11> args = {statePtr,     xPtr,        axisPtr,
                                   yPtr,         shapeArr,    rankVal,
                                   numElements,  dataTypeVal, axisDtypeVal,
                                   exclusiveVal, reverseVal};

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
