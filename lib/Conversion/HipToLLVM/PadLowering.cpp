/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.pad(ctx, data, pads, [cval], [axes], output) {mode}
//   -> wrap_pad(state, data_ptr, pads_ptr, cval_ptr (or null), axes_ptr (or
//               null), out_ptr,
//               data_shape_ptr, data_rank,
//               output_shape_ptr, output_rank,
//               pads_num_elements, axes_num_elements,
//               data_type, mode_id)
//
// `mode_id` encodes the string attribute as a small enum:
//   0 = constant, 1 = reflect, 2 = edge, 3 = wrap.
struct PadOpLowering : public ConvertOpToLLVMPattern<PadOp> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PadOpLowering)
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static int64_t modeIdFromString(StringRef m) {
    if (m == "reflect")
      return 1;
    if (m == "edge")
      return 2;
    if (m == "wrap")
      return 3;
    return 0; // constant default
  }

  LogicalResult
  matchAndRewrite(PadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto padsType = cast<MemRefType>(op.getPads().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t hipDtype = getHipdnnDataType(dataType.getElementType());
    if (hipDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported data element type");

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value padsPtr =
        extractContiguousMemRefPtr(adaptor.getPads(), rewriter, loc);
    Value outPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value cvalPtr =
        extractOptionalMemRefPtr(adaptor.getConstantValue(), rewriter, loc);
    Value axesPtr = extractOptionalMemRefPtr(adaptor.getAxes(), rewriter, loc);

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
    Value padsNum =
        computeNumElements(padsType, adaptor.getPads(), rewriter, loc);
    Value axesNum;
    if (op.getAxes()) {
      auto axesT = cast<MemRefType>(op.getAxes().getType());
      axesNum = computeNumElements(axesT, adaptor.getAxes(), rewriter, loc);
    } else {
      axesNum = createI64Const(0);
    }

    Value dataRank = createI64Const(dataType.getRank());
    Value outRank = createI64Const(outputType.getRank());
    Value dataTypeVal = createI64Const(hipDtype);
    Value modeIdVal = createI64Const(modeIdFromString(op.getMode()));

    SmallVector<Type, 14> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, // state + 5 ptrs
        ptrType, i64Type,  // data_shape, data_rank
        ptrType, i64Type,  // out_shape,  out_rank
        i64Type, i64Type,  // pads_num,   axes_num
        i64Type, i64Type}; // data_type,  mode_id

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapPad, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 14> args = {
        statePtr, dataPtr,  padsPtr, cvalPtr, axesPtr, outPtr,      dataShape,
        dataRank, outShape, outRank, padsNum, axesNum, dataTypeVal, modeIdVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePadLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns) {
  patterns.add<PadOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
