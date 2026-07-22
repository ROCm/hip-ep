/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

static Value buildShapeArray(MemRefType type, Value memref, Location loc,
                             ConversionPatternRewriter &rewriter) {
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));
  int rank = type.getRank();
  int arrLen = std::max(rank, 1);
  auto arrType = LLVM::LLVMArrayType::get(i64Type, arrLen);
  Value shapeArr =
      LLVM::AllocaOp::create(rewriter, loc, ptrType, arrType, one, 8);
  for (int i = 0; i < rank; ++i) {
    Value dim = getMemRefDimSize(type, i, memref, rewriter, loc);
    Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                         rewriter.getI32IntegerAttr(i));
    Value elemPtr =
        LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type, shapeArr, idx);
    LLVM::StoreOp::create(rewriter, loc, dim, elemPtr);
  }
  return shapeArr;
}

// hip.compress(ctx, input, condition, output, axis, flatten)
//   -> wrap_compress(state, input, condition, output, flatten, axis,
//                     input_rank, output_rank, input_shape_ptr,
//                     output_shape_ptr, condition_len, num_output_elements,
//                     element_size_bytes)
struct CompressOpLowering : public ConvertOpToLLVMPattern<CompressOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CompressOp op, OpAdaptor adaptor,
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

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value conditionPtr =
        extractContiguousMemRefPtr(adaptor.getCondition(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto conditionType = cast<MemRefType>(op.getCondition().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int inputRank = inputType.getRank();
    int outputRank = outputType.getRank();
    if (inputRank <= 0 || inputRank > 8 || outputRank <= 0 || outputRank > 8)
      return rewriter.notifyMatchFailure(op, "rank must be in [1, 8]");

    Value inputShapeArr =
        buildShapeArray(inputType, adaptor.getInput(), loc, rewriter);
    Value outputShapeArr =
        buildShapeArray(outputType, adaptor.getOutput(), loc, rewriter);

    Value conditionLen = getMemRefDimSize(
        conditionType, 0, adaptor.getCondition(), rewriter, loc);
    Value numOutputElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    int64_t axisAttr = op.getAxis();
    if (!op.getFlatten() && axisAttr < 0)
      axisAttr += inputRank;

    Value flattenVal = createI64Const(op.getFlatten() ? 1 : 0);
    Value axisVal = createI64Const(axisAttr);
    Value inputRankVal = createI64Const(inputRank);
    Value outputRankVal = createI64Const(outputRank);

    unsigned elementSizeBytes =
        inputType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, input, cond, output
        i64Type, i64Type,                   // flatten, axis
        i64Type, i64Type,                   // input_rank, output_rank
        ptrType, ptrType,                   // input_shape, output_shape
        i64Type, i64Type, i64Type};         // cond_len, num_out, elem_size

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCompress, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr,      inputPtr,       conditionPtr, outputPtr,
        flattenVal,    axisVal,        inputRankVal, outputRankVal,
        inputShapeArr, outputShapeArr, conditionLen, numOutputElements,
        elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateCompressLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns) {
  patterns.add<CompressOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
