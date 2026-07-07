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

// hip.gather_elements(ctx, data, indices, output, axis)
//   -> wrap_gather_elements(state, data, indices, output, axis, rank,
//                          data_shape_ptr, indices_shape_ptr, num_elements,
//                          element_size_bytes, indices_element_size_bytes)
struct GatherElementsOpLowering
    : public ConvertOpToLLVMPattern<GatherElementsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherElementsOp op, OpAdaptor adaptor,
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
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int rank = dataType.getRank();
    if (rank <= 0 || rank > 8)
      return rewriter.notifyMatchFailure(op, "rank must be in [1, 8]");

    Value dataShapeArr =
        buildShapeArray(dataType, adaptor.getData(), loc, rewriter);
    Value indicesShapeArr =
        buildShapeArray(indicesType, adaptor.getIndices(), loc, rewriter);

    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    int64_t axisAttr = op.getAxis();
    if (axisAttr < 0)
      axisAttr += rank;
    Value axisVal = createI64Const(axisAttr);
    Value rankVal = createI64Const(rank);

    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    unsigned indicesElemBytes =
        indicesType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);
    Value indicesElemSizeVal = createI64Const(indicesElemBytes);

    SmallVector<Type, 11> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, data, indices, output
        i64Type, i64Type,                   // axis, rank
        ptrType, ptrType,                   // data_shape, indices_shape
        i64Type, i64Type, i64Type};         // num_elements, elem, idx_elem

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGatherElements, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 11> args = {statePtr,     dataPtr,           indicesPtr,
                                   outputPtr,    axisVal,           rankVal,
                                   dataShapeArr, indicesShapeArr,   numElements,
                                   elemSizeVal,  indicesElemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGatherElementsLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns) {
  patterns.add<GatherElementsOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
