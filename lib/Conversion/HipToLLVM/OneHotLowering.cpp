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

// hip.one_hot(ctx, indices, depth, values, output, axis)
//   -> wrap_one_hot(state, indices, depth, values, output, axis,
//                   indices_rank, output_rank, indices_shape_ptr,
//                   output_shape_ptr, num_indices, num_output_elements,
//                   element_size_bytes, indices_element_size_bytes,
//                   depth_element_size_bytes)
struct OneHotOpLowering : public ConvertOpToLLVMPattern<OneHotOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OneHotOp op, OpAdaptor adaptor,
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
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value depthPtr =
        extractContiguousMemRefPtr(adaptor.getDepth(), rewriter, loc);
    Value valuesPtr =
        extractContiguousMemRefPtr(adaptor.getValues(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    auto depthType = cast<MemRefType>(op.getDepth().getType());

    int indicesRank = indicesType.getRank();
    int outputRank = outputType.getRank();
    if (indicesRank < 0 || indicesRank > 7 || outputRank <= indicesRank ||
        outputRank > 8)
      return rewriter.notifyMatchFailure(op, "invalid one_hot ranks");

    Value indicesShapeArr =
        buildShapeArray(indicesType, adaptor.getIndices(), loc, rewriter);
    Value outputShapeArr =
        buildShapeArray(outputType, adaptor.getOutput(), loc, rewriter);

    Value numIndices =
        computeNumElements(indicesType, adaptor.getIndices(), rewriter, loc);
    Value numOutputElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    int64_t axisAttr = op.getAxis();
    int64_t normAxis = axisAttr;
    if (normAxis < 0)
      normAxis += outputRank;

    Value axisVal = createI64Const(normAxis);
    Value indicesRankVal = createI64Const(indicesRank);
    Value outputRankVal = createI64Const(outputRank);

    unsigned elementSizeBytes =
        outputType.getElementType().getIntOrFloatBitWidth() / 8;
    unsigned indicesElemBytes =
        indicesType.getElementType().getIntOrFloatBitWidth() / 8;
    unsigned depthElemBytes =
        depthType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);
    Value indicesElemSizeVal = createI64Const(indicesElemBytes);
    Value depthElemSizeVal = createI64Const(depthElemBytes);

    SmallVector<Type, 14> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, // state, idx, depth, val, out
        i64Type, i64Type, i64Type,                   // axis, idx_rank, out_rank
        ptrType, ptrType,                              // idx_shape, out_shape
        i64Type, i64Type, i64Type, i64Type, i64Type};  // num_idx, num_out, elem, idx_elem, depth_elem

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapOneHot, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 14> args = {
        statePtr,          indicesPtr,        depthPtr, valuesPtr,
        outputPtr,         axisVal,           indicesRankVal, outputRankVal,
        indicesShapeArr,   outputShapeArr,    numIndices, numOutputElements,
        elemSizeVal,       indicesElemSizeVal, depthElemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateOneHotLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<OneHotOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
