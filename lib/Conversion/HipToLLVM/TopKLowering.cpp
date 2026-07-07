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

// hip.top_k(ctx, x, k, values, indices, axis, largest, sorted)
//   -> wrap_top_k(state, x, k, values, indices, axis, largest, sorted, rank,
//                 x_shape_ptr, num_elements, element_size_bytes)
struct TopKOpLowering : public ConvertOpToLLVMPattern<TopKOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TopKOp op, OpAdaptor adaptor,
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
    Value xPtr = extractContiguousMemRefPtr(adaptor.getX(), rewriter, loc);
    Value kPtr = extractContiguousMemRefPtr(adaptor.getK(), rewriter, loc);
    Value valuesPtr =
        extractContiguousMemRefPtr(adaptor.getValues(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);

    auto xType = cast<MemRefType>(op.getX().getType());
    int rank = xType.getRank();
    if (rank <= 0 || rank > 8)
      return rewriter.notifyMatchFailure(op, "rank must be in [1, 8]");

    Value xShapeArr = buildShapeArray(xType, adaptor.getX(), loc, rewriter);
    Value numElements =
        computeNumElements(xType, adaptor.getX(), rewriter, loc);

    int64_t axisAttr = op.getAxis();
    if (axisAttr < 0)
      axisAttr += rank;

    Value axisVal = createI64Const(axisAttr);
    Value rankVal = createI64Const(rank);
    Value largestVal = createI64Const(op.getLargest() ? 1 : 0);
    Value sortedVal = createI64Const(op.getSorted() ? 1 : 0);

    unsigned elementSizeBytes =
        xType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, // state, x, k, values, idx
        i64Type, i64Type, i64Type,                   // axis, largest, sorted
        i64Type, ptrType, i64Type, i64Type};         // rank, shape, num, elem

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapTopK, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,   xPtr,      kPtr,        valuesPtr,
                               indicesPtr, axisVal,   largestVal,  sortedVal,
                               rankVal,    xShapeArr, numElements, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateTopKLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<TopKOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
