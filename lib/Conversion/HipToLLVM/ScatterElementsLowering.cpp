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

struct ScatterElementsOpLowering
    : public ConvertOpToLLVMPattern<ScatterElementsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static int64_t reductionIdFromString(StringRef r) {
    if (r == "add")
      return 1;
    if (r == "mul")
      return 2;
    if (r == "min")
      return 3;
    if (r == "max")
      return 4;
    return 0;
  }

  LogicalResult
  matchAndRewrite(ScatterElementsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto indicesType = cast<MemRefType>(op.getIndices().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int rank = dataType.getRank();
    if (rank <= 0 || rank > 8)
      return rewriter.notifyMatchFailure(op, "rank must be in [1, 8]");

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value updatesPtr =
        extractContiguousMemRefPtr(adaptor.getUpdates(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value dataShapeArr =
        buildShapeArray(dataType, adaptor.getData(), loc, rewriter);
    Value indicesShapeArr =
        buildShapeArray(indicesType, adaptor.getIndices(), loc, rewriter);

    Value numUpdates =
        computeNumElements(indicesType, adaptor.getIndices(), rewriter, loc);

    int64_t axisAttr = op.getAxis();
    if (axisAttr < 0)
      axisAttr += rank;

    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    unsigned indicesElemBytes =
        indicesType.getElementType().getIntOrFloatBitWidth() / 8;

    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, // state, data, idx, upd, out
        i64Type, i64Type, i64Type,                   // axis, reduction, rank
        ptrType, ptrType,                            // data_shape, idx_shape
        i64Type, i64Type, i64Type};                  // num_upd, elem, idx_elem

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapScatterElements, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr,     dataPtr,      indicesPtr,   updatesPtr, outputPtr,
        createI64Const(axisAttr), createI64Const(reductionIdFromString(op.getReduction())),
        createI64Const(rank), dataShapeArr, indicesShapeArr, numUpdates,
        createI64Const(elementSizeBytes), createI64Const(indicesElemBytes)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateScatterElementsLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns) {
  patterns.add<ScatterElementsOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
