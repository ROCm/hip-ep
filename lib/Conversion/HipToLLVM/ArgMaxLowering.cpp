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

// hip.arg_max(ctx, data, output, axis, keepdims, select_last_index)
//   -> wrap_arg_max(state, data, indices, axis, keepdims, select_last_index,
//                   rank, data_shape_ptr, data_type)
struct ArgMaxOpLowering : public ConvertOpToLLVMPattern<ArgMaxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ArgMaxOp op, OpAdaptor adaptor,
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

    auto dataType = cast<MemRefType>(op.getData().getType());
    int rank = dataType.getRank();
    if (rank <= 0 || rank > 8)
      return rewriter.notifyMatchFailure(op, "rank must be in [1, 8]");

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    if (!outputType.getElementType().isInteger(64))
      return rewriter.notifyMatchFailure(op, "ArgMax indices must be i64");

    // isInteger(32/64) is true for ui32/ui64 as well, and getHipdnnDataType
    // would then label them INT32/INT64. The kernel compares those as signed,
    // so a high bit looks negative and the index is wrong. There is no
    // unsigned 32/64 ABI type; refuse them before the mapping.
    Type elemType = dataType.getElementType();
    if (elemType.isUnsignedInteger(32) || elemType.isUnsignedInteger(64))
      return rewriter.notifyMatchFailure(
          op, "ArgMax does not support ui32 or ui64");

    int64_t dataTypeEnum = getHipdnnDataType(elemType);
    if (dataTypeEnum < 0)
      return rewriter.notifyMatchFailure(op, "unsupported ArgMax element type");

    int64_t axisAttr = op.getAxis();
    if (axisAttr < 0)
      axisAttr += rank;
    if (axisAttr < 0 || axisAttr >= rank)
      return rewriter.notifyMatchFailure(op, "ArgMax axis out of range");

    Value statePtr = adaptor.getCtx();
    Value dataPtr =
        extractContiguousMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value shapeArr =
        buildShapeArray(dataType, adaptor.getData(), loc, rewriter);

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type,
                                       i64Type, ptrType, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapArgMax, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr,
                                  dataPtr,
                                  indicesPtr,
                                  createI64Const(axisAttr),
                                  createI64Const(op.getKeepdims()),
                                  createI64Const(op.getSelectLastIndex()),
                                  createI64Const(rank),
                                  shapeArr,
                                  createI64Const(dataTypeEnum)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateArgMaxLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<ArgMaxOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
