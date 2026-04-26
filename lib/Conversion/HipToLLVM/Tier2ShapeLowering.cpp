/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers the Tier 2 hip ops to wrap_* runtime calls:
//   - hip.reduce_mean      -> wrap_reduce_mean
//   - hip.concat           -> wrap_concat
//   - hip.constant_of_shape-> wrap_constant_of_shape

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

/// Stack-allocate an array of `ptrType` pointers and store each value.
static Value materialisePtrArray(ArrayRef<Value> ptrs, Type ptrType,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
  Type i64Type = rewriter.getI64Type();
  auto arrayType = LLVM::LLVMArrayType::get(ptrType, ptrs.size());
  Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));
  Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType, one,
                                        /*alignment=*/8);
  for (size_t i = 0; i < ptrs.size(); ++i) {
    SmallVector<LLVM::GEPArg, 2> indices = {0, static_cast<int32_t>(i)};
    Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrType, arrayType,
                                        alloca, indices);
    LLVM::StoreOp::create(rewriter, loc, ptrs[i], elemPtr);
  }
  return alloca;
}

//===----------------------------------------------------------------------===//
// hip.reduce_mean
//===----------------------------------------------------------------------===//

struct ReduceMeanLowering : public ConvertOpToLLVMPattern<ReduceMeanOp> {
  using ConvertOpToLLVMPattern<ReduceMeanOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReduceMeanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getData(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inType = dyn_cast<MemRefType>(op.getData().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.reduce_mean expects ranked memref operands");

    Value numIn =
        computeNumElements(inType, adaptor.getData(), rewriter, loc);
    Value numOut =
        computeNumElements(outType, adaptor.getOutput(), rewriter, loc);

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.reduce_mean unsupported element type");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value dtype = i64Const(dataType);
    Value innerSize;
    int64_t innerAxisVal = op.getInnerAxis();
    if (innerAxisVal >= 0) {
      int64_t rank = inType.getRank();
      innerSize = i64Const(1);
      for (int64_t d = innerAxisVal; d < rank; ++d)
        innerSize = LLVM::MulOp::create(
            rewriter, loc, innerSize,
            getMemRefDimSize(inType, d, adaptor.getData(), rewriter, loc));
    } else {
      innerSize = i64Const(op.getInnerSize());
    }

    // int wrap_reduce_mean(state, in, out, num_in, num_out, data_type,
    //                      inner_size)
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapReduceMean, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr, inputPtr, outputPtr, numIn,
                                  numOut,   dtype,    innerSize};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// hip.concat
//===----------------------------------------------------------------------===//

struct ConcatLowering : public ConvertOpToLLVMPattern<ConcatOp> {
  using ConvertOpToLLVMPattern<ConcatOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConcatOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!outType)
      return rewriter.notifyMatchFailure(
          op, "hip.concat lowering requires a ranked memref output");

    int64_t axis = op.getAxis();
    int64_t rank = outType.getRank();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "hip.concat axis out of range");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // Compute outer = product(outDims[0..axis)) from descriptor.
    Value outerVal = i64Const(1);
    for (int64_t d = 0; d < axis; ++d)
      outerVal = LLVM::MulOp::create(
          rewriter, loc, outerVal,
          getMemRefDimSize(outType, d, adaptor.getOutput(), rewriter, loc));

    // Compute outputInner = product(outDims[axis..rank)) from descriptor.
    Value outInnerVal = i64Const(1);
    for (int64_t d = axis; d < rank; ++d)
      outInnerVal = LLVM::MulOp::create(
          rewriter, loc, outInnerVal,
          getMemRefDimSize(outType, d, adaptor.getOutput(), rewriter, loc));

    int64_t elementSize =
        outType.getElementType().getIntOrFloatBitWidth() / 8;

    auto convertedInputs = adaptor.getInputs();
    SmallVector<Value> inputPtrs;
    SmallVector<Value> innerSizeValues;
    for (auto [in, val] : llvm::zip(op.getInputs(), convertedInputs)) {
      auto inType = dyn_cast<MemRefType>(in.getType());
      if (!inType)
        return rewriter.notifyMatchFailure(
            op, "hip.concat input must be a ranked memref");
      Value inner = i64Const(1);
      for (int64_t d = axis; d < rank; ++d)
        inner = LLVM::MulOp::create(
            rewriter, loc, inner,
            getMemRefDimSize(inType, d, val, rewriter, loc));
      innerSizeValues.push_back(inner);
      inputPtrs.push_back(extractMemRefPtr(val, rewriter, loc));
    }

    Value elemSize = i64Const(elementSize);
    Value numInputs = i64Const(static_cast<int64_t>(inputPtrs.size()));

    Value inputsArrPtr =
        materialisePtrArray(inputPtrs, ptrType, rewriter, loc);
    Value innerSizesPtr =
        materialiseValueArray(innerSizeValues, i64Type, ptrType, rewriter, loc);

    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, i64Type, i64Type,
                                       i64Type, i64Type, ptrType, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapConcat, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr,    outPtr,        elemSize,
                                  outerVal,    outInnerVal,   numInputs,
                                  inputsArrPtr, innerSizesPtr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// hip.constant_of_shape
//===----------------------------------------------------------------------===//

struct ConstantOfShapeLowering
    : public ConvertOpToLLVMPattern<ConstantOfShapeOp> {
  using ConvertOpToLLVMPattern<ConstantOfShapeOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConstantOfShapeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!outType)
      return rewriter.notifyMatchFailure(
          op, "hip.constant_of_shape requires a ranked memref output");

    int64_t elementSize =
        outType.getElementType().getIntOrFloatBitWidth() / 8;

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value numElems = computeNumElements(outType, adaptor.getOutput(),
                                        rewriter, loc);
    Value elemSize = i64Const(elementSize);
    Value scalarBits = i64Const(static_cast<int64_t>(op.getScalarBits()));

    SmallVector<Type, 5> paramTypes = {ptrType, ptrType, i64Type, i64Type,
                                       i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapConstantOfShape, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 5> args = {statePtr, outPtr, numElems, elemSize,
                                  scalarBits};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateTier2ShapeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<ReduceMeanLowering, ConcatLowering, ConstantOfShapeLowering>(
      converter);
}

} // namespace hip
} // namespace mlir
