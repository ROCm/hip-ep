/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.slice (already-normalised ONNX Slice) to wrap_slice in the
// runtime bitcode.  Output shapes and input strides are read from memref
// descriptors at runtime; starts/steps remain compile-time attributes.

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

struct SliceLowering : public ConvertOpToLLVMPattern<SliceOp> {
  using ConvertOpToLLVMPattern<SliceOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SliceOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.slice expects ranked memref operands");

    int64_t rank = inType.getRank();
    if (outType.getRank() != rank)
      return rewriter.notifyMatchFailure(
          op, "hip.slice expects matching input/output ranks");

    auto startsAttr = op.getStarts();
    auto stepsAttr = op.getSteps();
    if ((int64_t)startsAttr.size() != rank ||
        (int64_t)stepsAttr.size() != rank)
      return rewriter.notifyMatchFailure(
          op, "hip.slice starts/steps lengths must match input rank");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    auto outShape =
        getMemRefShape(outType, adaptor.getOutput(), rewriter, loc);

    // Row-major contiguous strides from input descriptor.
    SmallVector<Value> inputStrides(rank);
    Value acc = i64Const(1);
    for (int64_t d = rank - 1; d >= 0; --d) {
      inputStrides[d] = acc;
      acc = LLVM::MulOp::create(
          rewriter, loc, acc,
          getMemRefDimSize(inType, d, adaptor.getInput(), rewriter, loc));
    }

    // starts are attributes (compile-time), steps scale the strides.
    SmallVector<Value> starts(rank);
    SmallVector<Value> steppedStrides(rank);
    for (int64_t d = 0; d < rank; ++d) {
      starts[d] = i64Const(cast<IntegerAttr>(startsAttr[d]).getInt());
      int64_t step = cast<IntegerAttr>(stepsAttr[d]).getInt();
      steppedStrides[d] =
          LLVM::MulOp::create(rewriter, loc, inputStrides[d], i64Const(step));
    }

    int64_t elementSize =
        outType.getElementType().getIntOrFloatBitWidth() / 8;
    Value numElems = computeNumElements(outType, adaptor.getOutput(),
                                        rewriter, loc);
    Value elemSize = i64Const(elementSize);
    Value rankConst = i64Const(rank);

    Value outShapePtr =
        materialiseValueArray(outShape, i64Type, ptrType, rewriter, loc);
    Value strideArr =
        materialiseValueArray(steppedStrides, i64Type, ptrType, rewriter, loc);
    Value startArr =
        materialiseValueArray(starts, i64Type, ptrType, rewriter, loc);

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, i64Type, ptrType, ptrType,
                                       ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSlice, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr,    inputPtr,    outputPtr,
                                  numElems,    elemSize,    rankConst,
                                  outShapePtr, strideArr,   startArr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateSliceLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<SliceLowering>(converter);
}

} // namespace hip
} // namespace mlir
