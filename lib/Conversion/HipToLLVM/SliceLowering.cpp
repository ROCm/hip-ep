/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.slice (already-normalised ONNX Slice) to wrap_slice in the
// runtime bitcode.  All three i64 arrays (out_shape, in_strides_elems,
// starts_elems) are stack-allocated since the conversion pass already
// folded everything down to static shapes / step values.

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

static Value materialiseI64Array(ArrayRef<int64_t> vals, Type i64Type,
                                  Type ptrType,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
  auto arrayType = LLVM::LLVMArrayType::get(i64Type, vals.size());
  Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));
  Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayType, one,
                                        /*alignment=*/8);
  for (size_t i = 0; i < vals.size(); ++i) {
    Value v = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(vals[i]));
    SmallVector<LLVM::GEPArg, 2> indices = {0, static_cast<int32_t>(i)};
    Value elemPtr = LLVM::GEPOp::create(rewriter, loc, ptrType, arrayType,
                                        alloca, indices);
    LLVM::StoreOp::create(rewriter, loc, v, elemPtr);
  }
  return alloca;
}

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
    if (!inType.hasStaticShape() || !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.slice lowering requires static shapes");

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

    SmallVector<int64_t> outShape(outType.getShape().begin(),
                                  outType.getShape().end());

    // Row-major contiguous strides of the input tensor in elements.
    SmallVector<int64_t> inputStrides(rank, 0);
    int64_t acc = 1;
    for (int64_t d = rank - 1; d >= 0; --d) {
      inputStrides[d] = acc;
      acc *= inType.getDimSize(d);
    }

    SmallVector<int64_t> starts(rank, 0);
    SmallVector<int64_t> steppedStrides(rank, 0);
    for (int64_t d = 0; d < rank; ++d) {
      starts[d] = cast<IntegerAttr>(startsAttr[d]).getInt();
      int64_t step = cast<IntegerAttr>(stepsAttr[d]).getInt();
      steppedStrides[d] = inputStrides[d] * step;
    }

    int64_t elementSize =
        outType.getElementType().getIntOrFloatBitWidth() / 8;
    int64_t numElements = 1;
    for (int64_t d : outShape)
      numElements *= d;

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value numElems = i64Const(numElements);
    Value elemSize = i64Const(elementSize);
    Value rankConst = i64Const(rank);

    Value outShapePtr =
        materialiseI64Array(outShape, i64Type, ptrType, rewriter, loc);
    Value strideArr =
        materialiseI64Array(steppedStrides, i64Type, ptrType, rewriter, loc);
    Value startArr =
        materialiseI64Array(starts, i64Type, ptrType, rewriter, loc);

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
