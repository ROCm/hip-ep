/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers Tier 3/4 hip ops to wrap_* runtime calls:
//   - hip.compare    -> wrap_compare
//   - hip.where      -> wrap_where
//   - hip.layer_norm -> wrap_layer_norm

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

static SmallVector<int64_t> computeBroadcastStrides(ArrayRef<int64_t> inShape,
                                                    ArrayRef<int64_t> outShape) {
  int64_t rank = outShape.size();
  SmallVector<int64_t> padded(rank, 1);
  int64_t offset = rank - inShape.size();
  for (size_t i = 0; i < inShape.size(); ++i)
    padded[offset + i] = inShape[i];
  SmallVector<int64_t> strides(rank, 0);
  int64_t acc = 1;
  for (int64_t d = rank - 1; d >= 0; --d) {
    strides[d] = padded[d] == 1 ? 0 : acc;
    acc *= padded[d];
  }
  return strides;
}

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

//===----------------------------------------------------------------------===//
// hip.compare
//===----------------------------------------------------------------------===//

struct CompareLowering : public ConvertOpToLLVMPattern<CompareOp> {
  using ConvertOpToLLVMPattern<CompareOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CompareOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = extractMemRefPtr(adaptor.getLhs(), rewriter, loc);
    Value rhsPtr = extractMemRefPtr(adaptor.getRhs(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    auto lhsType = dyn_cast<MemRefType>(op.getLhs().getType());
    auto rhsType = dyn_cast<MemRefType>(op.getRhs().getType());
    if (!outType || !lhsType || !rhsType || !outType.hasStaticShape() ||
        !lhsType.hasStaticShape() || !rhsType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.compare lowering requires static shapes");

    // Use the lhs element type to derive the runtime data type.  And kind
    // ignores it (kernel always treats inputs as bool/i8) but we still need
    // to pass *something* sensible.
    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    if (dataType < 0)
      dataType = 0; // f32 sentinel; the And kernel doesn't read it.

    int64_t rank = outType.getRank();
    SmallVector<int64_t> outShape(outType.getShape().begin(),
                                  outType.getShape().end());
    SmallVector<int64_t> lhsStrides =
        computeBroadcastStrides(lhsType.getShape(), outShape);
    SmallVector<int64_t> rhsStrides =
        computeBroadcastStrides(rhsType.getShape(), outShape);

    int64_t numElements = 1;
    for (int64_t d : outShape)
      numElements *= d;

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value numElems = i64Const(numElements);
    Value dtype = i64Const(dataType);
    Value kind = i64Const(static_cast<int64_t>(op.getKind()));
    Value rankConst = i64Const(rank);
    Value outShapePtr =
        materialiseI64Array(outShape, i64Type, ptrType, rewriter, loc);
    Value lhsStridePtr =
        materialiseI64Array(lhsStrides, i64Type, ptrType, rewriter, loc);
    Value rhsStridePtr =
        materialiseI64Array(rhsStrides, i64Type, ptrType, rewriter, loc);

    SmallVector<Type, 11> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        i64Type, i64Type, i64Type, i64Type,
                                        ptrType, ptrType, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCompare, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 11> args = {statePtr,    lhsPtr,       rhsPtr,
                                   outPtr,      numElems,     dtype,
                                   kind,        rankConst,    outShapePtr,
                                   lhsStridePtr, rhsStridePtr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// hip.where
//===----------------------------------------------------------------------===//

struct WhereLowering : public ConvertOpToLLVMPattern<WhereOp> {
  using ConvertOpToLLVMPattern<WhereOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(WhereOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value condPtr = extractMemRefPtr(adaptor.getCond(), rewriter, loc);
    Value xPtr = extractMemRefPtr(adaptor.getX(), rewriter, loc);
    Value yPtr = extractMemRefPtr(adaptor.getY(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    auto condType = dyn_cast<MemRefType>(op.getCond().getType());
    auto xType = dyn_cast<MemRefType>(op.getX().getType());
    auto yType = dyn_cast<MemRefType>(op.getY().getType());
    if (!outType || !condType || !xType || !yType ||
        !outType.hasStaticShape() || !condType.hasStaticShape() ||
        !xType.hasStaticShape() || !yType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.where lowering requires static shapes");

    int64_t rank = outType.getRank();
    SmallVector<int64_t> outShape(outType.getShape().begin(),
                                  outType.getShape().end());
    SmallVector<int64_t> condStrides =
        computeBroadcastStrides(condType.getShape(), outShape);
    SmallVector<int64_t> xStrides =
        computeBroadcastStrides(xType.getShape(), outShape);
    SmallVector<int64_t> yStrides =
        computeBroadcastStrides(yType.getShape(), outShape);

    int64_t numElements = 1;
    for (int64_t d : outShape)
      numElements *= d;
    int64_t elementSize =
        outType.getElementType().getIntOrFloatBitWidth() / 8;

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value numElems = i64Const(numElements);
    Value elemSize = i64Const(elementSize);
    Value rankConst = i64Const(rank);
    Value outShapePtr =
        materialiseI64Array(outShape, i64Type, ptrType, rewriter, loc);
    Value condStridePtr =
        materialiseI64Array(condStrides, i64Type, ptrType, rewriter, loc);
    Value xStridePtr =
        materialiseI64Array(xStrides, i64Type, ptrType, rewriter, loc);
    Value yStridePtr =
        materialiseI64Array(yStrides, i64Type, ptrType, rewriter, loc);

    SmallVector<Type, 12> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        ptrType, i64Type, i64Type, i64Type,
                                        ptrType, ptrType, ptrType, ptrType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapWhere, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {statePtr,    condPtr,       xPtr,
                                   yPtr,        outPtr,        numElems,
                                   elemSize,    rankConst,     outShapePtr,
                                   condStridePtr, xStridePtr,  yStridePtr};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// hip.layer_norm
//===----------------------------------------------------------------------===//

struct LayerNormLowering : public ConvertOpToLLVMPattern<LayerNormOp> {
  using ConvertOpToLLVMPattern<LayerNormOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f64Type = rewriter.getF64Type();

    Value statePtr = adaptor.getCtx();
    Value inPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value gammaPtr = extractMemRefPtr(adaptor.getGamma(), rewriter, loc);
    Value betaPtr =
        extractOptionalMemRefPtr(adaptor.getBeta(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType || !inType.hasStaticShape() ||
        !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.layer_norm lowering requires static shapes");

    int64_t rank = inType.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "hip.layer_norm axis out of range");

    int64_t outer = 1;
    for (int64_t d = 0; d < axis; ++d)
      outer *= inType.getDimSize(d);
    int64_t normSize = 1;
    for (int64_t d = axis; d < rank; ++d)
      normSize *= inType.getDimSize(d);

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.layer_norm unsupported element type");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    auto f64Const = [&](double v) {
      return LLVM::ConstantOp::create(rewriter, loc, f64Type,
                                      rewriter.getF64FloatAttr(v));
    };

    Value outerVal = i64Const(outer);
    Value normVal = i64Const(normSize);
    Value epsVal = f64Const(op.getEpsilon().convertToDouble());
    Value dtypeVal = i64Const(dataType);

    // int wrap_layer_norm(state, x, gamma, beta, y, outer, norm_size,
    //                    epsilon, data_type)
    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type, f64Type,
                                       i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapLayerNorm, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr, inPtr,    gammaPtr, betaPtr,
                                  outPtr,   outerVal, normVal,  epsVal,
                                  dtypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateTier3CompareLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<CompareLowering, WhereLowering, LayerNormLowering>(converter);
}

} // namespace hip
} // namespace mlir
