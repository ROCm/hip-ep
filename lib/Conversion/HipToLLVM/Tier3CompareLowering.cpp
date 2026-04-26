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
    if (!outType || !lhsType || !rhsType)
      return rewriter.notifyMatchFailure(
          op, "hip.compare lowering requires ranked memrefs");

    int64_t dataType = getHipdnnDataType(lhsType.getElementType());
    if (dataType < 0)
      dataType = 0;

    int64_t rank = outType.getRank();

    auto outShape = getMemRefShape(outType, adaptor.getOutput(), rewriter, loc);
    auto lhsShape = getMemRefShape(lhsType, adaptor.getLhs(), rewriter, loc);
    auto rhsShape = getMemRefShape(rhsType, adaptor.getRhs(), rewriter, loc);

    auto lhsStrides =
        computeBroadcastStridesSSA(lhsShape, outShape, rewriter, loc);
    auto rhsStrides =
        computeBroadcastStridesSSA(rhsShape, outShape, rewriter, loc);

    Value numElems = computeNumElements(outType, adaptor.getOutput(),
                                        rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value dtype = i64Const(dataType);
    Value kind = i64Const(static_cast<int64_t>(op.getKind()));
    Value rankConst = i64Const(rank);
    Value outShapePtr =
        materialiseValueArray(outShape, i64Type, ptrType, rewriter, loc);
    Value lhsStridePtr =
        materialiseValueArray(lhsStrides, i64Type, ptrType, rewriter, loc);
    Value rhsStridePtr =
        materialiseValueArray(rhsStrides, i64Type, ptrType, rewriter, loc);

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
    if (!outType || !condType || !xType || !yType)
      return rewriter.notifyMatchFailure(
          op, "hip.where lowering requires ranked memrefs");

    int64_t rank = outType.getRank();
    int64_t elementSize =
        outType.getElementType().getIntOrFloatBitWidth() / 8;

    auto outShape = getMemRefShape(outType, adaptor.getOutput(), rewriter, loc);
    auto condShape =
        getMemRefShape(condType, adaptor.getCond(), rewriter, loc);
    auto xShape = getMemRefShape(xType, adaptor.getX(), rewriter, loc);
    auto yShape = getMemRefShape(yType, adaptor.getY(), rewriter, loc);

    auto condStrides =
        computeBroadcastStridesSSA(condShape, outShape, rewriter, loc);
    auto xStrides =
        computeBroadcastStridesSSA(xShape, outShape, rewriter, loc);
    auto yStrides =
        computeBroadcastStridesSSA(yShape, outShape, rewriter, loc);

    Value numElems = computeNumElements(outType, adaptor.getOutput(),
                                        rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value elemSize = i64Const(elementSize);
    Value rankConst = i64Const(rank);
    Value outShapePtr =
        materialiseValueArray(outShape, i64Type, ptrType, rewriter, loc);
    Value condStridePtr =
        materialiseValueArray(condStrides, i64Type, ptrType, rewriter, loc);
    Value xStridePtr =
        materialiseValueArray(xStrides, i64Type, ptrType, rewriter, loc);
    Value yStridePtr =
        materialiseValueArray(yStrides, i64Type, ptrType, rewriter, loc);

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
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.layer_norm lowering requires ranked memrefs");

    int64_t rank = inType.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0)
      axis += rank;
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "hip.layer_norm axis out of range");

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

    // Compute outer = product of dims [0, axis) and normSize = product of
    // dims [axis, rank).  Uses runtime dim queries for dynamic dimensions.
    Value outerVal = i64Const(1);
    for (int64_t d = 0; d < axis; ++d)
      outerVal = LLVM::MulOp::create(
          rewriter, loc, outerVal,
          getMemRefDimSize(inType, d, adaptor.getInput(), rewriter, loc));
    Value normVal = i64Const(1);
    for (int64_t d = axis; d < rank; ++d)
      normVal = LLVM::MulOp::create(
          rewriter, loc, normVal,
          getMemRefDimSize(inType, d, adaptor.getInput(), rewriter, loc));
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
