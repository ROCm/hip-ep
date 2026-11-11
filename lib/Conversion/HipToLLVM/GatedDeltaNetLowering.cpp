/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct GatedDeltaNetOpLowering
    : public ConvertOpToLLVMPattern<GatedDeltaNetOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatedDeltaNetOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    auto qkvType = cast<MemRefType>(op.getQkv().getType());
    auto stateType = cast<MemRefType>(op.getInitialState().getType());
    const int64_t numHeadsV = stateType.getDimSize(1);
    const int64_t headSizeV = stateType.getDimSize(2);
    const int64_t headSizeQk = stateType.getDimSize(3);
    const int64_t numHeadsQk =
        (qkvType.getDimSize(2) - numHeadsV * headSizeV) / (2 * headSizeQk);

    SmallVector<Value, 17> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getQkv(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getDecay(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getBeta(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getInitialState(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getALog(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getDtBias(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getFinalState(), rewriter, loc),
        getMemRefDimSize(qkvType, 0, adaptor.getQkv(), rewriter, loc),
        getMemRefDimSize(qkvType, 1, adaptor.getQkv(), rewriter, loc),
        createI64Const(numHeadsQk),
        createI64Const(numHeadsV),
        createI64Const(headSizeQk),
        createI64Const(headSizeV),
        LLVM::ConstantOp::create(
            rewriter, loc, f32Type,
            rewriter.getF32FloatAttr(op.getScale().convertToFloat())),
        createI64Const(qkvType.getElementType().getIntOrFloatBitWidth() / 8)};

    SmallVector<Type, 17> paramTypes(9, ptrType);
    paramTypes.append(6, i64Type);
    paramTypes.push_back(f32Type);
    paramTypes.push_back(i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGatedDeltaNet, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

struct GatedDeltaNetConvNormOpLowering
    : public ConvertOpToLLVMPattern<GatedDeltaNetConvNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatedDeltaNetConvNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();
    auto outputs = adaptor.getOutputs();
    if (outputs.size() != 9)
      return failure();

    auto i64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto f32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };
    auto projType = cast<MemRefType>(op.getProj().getType());
    auto gatesType = cast<MemRefType>(op.getGates().getType());
    auto convStateType = cast<MemRefType>(op.getConvState().getType());
    auto stateType = cast<MemRefType>(op.getRecurrentState().getType());
    auto weightType = cast<MemRefType>(op.getConvWeight().getType());
    const int64_t numHeadsV = stateType.getDimSize(1);
    const int64_t headSizeV = stateType.getDimSize(2);
    const int64_t headSizeQk = stateType.getDimSize(3);
    const int64_t channels = op.getConvChannels();
    const int64_t numHeadsQk =
        (channels - numHeadsV * headSizeV) / (2 * headSizeQk);

    SmallVector<Value> args = {adaptor.getCtx()};
    for (Value operand :
         {adaptor.getProj(), adaptor.getConvWeight(), adaptor.getConvBias(),
          adaptor.getConvState(), adaptor.getGates(),
          adaptor.getRecurrentState(), adaptor.getALog(), adaptor.getDtBias(),
          adaptor.getNormWeight()})
      args.push_back(extractContiguousMemRefPtr(operand, rewriter, loc));
    for (Value output : outputs)
      args.push_back(extractContiguousMemRefPtr(output, rewriter, loc));
    args.append({computeNumElements(projType, adaptor.getProj(), rewriter, loc),
                 i64Const(projType.getDimSize(2)),
                 computeNumElements(convStateType, adaptor.getConvState(),
                                    rewriter, loc),
                 i64Const(gatesType.getDimSize(2)), i64Const(channels),
                 i64Const(weightType.getDimSize(2)),
                 i64Const(op.getDecayColumn()), i64Const(op.getBetaColumn()),
                 i64Const(op.getZColumn()), i64Const(numHeadsQk),
                 i64Const(numHeadsV), i64Const(headSizeQk), i64Const(headSizeV),
                 f32Const(op.getScale().convertToFloat()),
                 f32Const(op.getEpsilon().convertToFloat())});

    // ctx, nine inputs and nine outputs.
    SmallVector<Type> paramTypes(19, ptrType);
    paramTypes.append(13, i64Type);
    paramTypes.append(2, f32Type);
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapGatedDeltaNetConvNorm,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGatedDeltaNetLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<GatedDeltaNetOpLowering, GatedDeltaNetConvNormOpLowering>(
      converter);
}

} // namespace hip
} // namespace mlir
