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
    const int64_t packed = qkvType.getDimSize(qkvType.getRank() - 1);
    const int64_t numHeadsQk =
        (packed - numHeadsV * headSizeV) / (2 * headSizeQk);

    // Every dim but the packed one is a token axis, whether qkv is the
    // uniform [B, L, P] or the ragged [T, P] form.
    Value totalTokens =
        getMemRefDimSize(qkvType, 0, adaptor.getQkv(), rewriter, loc);
    for (int64_t i = 1; i < qkvType.getRank() - 1; ++i) {
      Value dim = getMemRefDimSize(qkvType, i, adaptor.getQkv(), rewriter, loc);
      totalTokens = LLVM::MulOp::create(rewriter, loc, totalTokens, dim);
    }

    Value cu = op.getCumulativeSequenceLength()
                   ? extractContiguousMemRefPtr(
                         adaptor.getCumulativeSequenceLength(), rewriter, loc)
                   : LLVM::ZeroOp::create(rewriter, loc, ptrType).getResult();

    SmallVector<Value, 18> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getQkv(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getDecay(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getBeta(), rewriter, loc),
        cu,
        extractContiguousMemRefPtr(adaptor.getInitialState(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getALog(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getDtBias(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getFinalState(), rewriter, loc),
        totalTokens,
        getMemRefDimSize(stateType, 0, adaptor.getInitialState(), rewriter,
                         loc),
        createI64Const(numHeadsQk),
        createI64Const(numHeadsV),
        createI64Const(headSizeQk),
        createI64Const(headSizeV),
        LLVM::ConstantOp::create(
            rewriter, loc, f32Type,
            rewriter.getF32FloatAttr(op.getScale().convertToFloat())),
        createI64Const(qkvType.getElementType().getIntOrFloatBitWidth() / 8)};

    SmallVector<Type, 18> paramTypes(10, ptrType);
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

} // namespace

void populateGatedDeltaNetLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<GatedDeltaNetOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
