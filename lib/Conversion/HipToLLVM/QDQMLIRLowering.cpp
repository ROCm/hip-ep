/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lower hip.quantize_linear / hip.dequantize_linear to runtime wrappers.

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

template <typename OpTy>
struct QDQLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  const char *funcName;

  QDQLowering(const LLVMTypeConverter &converter, const char *name)
      : ConvertOpToLLVMPattern<OpTy>(converter), funcName(name) {}

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto scaleType = cast<MemRefType>(op.getScale().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t inputDtype = getHipdnnDataType(inputType.getElementType());
    int64_t scaleDtype = getHipdnnDataType(scaleType.getElementType());
    int64_t outputDtype = getHipdnnDataType(outputType.getElementType());
    if (inputDtype < 0 || scaleDtype < 0 || outputDtype < 0)
      return rewriter.notifyMatchFailure(op, "unsupported Q/DQ element type");

    Value zeroPointPtr =
        extractMemRefPtr(adaptor.getZeroPoint(), rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value numElements =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    SmallVector<Type, 11> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, funcName, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 11> args = {
        adaptor.getCtx(),
        extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getScale(), rewriter, loc),
        zeroPointPtr,
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        numElements,
        i64Const(inputDtype),
        i64Const(scaleDtype),
        i64Const(outputDtype),
        i64Const(op.getScaleNumElements()),
        i64Const(op.getInnerSize())};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQDQMLIRLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.add<QDQLowering<DequantizeLinearOp>>(converter,
                                                kWrapDequantizeLinear);
  patterns.add<QDQLowering<QuantizeLinearOp>>(converter, kWrapQuantizeLinear);
}

} // namespace hip
} // namespace mlir
