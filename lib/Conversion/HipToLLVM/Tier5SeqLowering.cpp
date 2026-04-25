/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers Tier 5 sequence/recurrent hip ops to wrap_* runtime calls:
//   - hip.cumsum -> wrap_cumsum
//
// LSTM/STFT live in their own lowering files (TODO).

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

struct CumSumLowering : public ConvertOpToLLVMPattern<CumSumOp> {
  using ConvertOpToLLVMPattern<CumSumOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CumSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value inPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType || !inType.hasStaticShape() ||
        !outType.hasStaticShape())
      return rewriter.notifyMatchFailure(
          op, "hip.cumsum lowering requires static shapes");

    int64_t rank = inType.getRank();
    int64_t axis = op.getAxis();
    if (axis < 0 || axis >= rank)
      return rewriter.notifyMatchFailure(op, "hip.cumsum axis out of range");

    int64_t outer = 1;
    for (int64_t d = 0; d < axis; ++d)
      outer *= inType.getDimSize(d);
    int64_t axisSize = inType.getDimSize(axis);
    int64_t inner = 1;
    for (int64_t d = axis + 1; d < rank; ++d)
      inner *= inType.getDimSize(d);

    int64_t dataType = getHipdnnDataType(inType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.cumsum unsupported element type");

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };
    Value outerVal = i64Const(outer);
    Value axisVal = i64Const(axisSize);
    Value innerVal = i64Const(inner);
    Value dtypeVal = i64Const(dataType);
    Value exclVal = i64Const(op.getExclusive());
    Value revVal = i64Const(op.getReverse());

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, i64Type, i64Type, i64Type,
                                       i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCumSum, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr, inPtr,    outPtr,  outerVal,
                                  axisVal,  innerVal, dtypeVal, exclVal,
                                  revVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateTier5SeqLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<CumSumLowering>(converter);
}

} // namespace hip
} // namespace mlir
