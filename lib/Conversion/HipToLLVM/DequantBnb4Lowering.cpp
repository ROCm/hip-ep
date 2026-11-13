/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.dequant_bnb4(ctx, packed, absmax, output)
//   -> wrap_dequant_bnb4(state, packed, absmax, output, N, K, block, quant)
struct DequantBnb4OpLowering : public ConvertOpToLLVMPattern<DequantBnb4Op> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DequantBnb4Op op, OpAdaptor adaptor,
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

    Value statePtr = adaptor.getCtx();
    Value packedPtr =
        extractContiguousMemRefPtr(adaptor.getPacked(), rewriter, loc);
    Value absmaxPtr =
        extractContiguousMemRefPtr(adaptor.getAbsmax(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapDequantBnb4, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr,
                                  packedPtr,
                                  absmaxPtr,
                                  outputPtr,
                                  createI64Const(op.getN()),
                                  createI64Const(op.getK()),
                                  createI64Const(op.getBlockSize()),
                                  createI64Const(op.getQuantType())};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateDequantBnb4LoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<DequantBnb4OpLowering>(converter);
}

} // namespace hip
} // namespace mlir
