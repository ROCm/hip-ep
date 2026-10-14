/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.qlinear_matmul -> wrap_qlinear_matmul
//
// Before:
//   hip.qlinear_matmul(%ctx) ins(%a, %as, %az, %b, %bs, %bz, %ys, %yz :
//                           memref<1x4xui8, 1>, memref<f32, 1>,
//                           memref<ui8, 1>, memref<4x3xi8, 1>, ...)
//                      outs(%y : memref<1x3xui8, 1>)
// After:
//   llvm.call @wrap_qlinear_matmul(%ctx, %a, %as, %az, %b, %bs, %bz, %ys, %yz,
//                                  %y, 1, 4, 3, 7, 5, 7, 1, 1, 1, 1, 1, 1)
//
// M, K, and N come from the memref descriptors. Scale and zero-point lengths
// are element counts; this op only accepts the per-tensor count of 1.
struct QLinearMatMulOpLowering
    : public ConvertOpToLLVMPattern<QLinearMatMulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QLinearMatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto aType = dyn_cast<MemRefType>(op.getA().getType());
    auto bType = dyn_cast<MemRefType>(op.getB().getType());
    auto yType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!aType || !bType || !yType || aType.getRank() != 2 ||
        bType.getRank() != 2 || yType.getRank() != 2)
      return op.emitError("hip.qlinear_matmul: a, b, and output must be "
                          "rank-2 memrefs");

    int64_t aDtype = getHipdnnDataType(aType.getElementType());
    int64_t bDtype = getHipdnnDataType(bType.getElementType());
    int64_t yDtype = getHipdnnDataType(yType.getElementType());
    if (aDtype < 0 || bDtype < 0 || yDtype < 0)
      return op.emitError("hip.qlinear_matmul: element type the runtime "
                          "cannot name");

    auto createI64 = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value a = adaptor.getA();
    Value b = adaptor.getB();
    Value y = adaptor.getOutput();

    SmallVector<Type> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType,
        ptrType, ptrType, i64Type, i64Type, i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type,
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQLinearMatMul, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(a, rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getAScale(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getAZeroPoint(), rewriter, loc),
        extractContiguousMemRefPtr(b, rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getBScale(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getBZeroPoint(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getYScale(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getYZeroPoint(), rewriter, loc),
        extractContiguousMemRefPtr(y, rewriter, loc),
        getMemRefDimSize(aType, 0, a, rewriter, loc),
        getMemRefDimSize(aType, 1, a, rewriter, loc),
        getMemRefDimSize(bType, 1, b, rewriter, loc),
        createI64(aDtype),
        createI64(bDtype),
        createI64(yDtype),
        computeNumElements(cast<MemRefType>(op.getAScale().getType()),
                           adaptor.getAScale(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getBScale().getType()),
                           adaptor.getBScale(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getYScale().getType()),
                           adaptor.getYScale(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getAZeroPoint().getType()),
                           adaptor.getAZeroPoint(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getBZeroPoint().getType()),
                           adaptor.getBZeroPoint(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getYZeroPoint().getType()),
                           adaptor.getYZeroPoint(), rewriter, loc),
    };

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQLinearMatMulLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<QLinearMatMulOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
