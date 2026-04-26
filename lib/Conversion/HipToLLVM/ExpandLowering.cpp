/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//
// Lowers hip.expand (ONNX Expand: numpy-style broadcast) to wrap_expand.
//
// Reads input and output shapes from memref descriptors at runtime,
// computes row-major strides via SSA arithmetic, and passes them to
// wrap_expand.  Supports both static and dynamic dimensions.

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"

namespace mlir {
namespace hip {
namespace {

struct ExpandLowering : public ConvertOpToLLVMPattern<ExpandOp> {
  using ConvertOpToLLVMPattern<ExpandOp>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ExpandOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inType = dyn_cast<MemRefType>(op.getInput().getType());
    auto outType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inType || !outType)
      return rewriter.notifyMatchFailure(
          op, "hip.expand expects ranked memref operands");

    int64_t inRank = inType.getRank();
    int64_t outRank = outType.getRank();
    if (inRank > outRank)
      return rewriter.notifyMatchFailure(
          op, "hip.expand input rank exceeds output rank");

    int64_t dataType = getHipdnnDataType(outType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "hip.expand unsupported element type");

    auto inShape =
        getMemRefShape(inType, adaptor.getInput(), rewriter, loc);
    auto outShape =
        getMemRefShape(outType, adaptor.getOutput(), rewriter, loc);

    auto i64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // Row-major contiguous strides (SSA).
    SmallVector<Value> inStrides(inRank);
    Value acc = i64Const(1);
    for (int64_t d = inRank - 1; d >= 0; --d) {
      inStrides[d] = acc;
      acc = LLVM::MulOp::create(rewriter, loc, acc, inShape[d]);
    }

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value inShapePtr =
        materialiseValueArray(inShape, i64Type, ptrType, rewriter, loc);
    Value inStridePtr =
        materialiseValueArray(inStrides, i64Type, ptrType, rewriter, loc);
    Value outShapePtr =
        materialiseValueArray(outShape, i64Type, ptrType, rewriter, loc);
    Value inRankVal = i64Const(inRank);
    Value outRankVal = i64Const(outRank);
    Value dtypeVal = i64Const(dataType);

    SmallVector<Type, 9> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       ptrType, ptrType, i64Type, i64Type,
                                       i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapExpand, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr,    inputPtr,    outputPtr,
                                  inShapePtr,  inStridePtr, outShapePtr,
                                  inRankVal,   outRankVal,  dtypeVal};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateExpandLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.insert<ExpandLowering>(converter);
}

} // namespace hip
} // namespace mlir
