/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.gemm(handle, input_A, input_B, input_C, output, alpha, beta, transA,
// transB, typeCode)
struct GemmOpLowering : public ConvertOpToLLVMPattern<GemmOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GemmOp op, OpAdaptor adaptor,
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

    auto AType = cast<MemRefType>(op.getInputA().getType());
    Type elemType = AType.getElementType();
    int64_t typeCode;
    if (elemType.isF16())
      typeCode = 0;
    else if (elemType.isF32())
      typeCode = 1;
    else if (elemType.isF64())
      typeCode = 2;
    else if (elemType.isBF16())
      typeCode = 3;
    else
      return failure();
    Value typeCodeVal = createI64Const(typeCode);

    Value statePtr = adaptor.getCtx();
    Value input_A_ptr =
        extractContiguousMemRefPtr(adaptor.getInputA(), rewriter, loc);
    Value input_B_ptr =
        extractContiguousMemRefPtr(adaptor.getInputB(), rewriter, loc);
    Value input_C_ptr =
        extractOptionalMemRefPtr(adaptor.getInputC(), rewriter, loc);
    Value output_ptr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value alpha = LLVM::ConstantOp::create(
        rewriter, loc, f32Type,
        rewriter.getF32FloatAttr(adaptor.getAlpha().convertToFloat()));
    Value beta = LLVM::ConstantOp::create(
        rewriter, loc, f32Type,
        rewriter.getF32FloatAttr(adaptor.getBeta().convertToFloat()));
    Value transA = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(adaptor.getTransA()));
    Value transB = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(adaptor.getTransB()));

    // A.shape = transA ? [K, M] : [M, K]
    // B.shape = transB ? [N, K] : [K, N]
    Value M, K, N;
    if (adaptor.getTransA()) {
      K = getMemRefDimSize(AType, 0, adaptor.getInputA(), rewriter, loc);
      M = getMemRefDimSize(AType, 1, adaptor.getInputA(), rewriter, loc);
    } else {
      M = getMemRefDimSize(AType, 0, adaptor.getInputA(), rewriter, loc);
      K = getMemRefDimSize(AType, 1, adaptor.getInputA(), rewriter, loc);
    }
    auto BType = cast<MemRefType>(op.getInputB().getType());
    if (adaptor.getTransB()) {
      N = getMemRefDimSize(BType, 0, adaptor.getInputB(), rewriter, loc);
    } else {
      N = getMemRefDimSize(BType, 1, adaptor.getInputB(), rewriter, loc);
    }

    // Extract C's shape normalized to 2D [cDim0, cDim1] for broadcast support.
    // Scalar/absent → [0,0], 1D [X] → [1,X], 2D [X,Y] → [X,Y].
    Value cDim0, cDim1;
    if (adaptor.getInputC()) {
      auto CType = cast<MemRefType>(op.getInputC().getType());
      int cRank = CType.getRank();
      if (cRank == 0) {
        cDim0 = createI64Const(1);
        cDim1 = createI64Const(1);
      } else if (cRank == 1) {
        cDim0 = createI64Const(1);
        cDim1 = getMemRefDimSize(CType, 0, adaptor.getInputC(), rewriter, loc);
      } else {
        cDim0 = getMemRefDimSize(CType, 0, adaptor.getInputC(), rewriter, loc);
        cDim1 = getMemRefDimSize(CType, 1, adaptor.getInputC(), rewriter, loc);
      }
    } else {
      cDim0 = createI64Const(0);
      cDim1 = createI64Const(0);
    }

    SmallVector<Type, 16> paramTypes = {
        ptrType, // state
        ptrType, // A
        ptrType, // B
        ptrType, // C (nullable)
        ptrType, // output
        i64Type, // M
        i64Type, // N
        i64Type, // K
        f32Type, // alpha
        f32Type, // beta
        i64Type, // transA
        i64Type, // transB
        i64Type, // typeCode
        i64Type, // cDim0
        i64Type, // cDim1
        i32Type, // op_state_slot
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGemm, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 16> args = {
        statePtr,    input_A_ptr,
        input_B_ptr, input_C_ptr,
        output_ptr,  M,
        N,           K,
        alpha,       beta,
        transA,      transB,
        typeCodeVal, cDim0,
        cDim1,       getOpStateSlotValue(op, rewriter, loc)};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateGemmLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<GemmOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
