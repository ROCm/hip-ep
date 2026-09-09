/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.qmatmul -> wrap_qmatmul(..., M_scale, A_zp, B_zp, Y_zp)
//
// The fused chain is DQ(A) @ DQ(B) -> Q, which over the integer accumulator
// `acc[m,n] = sum_k A[m,k]*B[k,n]` is
//
//   Y = saturate(round(M * (acc - z_b*rowA - z_a*colB + K*z_a*z_b)) + z_y)
//
// with M = s_a * s_b / s_y. Only M is compile-time foldable, and folding it
// here is what keeps the kernel down to one float multiply and one round per
// output element instead of three dequant/requant divisions. The zero-point
// correction terms depend on the operand data, so they stay in the kernel.
struct QMatMulOpLowering : public ConvertOpToLLVMPattern<QMatMulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QMatMulOp op, OpAdaptor adaptor,
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

    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());
    auto YType = cast<MemRefType>(op.getY().getType());

    // A and Y may be 8- or 16-bit; B must be 8-bit. The kernel handles a 16-bit
    // A by splitting it into two bytes and running one 8-bit dot per byte, and
    // only A is split, so a wider B has no path through it. PDL fusion already
    // refuses to build such a hip.qmatmul, so this is the backstop for
    // hand-written IR.
    for (Type elemType : {AType.getElementType(), YType.getElementType()}) {
      if (!elemType.isInteger(8) && !elemType.isInteger(16))
        return rewriter.notifyMatchFailure(
            op, "expected 8- or 16-bit A and Y element types");
    }
    if (!BType.getElementType().isInteger(8))
      return rewriter.notifyMatchFailure(op,
                                         "expected an 8-bit B element type");
    int64_t aDataType = getHipdnnDataType(AType.getElementType());
    int64_t bDataType = getHipdnnDataType(BType.getElementType());
    int64_t yDataType = getHipdnnDataType(YType.getElementType());

    float yScale = op.getYScale().convertToFloat();
    if (yScale == 0.0f)
      return rewriter.notifyMatchFailure(op, "Y_scale must be non-zero");
    // Accumulate the product in double so the single narrowing to f32 is the
    // only rounding the folded coefficient carries.
    double mScale = (static_cast<double>(op.getAScale().convertToFloat()) *
                     static_cast<double>(op.getBScale().convertToFloat())) /
                    static_cast<double>(yScale);
    Value mScaleValue = LLVM::ConstantOp::create(
        rewriter, loc, f32Type,
        rewriter.getF32FloatAttr(static_cast<float>(mScale)));

    int64_t ARank = AType.getRank();
    int64_t BRank = BType.getRank();
    int64_t transA = op.getTransA();
    int64_t transB = op.getTransB();
    MemRefDescriptor ADesc(adaptor.getA());
    MemRefDescriptor BDesc(adaptor.getB());

    // A: [..., M, K] or [..., K, M] when transA; B: [..., K, N] or [..., N, K]
    // when transB. M/N/K are the logical extents either way, and the kernel
    // gets the flags so it can pick the matching load stride.
    Value M = (ARank >= 2) ? (transA ? ADesc.size(rewriter, loc, ARank - 1)
                                     : ADesc.size(rewriter, loc, ARank - 2))
                           : createI64Const(1);
    Value K = transA ? ADesc.size(rewriter, loc, ARank - 2)
                     : ADesc.size(rewriter, loc, ARank - 1);
    Value N = transB ? BDesc.size(rewriter, loc, BRank - 2)
                     : BDesc.size(rewriter, loc, BRank - 1);

    Value batchCount;
    if (ARank == 2) {
      batchCount = createI64Const(1);
    } else {
      batchCount = ADesc.size(rewriter, loc, 0);
      for (int64_t i : llvm::seq<int64_t>(1, ARank - 2)) {
        Value dim = ADesc.size(rewriter, loc, i);
        batchCount = LLVM::MulOp::create(rewriter, loc, batchCount, dim);
      }
    }

    // b_batch_stride: per-batch element stride into B. Three shapes reach here
    // with batch_count > 1 and need different strides:
    //   * rank-2 weight                              -> one matrix -> 0
    //   * rank-N weight whose leading dims multiply   -> K * N
    //   * rank-N weight whose leading dims are all 1  -> still one matrix -> 0
    // transB only swaps the trailing two extents, so the per-matrix element
    // count stays K * N either way.
    // The leading-one case ([1, K, N]) is why this is not simply
    // `BRank > 2 ? K*N : 0`: treating it as per-batch reads K*N elements past
    // the end of the weight on every batch after the first.
    Value bBatchStride;
    if (BRank == 2) {
      bBatchStride = createI64Const(0);
    } else {
      bool allLeadingStatic = true;
      int64_t staticLeadingProduct = 1;
      for (int64_t i : llvm::seq<int64_t>(0, BRank - 2)) {
        if (BType.isDynamicDim(i)) {
          allLeadingStatic = false;
          break;
        }
        staticLeadingProduct *= BType.getDimSize(i);
      }
      if (allLeadingStatic) {
        bBatchStride = (staticLeadingProduct <= 1)
                           ? createI64Const(0)
                           : LLVM::MulOp::create(rewriter, loc, K, N).getRes();
      } else {
        Value one = createI64Const(1);
        Value zero = createI64Const(0);
        Value leadingProduct = one;
        for (int64_t i : llvm::seq<int64_t>(0, BRank - 2)) {
          Value dim = BDesc.size(rewriter, loc, i);
          leadingProduct =
              LLVM::MulOp::create(rewriter, loc, leadingProduct, dim).getRes();
        }
        Value isBroadcast = LLVM::ICmpOp::create(
            rewriter, loc, LLVM::ICmpPredicate::sle, leadingProduct, one);
        Value kn = LLVM::MulOp::create(rewriter, loc, K, N).getRes();
        bBatchStride =
            LLVM::SelectOp::create(rewriter, loc, isBroadcast, zero, kn)
                .getRes();
      }
    }

    // Runtime signature:
    // int wrap_qmatmul(RuntimeState* state,
    //                  const void* A, const void* B, void* Y,
    //                  int64_t M, int64_t N, int64_t K,
    //                  int64_t batch_count, int64_t b_batch_stride,
    //                  int64_t trans_a, int64_t trans_b,
    //                  int64_t a_data_type, int64_t b_data_type,
    //                  int64_t y_data_type, float M_scale,
    //                  int64_t A_zero_point, int64_t B_zero_point,
    //                  int64_t Y_zero_point)
    SmallVector<Type, 18> paramTypes = {
        ptrType, // state
        ptrType, // A
        ptrType, // B
        ptrType, // Y
        i64Type, // M
        i64Type, // N
        i64Type, // K
        i64Type, // batch_count
        i64Type, // b_batch_stride
        i64Type, // trans_a
        i64Type, // trans_b
        i64Type, // a_data_type
        i64Type, // b_data_type
        i64Type, // y_data_type
        f32Type, // M_scale
        i64Type, // A_zero_point
        i64Type, // B_zero_point
        i64Type  // Y_zero_point
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQMatMul, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 18> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getB(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc),
        M,
        N,
        K,
        batchCount,
        bBatchStride,
        createI64Const(transA),
        createI64Const(transB),
        createI64Const(aDataType),
        createI64Const(bDataType),
        createI64Const(yDataType),
        mScaleValue,
        createI64Const(op.getAZeroPoint()),
        createI64Const(op.getBZeroPoint()),
        createI64Const(op.getYZeroPoint())};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQMatMulLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns) {
  patterns.add<QMatMulOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
