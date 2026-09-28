/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.qgemm -> wrap_qgemm(..., M_ab, M_c, A_zp, B_zp, C_zp, Y_zp)
//
// The fused chain is Q(alpha * DQ(A)' @ DQ(B)' + beta * DQ(C)), which over the
// integer accumulator `acc[m,n] = sum_k A[m,k]*B[k,n]` is
//
//   Y = saturate(round(M_ab * (acc - z_b*rowA - z_a*colB + K*z_a*z_b)
//                      + M_c * (C - z_c))
//                + z_y)
//
// with M_ab = alpha*s_a*s_b/s_y and M_c = beta*s_c/s_y. Both are compile-time
// constants, and folding them here is what keeps the kernel to one float
// multiply-add and one round per output element instead of four dequant and
// requant divisions. The zero-point corrections depend on the operand data, so
// they stay in the kernel.
//
// A per-channel B leaves s_b out of M_ab on its own: the B_scale attribute is
// then at its 1.0 identity, and the kernel applies B_scales[n] instead. That
// costs one extra float multiply per output element, which is what a
// coefficient varying with n is worth.
//
// Before:
//   hip.qgemm(%ctx) ins(%A, %B : memref<8x32xui16, 1>, memref<16x32xi8, 1>)
//                   b_per_channel(%Bs, %Bzp : memref<16xf32, 1>,
//                                             memref<16xi8, 1>)
//                   outs(%Y : memref<8x16xui16, 1>)
//                   {A_scale = 0.25, A_zero_point = 32768, B_bits = 4,
//                    Y_scale = 0.5, Y_zero_point = 32768, transB = 1}
// After:
//   llvm.call @wrap_qgemm(%ctx, %A, %B, null, %Bs, %Bzp, %Y,
//                         8, 16, 32,      // M, N, K
//                         0, 1,           // trans_a, trans_b
//                         9, 5, -1, 9,    // a/b/c/y dtype; -1 is "no bias"
//                         4, 0, 0,        // b_bits, c_dim0, c_dim1
//                         0.5, 0.0,       // M_ab, M_c
//                         32768, 0, 0, 32768)
struct QGemmOpLowering : public ConvertOpToLLVMPattern<QGemmOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QGemmOp op, OpAdaptor adaptor,
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
    auto createF32Const = [&](double value) -> Value {
      return LLVM::ConstantOp::create(
          rewriter, loc, f32Type,
          rewriter.getF32FloatAttr(static_cast<float>(value)));
    };

    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());
    auto YType = cast<MemRefType>(op.getY().getType());

    // A and Y may be 8- or 16-bit; B must be 8-bit STORAGE, which a 4-bit
    // weight uses too by holding two nibbles per byte. The kernel keeps a
    // 16-bit A exact by splitting it into two bytes and running one 8-bit dot
    // per byte, and only A is split, so a wider B has no path through it. PDL
    // fusion already refuses to build such a hip.qgemm; this is the backstop
    // for hand-written IR.
    for (Type elemType : {AType.getElementType(), YType.getElementType()}) {
      if (!elemType.isInteger(8) && !elemType.isInteger(16))
        return rewriter.notifyMatchFailure(
            op, "expected 8- or 16-bit A and Y element types");
    }
    if (!BType.getElementType().isInteger(8))
      return rewriter.notifyMatchFailure(op,
                                         "expected an 8-bit B element type");
    int64_t bBits = op.getBBits();
    if (bBits != 4 && bBits != 8)
      return rewriter.notifyMatchFailure(op, "expected B_bits of 4 or 8");

    float yScale = op.getYScale().convertToFloat();
    if (yScale == 0.0f)
      return rewriter.notifyMatchFailure(op, "Y_scale must be non-zero");

    // Accumulate the products in double so the single narrowing to f32 is the
    // only rounding the folded coefficients carry.
    double invYScale = 1.0 / static_cast<double>(yScale);
    double mAb = static_cast<double>(op.getAlpha().convertToFloat()) *
                 static_cast<double>(op.getAScale().convertToFloat()) *
                 static_cast<double>(op.getBScale().convertToFloat()) *
                 invYScale;

    // Absent C zeroes the whole bias term rather than leaving a live beta
    // behind a null pointer, and reports no bias dtype at all.
    double mC = 0.0;
    int64_t cDataType = HIPDNN_EP_DATATYPE_UNSUPPORTED;
    Value cDim0;
    Value cDim1;
    if (Value c = op.getC()) {
      auto CType = cast<MemRefType>(c.getType());
      Type cElemType = CType.getElementType();
      if (!cElemType.isInteger(8) && !cElemType.isInteger(16) &&
          !cElemType.isInteger(32))
        return rewriter.notifyMatchFailure(
            op, "expected an 8-, 16- or 32-bit C element type");
      cDataType = getHipdnnDataType(cElemType);
      mC = static_cast<double>(op.getBeta().convertToFloat()) *
           static_cast<double>(op.getCScale().convertToFloat()) * invYScale;

      // C normalized to 2D for the unidirectional broadcast to [M, N], spelled
      // the same way as wrap_gemm: scalar -> [1, 1], [X] -> [1, X],
      // [X, Y] -> [X, Y].
      int64_t cRank = CType.getRank();
      if (cRank == 0) {
        cDim0 = createI64Const(1);
        cDim1 = createI64Const(1);
      } else if (cRank == 1) {
        cDim0 = createI64Const(1);
        cDim1 = getMemRefDimSize(CType, 0, adaptor.getC(), rewriter, loc);
      } else {
        cDim0 = getMemRefDimSize(CType, 0, adaptor.getC(), rewriter, loc);
        cDim1 = getMemRefDimSize(CType, 1, adaptor.getC(), rewriter, loc);
      }
    } else {
      cDim0 = createI64Const(0);
      cDim1 = createI64Const(0);
    }

    int64_t transA = op.getTransA();
    int64_t transB = op.getTransB();
    // A: [M, K] or [K, M] when transA; B: [K, N] or [N, K] when transB. M/N/K
    // are the logical extents either way, and the kernel gets the flags so it
    // can pick the matching load stride.
    Value M =
        getMemRefDimSize(AType, transA ? 1 : 0, adaptor.getA(), rewriter, loc);
    Value K =
        getMemRefDimSize(AType, transA ? 0 : 1, adaptor.getA(), rewriter, loc);
    Value N =
        getMemRefDimSize(BType, transB ? 0 : 1, adaptor.getB(), rewriter, loc);

    // Runtime signature:
    // int wrap_qgemm(RuntimeState* state,
    //                const void* A, const void* B, const void* C,
    //                const void* B_scales, const void* B_zero_points, void* Y,
    //                int64_t M, int64_t N, int64_t K,
    //                int64_t trans_a, int64_t trans_b,
    //                int64_t a_data_type, int64_t b_data_type,
    //                int64_t c_data_type, int64_t y_data_type,
    //                int64_t b_bits, int64_t c_dim0, int64_t c_dim1,
    //                float M_ab, float M_c,
    //                int64_t A_zero_point, int64_t B_zero_point,
    //                int64_t C_zero_point, int64_t Y_zero_point)
    SmallVector<Type, 25> paramTypes = {
        ptrType, // state
        ptrType, // A
        ptrType, // B
        ptrType, // C (nullable)
        ptrType, // B_scales (nullable)
        ptrType, // B_zero_points (nullable)
        ptrType, // Y
        i64Type, // M
        i64Type, // N
        i64Type, // K
        i64Type, // trans_a
        i64Type, // trans_b
        i64Type, // a_data_type
        i64Type, // b_data_type
        i64Type, // c_data_type
        i64Type, // y_data_type
        i64Type, // b_bits
        i64Type, // c_dim0
        i64Type, // c_dim1
        f32Type, // M_ab
        f32Type, // M_c
        i64Type, // A_zero_point
        i64Type, // B_zero_point
        i64Type, // C_zero_point
        i64Type  // Y_zero_point
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQGemm, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 25> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getB(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getC(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getBScales(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getBZeroPoints(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getY(), rewriter, loc),
        M,
        N,
        K,
        createI64Const(transA),
        createI64Const(transB),
        createI64Const(getHipdnnDataType(AType.getElementType())),
        createI64Const(getHipdnnDataType(BType.getElementType())),
        createI64Const(cDataType),
        createI64Const(getHipdnnDataType(YType.getElementType())),
        createI64Const(bBits),
        cDim0,
        cDim1,
        createF32Const(mAb),
        createF32Const(mC),
        createI64Const(op.getAZeroPoint()),
        createI64Const(op.getBZeroPoint()),
        createI64Const(op.getCZeroPoint()),
        createI64Const(op.getYZeroPoint())};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQGemmLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<QGemmOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
