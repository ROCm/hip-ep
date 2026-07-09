/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.ms_matmul_nbits_i2_fused lowering
//===----------------------------------------------------------------------===//
//
// Before:
//   hip.ms_matmul_nbits_i2_fused(%ctx)
//       ins(%input, %B, %w_scales : ...) w_zp(...) dq(...) q(...)
//       outs(%out : ...)
//       {N=17920, K=5120, block_size=64}
//
// After:
//   llvm.call @wrap_ms_matmul_nbits_i2_fused(
//       %state, %input, %B, %w_scales, %w_zp (nullable),
//       %dq_scale, %dq_zp (nullable), %q_scale, %q_zp (nullable),
//       %output, M, N, K, block_size)

static constexpr const char *kWrapMsMatMulNBitsI2Fused =
    "wrap_ms_matmul_nbits_i2_fused";

struct MsMatMulNBitsI2FusedOpLowering
    : public ConvertOpToLLVMPattern<MsMatMulNBitsI2FusedOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MsMatMulNBitsI2FusedOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();

    auto inputType  = cast<MemRefType>(op.getInput().getType());

    Value statePtr   = adaptor.getCtx();
    Value inputPtr   = extractContiguousMemRefPtr(adaptor.getInput(),   rewriter, loc);
    Value bPtr       = extractContiguousMemRefPtr(adaptor.getB(),       rewriter, loc);
    Value wscalesPtr = extractContiguousMemRefPtr(adaptor.getWScales(), rewriter, loc);
    Value wzpPtr     = extractOptionalMemRefPtr(adaptor.getWZp(),       rewriter, loc);
    Value dqscalePtr = extractContiguousMemRefPtr(adaptor.getDqScale(), rewriter, loc);
    Value dqzpPtr    = extractOptionalMemRefPtr(adaptor.getDqZp(),      rewriter, loc);
    Value qscalePtr  = extractContiguousMemRefPtr(adaptor.getQScale(),  rewriter, loc);
    Value qzpPtr     = extractOptionalMemRefPtr(adaptor.getQZp(),       rewriter, loc);
    Value outputPtr  = extractContiguousMemRefPtr(adaptor.getOutput(),  rewriter, loc);

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // M: first dynamic dim of input tensor (batch*M rows)
    int64_t rank = inputType.getRank();
    Value M = createI64(1);
    for (int r = 0; r < rank - 1; ++r) {
      Value dim = getMemRefDimSize(inputType, r, adaptor.getInput(), rewriter, loc);
      M = LLVM::MulOp::create(rewriter, loc, i64Type, M, dim);
    }

    Value N          = createI64(op.getN());
    Value K          = createI64(op.getK());
    Value block_size = createI64(op.getBlockSize());

    SmallVector<Type, 14> paramTypes = {
        ptrType,  // state
        ptrType, ptrType, ptrType, ptrType,  // input, B, w_scales, w_zp
        ptrType, ptrType, ptrType, ptrType,  // dq_scale, dq_zp, q_scale, q_zp
        ptrType,  // output
        i64Type, i64Type, i64Type, i64Type   // M, N, K, block_size
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMsMatMulNBitsI2Fused, paramTypes,
        rewriter.getI32Type());
    if (failed(funcOp)) return failure();

    SmallVector<Value, 14> args = {
        statePtr, inputPtr, bPtr, wscalesPtr, wzpPtr,
        dqscalePtr, dqzpPtr, qscalePtr, qzpPtr,
        outputPtr, M, N, K, block_size};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMsMatMulNBitsI2FusedLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MsMatMulNBitsI2FusedOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
