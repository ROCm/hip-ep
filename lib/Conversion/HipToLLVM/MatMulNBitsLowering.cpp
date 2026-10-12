/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// MatMulNBits Lowering
//===----------------------------------------------------------------------===//

struct MatMulNBitsOpLowering : public ConvertOpToLLVMPattern<MatMulNBitsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatMulNBitsOp op, OpAdaptor adaptor,
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

    Value statePtr = adaptor.getHandle();
    Value APtr = extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc);
    Value BPtr = extractContiguousMemRefPtr(adaptor.getB(), rewriter, loc);
    Value scalesPtr =
        extractContiguousMemRefPtr(adaptor.getScales(), rewriter, loc);
    Value zeroPointsPtr =
        extractOptionalMemRefPtr(adaptor.getZeroPoints(), rewriter, loc);
    Value gIdxPtr = extractOptionalMemRefPtr(adaptor.getGIdx(), rewriter, loc);
    Value biasPtr = extractOptionalMemRefPtr(adaptor.getBias(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto AType = cast<MemRefType>(op.getA().getType());
    int64_t ARank = AType.getRank();
    int64_t elemSize = AType.getElementType().getIntOrFloatBitWidth() / 8;

    // A shape: [..., M, K] — M is the second-to-last dim
    Value m = (ARank >= 2) ? getMemRefDimSize(AType, ARank - 2, adaptor.getA(),
                                              rewriter, loc)
                           : createI64Const(1);
    // batch_count = product of all leading dimensions before M
    Value batch = createI64Const(1);
    for (int64_t i = 0; i < ARank - 2; ++i) {
      batch = LLVM::MulOp::create(
          rewriter, loc, batch,
          getMemRefDimSize(AType, i, adaptor.getA(), rewriter, loc));
    }

    Value n = createI64Const(op.getN());
    Value k = createI64Const(op.getK());
    Value bits = createI64Const(op.getBits());
    Value blockSize = createI64Const(op.getBlockSize());
    Value elemSizeVal = createI64Const(elemSize);
    Value zpElemSizeVal = createI64Const(op.getZpElemSize());
    Value scaleElemSizeVal = createI64Const(op.getScaleElemSize());

    SmallVector<Type, 18> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // A
        ptrType, // B
        ptrType, // scales
        ptrType, // zero_points (nullable)
        ptrType, // g_idx (nullable)
        ptrType, // bias (nullable)
        ptrType, // output
        i64Type, // M
        i64Type, // N
        i64Type, // K
        i64Type, // batch_count
        i64Type, // bits
        i64Type, // block_size
        i64Type, // elem_size
        i64Type, // zp_elem_size
        i64Type  // scale_elem_size
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMatMulNBits, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 18> args = {statePtr,
                                   getOpStateSlotValue(op, rewriter, loc),
                                   APtr,
                                   BPtr,
                                   scalesPtr,
                                   zeroPointsPtr,
                                   gIdxPtr,
                                   biasPtr,
                                   outputPtr,
                                   m,
                                   n,
                                   k,
                                   batch,
                                   bits,
                                   blockSize,
                                   elemSizeVal,
                                   zpElemSizeVal,
                                   scaleElemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMulNBitsGroup Lowering
//===----------------------------------------------------------------------===//

struct MatMulNBitsGroupOpLowering
    : public ConvertOpToLLVMPattern<MatMulNBitsGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static constexpr int kMaxSegments = 3;

  LogicalResult
  matchAndRewrite(MatMulNBitsGroupOp op, OpAdaptor adaptor,
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

    ValueRange weights = adaptor.getWeights();
    ValueRange scales = adaptor.getScales();
    ValueRange outputs = adaptor.getOutputs();
    ArrayAttr nAttr = op.getN();
    int64_t nseg = static_cast<int64_t>(weights.size());
    if (nseg < 1 || nseg > kMaxSegments)
      return rewriter.notifyMatchFailure(op, "unsupported segment count");

    auto segmentPtr = [&](ValueRange vals, int64_t i) -> Value {
      return extractOptionalMemRefPtr(
          i < static_cast<int64_t>(vals.size()) ? vals[i] : Value(), rewriter,
          loc);
    };

    auto AType = cast<MemRefType>(op.getA().getType());
    int64_t ARank = AType.getRank();
    int64_t elemSize = AType.getElementType().getIntOrFloatBitWidth() / 8;
    Value m = (ARank >= 2) ? getMemRefDimSize(AType, ARank - 2, adaptor.getA(),
                                              rewriter, loc)
                           : createI64Const(1);
    Value batch = createI64Const(1);
    for (int64_t i = 0; i < ARank - 2; ++i) {
      batch = LLVM::MulOp::create(
          rewriter, loc, batch,
          getMemRefDimSize(AType, i, adaptor.getA(), rewriter, loc));
    }

    SmallVector<Type, 24> paramTypes = {ptrType, i32Type, ptrType, i64Type};
    SmallVector<Value, 24> args = {
        adaptor.getHandle(), getOpStateSlotValue(op, rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc),
        createI64Const(nseg)};
    for (ValueRange vals : {weights, scales, outputs}) {
      for (int64_t i = 0; i < kMaxSegments; ++i) {
        paramTypes.push_back(ptrType);
        args.push_back(segmentPtr(vals, i));
      }
    }
    for (int64_t i = 0; i < kMaxSegments; ++i) {
      paramTypes.push_back(i64Type);
      args.push_back(createI64Const(
          i < nseg ? cast<IntegerAttr>(nAttr[i]).getInt() : 0));
    }
    // M, K, batch_count, bits, block_size, elem_size, scale_elem_size,
    // epilogue
    paramTypes.append(8, i64Type);
    args.append({m, createI64Const(op.getK()), batch,
                 createI64Const(op.getBits()),
                 createI64Const(op.getBlockSize()), createI64Const(elemSize),
                 createI64Const(op.getScaleElemSize()),
                 createI64Const(op.getEpilogue())});
    // a_quant, a_scale
    paramTypes.append(2, ptrType);
    args.push_back(extractOptionalMemRefPtr(adaptor.getAQuant(), rewriter, loc));
    args.push_back(extractOptionalMemRefPtr(adaptor.getAScale(), rewriter, loc));

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMatMulNBitsGroup, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMatMulNBitsLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<MatMulNBitsOpLowering, MatMulNBitsGroupOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
