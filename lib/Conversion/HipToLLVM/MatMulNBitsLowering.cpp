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
//
//   hip.matmul_nbits_group(%ctx) ins(%A) weights(%B0, %B1) scales(%S0, %S1)
//       outs(%Y0, %Y1) {K, Ns = [N0, N1], ...}
//   -> wrap_matmul_nbits_group(state, slot, A, count,
//                              B0, S0, Y0, N0, B1, S1, Y1, N1,
//                              null, null, null, 0, null, null, null, 0,
//                              M, K, batch_count, bits, block_size, elem_size)
//
// The member slots are fixed at kMaxMembers so the ABI needs no arrays;
// unused slots pass null pointers and N = 0.
//===----------------------------------------------------------------------===//

struct MatMulNBitsGroupOpLowering
    : public ConvertOpToLLVMPattern<MatMulNBitsGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

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

    auto AType = cast<MemRefType>(op.getA().getType());
    int64_t ARank = AType.getRank();
    int64_t elemSize = AType.getElementType().getIntOrFloatBitWidth() / 8;
    Value m = (ARank >= 2) ? getMemRefDimSize(AType, ARank - 2, adaptor.getA(),
                                              rewriter, loc)
                           : createI64Const(1);
    Value batch = createI64Const(1);
    for (int64_t i = 0; i < ARank - 2; ++i)
      batch = LLVM::MulOp::create(
          rewriter, loc, batch,
          getMemRefDimSize(AType, i, adaptor.getA(), rewriter, loc));

    const unsigned count = op.getOutputs().size();
    SmallVector<Type> paramTypes = {ptrType, i32Type, ptrType, i64Type};
    SmallVector<Value> args = {adaptor.getHandle(),
                               getOpStateSlotValue(op, rewriter, loc),
                               extractContiguousMemRefPtr(adaptor.getA(),
                                                          rewriter, loc),
                               createI64Const(count)};
    for (unsigned i = 0; i < MatMulNBitsGroupOp::kMaxMembers; ++i) {
      paramTypes.append({ptrType, ptrType, ptrType, i64Type});
      if (i < count) {
        args.push_back(
            extractContiguousMemRefPtr(adaptor.getB()[i], rewriter, loc));
        args.push_back(
            extractContiguousMemRefPtr(adaptor.getScales()[i], rewriter, loc));
        args.push_back(
            extractContiguousMemRefPtr(adaptor.getOutputs()[i], rewriter, loc));
        args.push_back(createI64Const(op.getNs()[i]));
      } else {
        for (int j = 0; j < 3; ++j)
          args.push_back(extractOptionalMemRefPtr(Value(), rewriter, loc));
        args.push_back(createI64Const(0));
      }
    }
    paramTypes.append(6, i64Type);
    args.append({m, createI64Const(op.getK()), batch,
                 createI64Const(op.getBits()),
                 createI64Const(op.getBlockSize()), createI64Const(elemSize)});

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
