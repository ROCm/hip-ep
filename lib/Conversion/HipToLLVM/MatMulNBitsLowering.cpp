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
    Value aQuantPtr =
        extractOptionalMemRefPtr(adaptor.getAQuant(), rewriter, loc);
    Value aScalePtr =
        extractOptionalMemRefPtr(adaptor.getAScale(), rewriter, loc);
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

    SmallVector<Type, 20> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // A
        ptrType, // B
        ptrType, // scales
        ptrType, // zero_points (nullable)
        ptrType, // g_idx (nullable)
        ptrType, // bias (nullable)
        ptrType, // a_quant (nullable)
        ptrType, // a_scale (nullable)
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

    SmallVector<Value, 20> args = {statePtr,
                                   getOpStateSlotValue(op, rewriter, loc),
                                   APtr,
                                   BPtr,
                                   scalesPtr,
                                   zeroPointsPtr,
                                   gIdxPtr,
                                   biasPtr,
                                   aQuantPtr,
                                   aScalePtr,
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
// QuantizeAct Lowering
//===----------------------------------------------------------------------===//

struct QuantizeActOpLowering : public ConvertOpToLLVMPattern<QuantizeActOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QuantizeActOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto AType = cast<MemRefType>(op.getA().getType());
    int64_t ARank = AType.getRank();
    Value rows = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                          rewriter.getI64IntegerAttr(1));
    for (int64_t i = 0; i < ARank - 1; ++i)
      rows = LLVM::MulOp::create(
          rewriter, loc, rows,
          getMemRefDimSize(AType, i, adaptor.getA(), rewriter, loc));

    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQuantizeAct, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {
        adaptor.getHandle(),
        extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getAQuant(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getAScale(), rewriter, loc),
        rows,
        LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            rewriter.getI64IntegerAttr(AType.getShape().back())),
        LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            rewriter.getI64IntegerAttr(op.getBlockSize()))};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMulNBitsGroup Lowering
//===----------------------------------------------------------------------===//

/// Segments go in fixed slots of three (null / 0 when absent). With the
/// silu_mul epilogue, up_tmp fills the second output slot: it is where the
/// per-segment path writes the second projection.
struct MatMulNBitsGroupOpLowering
    : public ConvertOpToLLVMPattern<MatMulNBitsGroupOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  static constexpr int64_t kMaxSegments = 3;

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

    int64_t numSegments = static_cast<int64_t>(op.getWeights().size());
    if (numSegments > kMaxSegments)
      return rewriter.notifyMatchFailure(op, "more than 3 segments");

    SmallVector<Value> outputs(adaptor.getOutputs());
    if (Value upTmp = adaptor.getUpTmp())
      outputs.push_back(upTmp);
    auto slots = [&](ValueRange vals) {
      SmallVector<Value, 3> ptrs;
      for (int64_t i : llvm::seq<int64_t>(0, kMaxSegments))
        ptrs.push_back(extractOptionalMemRefPtr(
            i < static_cast<int64_t>(vals.size()) ? vals[i] : Value(), rewriter,
            loc));
      return ptrs;
    };
    SmallVector<Value, 3> weightPtrs = slots(adaptor.getWeights());
    SmallVector<Value, 3> scalePtrs = slots(adaptor.getScales());
    SmallVector<Value, 3> zpPtrs = slots(adaptor.getZeroPoints());
    SmallVector<Value, 3> outPtrs = slots(outputs);

    auto AType = cast<MemRefType>(op.getA().getType());
    int64_t ARank = AType.getRank();
    Value m = (ARank >= 2) ? getMemRefDimSize(AType, ARank - 2, adaptor.getA(),
                                              rewriter, loc)
                           : createI64Const(1);
    Value batch = createI64Const(1);
    for (int64_t i : llvm::seq<int64_t>(0, ARank - 2))
      batch = LLVM::MulOp::create(
          rewriter, loc, batch,
          getMemRefDimSize(AType, i, adaptor.getA(), rewriter, loc));

    ArrayAttr nAttr = op.getN();
    SmallVector<Value, 3> ns;
    for (int64_t i : llvm::seq<int64_t>(0, kMaxSegments))
      ns.push_back(createI64Const(
          i < numSegments ? cast<IntegerAttr>(nAttr[i]).getInt() : 0));

    SmallVector<Type, 26> paramTypes = {ptrType, i32Type, ptrType,
                                        ptrType, ptrType, i64Type};
    paramTypes.append(4 * kMaxSegments, ptrType); // B, scales, zp, out
    paramTypes.append(kMaxSegments, i64Type);     // N
    paramTypes.append(5, i64Type); // M, K, batch, block_size, epilogue
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMatMulNBitsGroup, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 26> args = {
        adaptor.getHandle(),
        getOpStateSlotValue(op, rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getA(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getAQuant(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getAScale(), rewriter, loc),
        createI64Const(numSegments)};
    llvm::append_range(args, weightPtrs);
    llvm::append_range(args, scalePtrs);
    llvm::append_range(args, zpPtrs);
    llvm::append_range(args, outPtrs);
    llvm::append_range(args, ns);
    args.append({m, createI64Const(op.getK()), batch,
                 createI64Const(op.getBlockSize()),
                 createI64Const(op.getEpilogue())});

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMatMulNBitsLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<MatMulNBitsOpLowering, QuantizeActOpLowering,
               MatMulNBitsGroupOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
