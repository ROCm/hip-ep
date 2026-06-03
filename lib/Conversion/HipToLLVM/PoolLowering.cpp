/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.pool -> wrap_pool runtime call
//===----------------------------------------------------------------------===//
//
// Generic 1D / 2D / 3D window-pool lowering shared by MaxPool / AveragePool /
// LpPool.  spatial_rank in {1, 2, 3} is derived from `input`'s rank
// (= 2 + spatial_rank).  All shape-derived scalar params come from the
// memref descriptors so dynamic N is honored; spatial dims are required
// static at conversion time (see PoolConversion.cpp), so they fold to
// constants here.  The reduction kind is carried verbatim in `pool_mode`
// (0 = AVERAGE, 1 = MAX, 2 = LP) and resolved by the runtime kernel.
//
// Runtime ABI (matches `wrap_pool` in real/pool.cpp):
//   wrap_pool(state, input, output, indices, /* indices nullable, MAX only */
//             data_type,
//             pool_mode,
//             spatial_rank,
//             N, C,
//             in0, in1, in2,         // unused dims pass 1
//             out0, out1, out2,
//             k0, k1, k2,
//             s0, s1, s2,
//             p0, p1, p2,            // pad_begin per axis (pad_end is
//                                      implied by output shape, so we
//                                      do not pass pad_end)
//             dil0, dil1, dil2,
//             storage_order, ceil_mode, has_indices,
//             count_include_pad, p)  // AVERAGE / LP extras
//   -> i32

struct PoolOpLowering : public ConvertOpToLLVMPattern<PoolOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(PoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t rank = inputType.getRank();
    if (rank < 3 || rank > 5)
      return rewriter.notifyMatchFailure(op, "expected rank in [3, 5]");
    int64_t spatialRank = rank - 2;

    auto outputs = op.getOutputs();
    if (outputs.size() < 1 || outputs.size() > 2)
      return rewriter.notifyMatchFailure(op, "expected 1 or 2 DPS outputs");
    bool hasIndices = (outputs.size() == 2);
    auto outputType = cast<MemRefType>(outputs[0].getType());
    if (outputType.getRank() != rank)
      return rewriter.notifyMatchFailure(op, "output rank mismatch");

    // Element type of input (and output) drives the runtime dtype.
    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0 || (dataType > 2 && dataType != 6))
      return rewriter.notifyMatchFailure(
          op, "pool: only f16 / f32 / bf16 / f64 supported");

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutputs()[0], rewriter, loc);
    Value indicesPtr;
    if (hasIndices) {
      auto idxType = cast<MemRefType>(outputs[1].getType());
      if (!idxType.getElementType().isInteger(64))
        return rewriter.notifyMatchFailure(op, "indices must be i64");
      indicesPtr =
          extractContiguousMemRefPtr(adaptor.getOutputs()[1], rewriter, loc);
    } else {
      indicesPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    }

    Value inputDesc = adaptor.getInput();
    Value outputDesc = adaptor.getOutputs()[0];

    Value batchN =
        getMemRefDimSize(inputType, 0, inputDesc, rewriter, loc);
    Value channelC =
        getMemRefDimSize(inputType, 1, inputDesc, rewriter, loc);

    SmallVector<Value, 3> inSpatial(3, createI64(1));
    SmallVector<Value, 3> outSpatial(3, createI64(1));
    for (int64_t i : llvm::seq<int64_t>(spatialRank)) {
      inSpatial[i] =
          getMemRefDimSize(inputType, 2 + i, inputDesc, rewriter, loc);
      outSpatial[i] =
          getMemRefDimSize(outputType, 2 + i, outputDesc, rewriter, loc);
    }

    auto attrToOnes3 = [&](mlir::ArrayAttr attr) -> SmallVector<Value, 3> {
      SmallVector<Value, 3> out(3, createI64(1));
      for (int64_t i = 0; i < spatialRank; ++i)
        out[i] = createI64(cast<IntegerAttr>(attr[i]).getInt());
      return out;
    };

    SmallVector<Value, 3> kernel = attrToOnes3(op.getKernelShape());
    SmallVector<Value, 3> strides = attrToOnes3(op.getStrides());
    SmallVector<Value, 3> dilations = attrToOnes3(op.getDilations());

    // pads: layout is [x1_begin, ..., xk_begin, x1_end, ..., xk_end].  We
    // only need pad_begin per axis at runtime — pad_end is implicit in
    // the output extent.
    SmallVector<Value, 3> padBegin(3, createI64(0));
    {
      auto padsAttr = op.getPads();
      for (int64_t i = 0; i < spatialRank; ++i)
        padBegin[i] = createI64(cast<IntegerAttr>(padsAttr[i]).getInt());
    }

    Value poolMode = createI64(op.getPoolMode());
    Value storageOrder = createI64(op.getStorageOrder());
    Value ceilMode = createI64(op.getCeilMode());
    Value hasIdx = createI64(hasIndices ? 1 : 0);
    Value countIncludePad = createI64(op.getCountIncludePad());
    Value pNorm = createI64(op.getP());
    Value spatialRankV = createI64(spatialRank);
    Value dataTypeV = createI64(dataType);

    // Argument list: 4 ptrs + 28 i64.
    SmallVector<Type, 36> paramTypes;
    SmallVector<Value, 36> args;
    auto addPtr = [&](Value v) {
      paramTypes.push_back(ptrType);
      args.push_back(v);
    };
    auto addI64 = [&](Value v) {
      paramTypes.push_back(i64Type);
      args.push_back(v);
    };

    addPtr(statePtr);
    addPtr(inputPtr);
    addPtr(outputPtr);
    addPtr(indicesPtr);
    addI64(dataTypeV);
    addI64(poolMode);
    addI64(spatialRankV);
    addI64(batchN);
    addI64(channelC);
    addI64(inSpatial[0]);
    addI64(inSpatial[1]);
    addI64(inSpatial[2]);
    addI64(outSpatial[0]);
    addI64(outSpatial[1]);
    addI64(outSpatial[2]);
    addI64(kernel[0]);
    addI64(kernel[1]);
    addI64(kernel[2]);
    addI64(strides[0]);
    addI64(strides[1]);
    addI64(strides[2]);
    addI64(padBegin[0]);
    addI64(padBegin[1]);
    addI64(padBegin[2]);
    addI64(dilations[0]);
    addI64(dilations[1]);
    addI64(dilations[2]);
    addI64(storageOrder);
    addI64(ceilMode);
    addI64(hasIdx);
    addI64(countIncludePad);
    addI64(pNorm);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapPool, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populatePoolLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<PoolOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
