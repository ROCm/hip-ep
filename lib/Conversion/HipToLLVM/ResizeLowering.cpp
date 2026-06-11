/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.resize -> wrap_resize runtime call
//===----------------------------------------------------------------------===//
//
// Generic 1D / 2D / 3D spatial resize.  spatial_rank is derived from the
// input rank (= 2 + spatial_rank).  All shape-derived scalar args come from
// the memref descriptors so dynamic N (and C) is honored.  Output spatial
// dims are required static at the conversion (see ResizeConversion.cpp), so
// they fold to constants here; the runtime kernel reads in_dim and out_dim
// per axis and computes per-axis scale internally.
//
// Runtime ABI:
//   wrap_resize(state, input, output,
//               data_type,
//               spatial_rank,
//               N, C,
//               in0..2, out0..2,
//               mode, coord_transform, nearest_mode)
//   -> i32

struct ResizeOpLowering : public ConvertOpToLLVMPattern<ResizeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ResizeOp op, OpAdaptor adaptor,
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
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t rank = inputType.getRank();
    if (rank < 3 || rank > 5)
      return rewriter.notifyMatchFailure(op, "expected rank in [3, 5]");
    int64_t spatialRank = rank - 2;

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType < 0 || (dataType > 2 && dataType != 6))
      return rewriter.notifyMatchFailure(
          op, "Resize: only f16 / f32 / bf16 / f64 supported");

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value inputDesc = adaptor.getInput();
    Value outputDesc = adaptor.getOutput();

    Value N = getMemRefDimSize(inputType, 0, inputDesc, rewriter, loc);
    Value C = getMemRefDimSize(inputType, 1, inputDesc, rewriter, loc);

    SmallVector<Value, 3> inSpatial(3, createI64(1));
    SmallVector<Value, 3> outSpatial(3, createI64(1));
    for (int64_t i : llvm::seq<int64_t>(spatialRank)) {
      inSpatial[i] =
          getMemRefDimSize(inputType, 2 + i, inputDesc, rewriter, loc);
      outSpatial[i] =
          getMemRefDimSize(outputType, 2 + i, outputDesc, rewriter, loc);
    }

    Value mode = createI64(op.getMode());
    Value coord = createI64(op.getCoordTransform());
    Value nearest = createI64(op.getNearestMode());
    Value spatialRankV = createI64(spatialRank);
    Value dataTypeV = createI64(dataType);

    SmallVector<Type, 16> paramTypes;
    SmallVector<Value, 16> args;
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
    addI64(dataTypeV);
    addI64(spatialRankV);
    addI64(N);
    addI64(C);
    addI64(inSpatial[0]);
    addI64(inSpatial[1]);
    addI64(inSpatial[2]);
    addI64(outSpatial[0]);
    addI64(outSpatial[1]);
    addI64(outSpatial[2]);
    addI64(mode);
    addI64(coord);
    addI64(nearest);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapResize, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateResizeLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<ResizeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
