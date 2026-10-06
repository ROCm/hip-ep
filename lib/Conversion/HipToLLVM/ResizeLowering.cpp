/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "../ResizeLayout.h"
#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.resize -> wrap_resize runtime call
//===----------------------------------------------------------------------===//
//
// The kernel takes a copied prefix (N, C) and a trailing window of
// spatial_rank axes (1..3).  planHipResizeLaunch chooses
// that split from which extents change:
//
//   NCHW  1x3x16x16 -> 1x3x32x32 : N, C,     spatial_rank=2, (H, W)
//   NHWC  1x16x16x3 -> 1x32x32x3 : N, C=1,   spatial_rank=3, (H, W, C)
//
// An empty prefix slot is the constant 1, so it does not add a tensor axis.
// A window axis whose extents match is copied.  Dynamic prefix dims are read
// from the memref descriptor.
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
    std::optional<HipResizeLaunch> launch =
        planHipResizeLaunch(inputType, outputType);
    if (!launch)
      return rewriter.notifyMatchFailure(
          op, "expected a copied prefix of at most 2 axes and a trailing "
              "window of 1..3 axes");

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
    Value one = createI64(1);

    Value N = launch->prefixCount > 0
                  ? getMemRefDimSize(inputType, 0, inputDesc, rewriter, loc)
                  : one;
    Value C = launch->prefixCount > 1
                  ? getMemRefDimSize(inputType, 1, inputDesc, rewriter, loc)
                  : one;

    SmallVector<Value, 3> inSpatial(3, one);
    SmallVector<Value, 3> outSpatial(3, one);
    for (int64_t i : llvm::seq<int64_t>(launch->spatialRank)) {
      int64_t axis = launch->prefixCount + i;
      inSpatial[i] =
          getMemRefDimSize(inputType, axis, inputDesc, rewriter, loc);
      outSpatial[i] =
          getMemRefDimSize(outputType, axis, outputDesc, rewriter, loc);
    }

    Value mode = createI64(op.getMode());
    Value coord = createI64(op.getCoordTransform());
    Value nearest = createI64(op.getNearestMode());
    Value spatialRankV = createI64(launch->spatialRank);
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
