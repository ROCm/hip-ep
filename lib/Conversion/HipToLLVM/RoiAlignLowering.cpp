/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.roi_align -> wrap_roi_align runtime call
//===----------------------------------------------------------------------===//
//
// Runtime ABI:
//   wrap_roi_align(state, input, rois, batch_indices, output,
//                  data_type, N, C, inH, inW, num_rois, outH, outW,
//                  sampling_ratio, spatial_scale (f32), half_pixel, mode)
//   -> i32

struct RoiAlignOpLowering : public ConvertOpToLLVMPattern<RoiAlignOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RoiAlignOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto roisType = dyn_cast<MemRefType>(op.getRois().getType());
    auto indicesType = dyn_cast<MemRefType>(op.getBatchIndices().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !roisType || !indicesType || !outputType)
      return rewriter.notifyMatchFailure(op, "expected memref operands");
    if (inputType.getRank() != 4 || roisType.getRank() != 2 ||
        indicesType.getRank() != 1 || outputType.getRank() != 4)
      return rewriter.notifyMatchFailure(op, "unexpected operand ranks");

    int64_t dataType = getHipdnnDataType(inputType.getElementType());
    if (dataType != HIPDNN_EP_DATATYPE_FLOAT &&
        dataType != HIPDNN_EP_DATATYPE_HALF &&
        dataType != HIPDNN_EP_DATATYPE_BFLOAT16 &&
        dataType != HIPDNN_EP_DATATYPE_DOUBLE)
      return rewriter.notifyMatchFailure(
          op, "RoiAlign: only f16 / f32 / bf16 / f64 supported");

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value roisPtr = extractContiguousMemRefPtr(adaptor.getRois(), rewriter, loc);
    Value indicesPtr =
        extractContiguousMemRefPtr(adaptor.getBatchIndices(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value inputDesc = adaptor.getInput();
    Value roisDesc = adaptor.getRois();

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value N = getMemRefDimSize(inputType, 0, inputDesc, rewriter, loc);
    Value C = getMemRefDimSize(inputType, 1, inputDesc, rewriter, loc);
    Value inH = getMemRefDimSize(inputType, 2, inputDesc, rewriter, loc);
    Value inW = getMemRefDimSize(inputType, 3, inputDesc, rewriter, loc);
    Value numRois = getMemRefDimSize(roisType, 0, roisDesc, rewriter, loc);
    Value spatialScale = LLVM::ConstantOp::create(
        rewriter, loc, f32Type,
        rewriter.getF32FloatAttr(op.getSpatialScale().convertToFloat()));

    SmallVector<Type, 18> paramTypes;
    SmallVector<Value, 18> args;
    auto addArg = [&](Type t, Value v) {
      paramTypes.push_back(t);
      args.push_back(v);
    };

    addArg(ptrType, statePtr);
    addArg(ptrType, inputPtr);
    addArg(ptrType, roisPtr);
    addArg(ptrType, indicesPtr);
    addArg(ptrType, outputPtr);
    addArg(i64Type, createI64(dataType));
    addArg(i64Type, N);
    addArg(i64Type, C);
    addArg(i64Type, inH);
    addArg(i64Type, inW);
    addArg(i64Type, numRois);
    addArg(i64Type, createI64(op.getOutputHeight()));
    addArg(i64Type, createI64(op.getOutputWidth()));
    addArg(i64Type, createI64(op.getSamplingRatio()));
    addArg(f32Type, spatialScale);
    addArg(i64Type, createI64(op.getHalfPixel()));
    addArg(i64Type, createI64(op.getMode()));

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRoiAlign, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateRoiAlignLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns) {
  patterns.add<RoiAlignOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
