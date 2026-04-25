/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== ConvTranspose ===================================
//
// hip.conv_transpose(%ctx, %input, %weights, %bias, %output)
//   -> wrap_miopenConvolutionBackwardData(ctx,
//          input, input_n, input_c, input_h, input_w,
//          weights, weights_k,
//          bias,
//          output, output_h, output_w, output_c,
//          kernel_h, kernel_w,
//          stride_h, stride_w,
//          pad_top, pad_left, pad_bottom, pad_right,
//          dilation_h, dilation_w,
//          output_padding_h, output_padding_w,
//          group, data_type)
//
// Spatial-rank tolerance:
//   - rank 4 (NCHW): standard 2-D ConvTranspose (other ONNX models use this).
//   - rank 3 (NCW):  1-D ConvTranspose -- the only spatial dim is W, so we
//     pad H to 1 (kernel_h, stride_h, pad_top/bottom, dilation_h, opH all
//     forced to 1/0).  Used by Kokoro's iSTFT decoder block.
//
struct ConvTransposeOpLowering
    : public ConvertOpToLLVMPattern<ConvTransposeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConvTransposeOp op, OpAdaptor adaptor,
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

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value weightsPtr = extractMemRefPtr(adaptor.getWeights(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value biasPtr;
    if (adaptor.getBias()) {
      biasPtr = extractMemRefPtr(adaptor.getBias(), rewriter, loc);
    } else {
      biasPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    }

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    int64_t inRank = inputType.getRank();
    int64_t wRank = weightsType.getRank();
    int64_t outRank = outputType.getRank();

    if (inRank != 3 && inRank != 4)
      return op.emitError() << "input rank must be 3 (NCW) or 4 (NCHW), got "
                            << inRank;
    if (wRank != inRank)
      return op.emitError() << "weights rank (" << wRank
                            << ") must match input rank (" << inRank << ")";
    if (outRank != inRank)
      return op.emitError() << "output rank (" << outRank
                            << ") must match input rank (" << inRank << ")";

    Value inputN, inputC, inputH, inputW;
    Value outputH, outputW, outputC;
    Value weightsK;

    if (inRank == 4) {
      inputN =
          getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
      inputC =
          getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
      inputH =
          getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);
      inputW =
          getMemRefDimSize(inputType, 3, adaptor.getInput(), rewriter, loc);
      outputC = getMemRefDimSize(outputType, 1, adaptor.getOutput(), rewriter,
                                 loc);
      outputH = getMemRefDimSize(outputType, 2, adaptor.getOutput(), rewriter,
                                 loc);
      outputW = getMemRefDimSize(outputType, 3, adaptor.getOutput(), rewriter,
                                 loc);
    } else {
      inputN =
          getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
      inputC =
          getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
      inputH = createI64Const(1);
      inputW =
          getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);
      outputC = getMemRefDimSize(outputType, 1, adaptor.getOutput(), rewriter,
                                 loc);
      outputH = createI64Const(1);
      outputW =
          getMemRefDimSize(outputType, 2, adaptor.getOutput(), rewriter, loc);
    }

    weightsK =
        getMemRefDimSize(weightsType, 0, adaptor.getWeights(), rewriter, loc);

    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    auto dilations = op.getDilations();
    auto outputPadding = op.getOutputPadding();
    auto group = op.getGroup();

    auto getI64 = [](Attribute attr) -> int64_t {
      return cast<IntegerAttr>(attr).getInt();
    };

    int64_t spatialRank = inRank - 2;

    int64_t kH, kW, sH, sW, pT, pL, pB, pR, dH, dW, opH, opW;
    if (spatialRank == 2) {
      if (kernelShape.size() != 2 || strides.size() != 2 ||
          pads.size() != 4 || dilations.size() != 2)
        return op.emitError() << "rank-4 conv_transpose requires "
                                 "kernel_shape/strides/dilations of size 2 "
                                 "and pads of size 4";
      kH = getI64(kernelShape[0]);
      kW = getI64(kernelShape[1]);
      sH = getI64(strides[0]);
      sW = getI64(strides[1]);
      pT = getI64(pads[0]);
      pL = getI64(pads[1]);
      pB = getI64(pads[2]);
      pR = getI64(pads[3]);
      dH = getI64(dilations[0]);
      dW = getI64(dilations[1]);
      opH = 0;
      opW = 0;
      if (!outputPadding.empty()) {
        if (outputPadding.size() != 2)
          return op.emitError() << "rank-4 conv_transpose output_padding "
                                   "must have size 0 or 2";
        opH = getI64(outputPadding[0]);
        opW = getI64(outputPadding[1]);
      }
    } else {
      if (kernelShape.size() != 1 || strides.size() != 1 ||
          pads.size() != 2 || dilations.size() != 1)
        return op.emitError() << "rank-3 conv_transpose requires "
                                 "kernel_shape/strides/dilations of size 1 "
                                 "and pads of size 2";
      kH = 1;
      kW = getI64(kernelShape[0]);
      sH = 1;
      sW = getI64(strides[0]);
      pT = 0;
      pB = 0;
      pL = getI64(pads[0]);
      pR = getI64(pads[1]);
      dH = 1;
      dW = getI64(dilations[0]);
      opH = 0;
      opW = 0;
      if (!outputPadding.empty()) {
        if (outputPadding.size() != 1)
          return op.emitError() << "rank-3 conv_transpose output_padding "
                                   "must have size 0 or 1";
        opW = getI64(outputPadding[0]);
      }
    }

    Value kernelH = createI64Const(kH);
    Value kernelW = createI64Const(kW);
    Value strideH = createI64Const(sH);
    Value strideW = createI64Const(sW);
    Value padTop = createI64Const(pT);
    Value padLeft = createI64Const(pL);
    Value padBottom = createI64Const(pB);
    Value padRight = createI64Const(pR);
    Value dilationH = createI64Const(dH);
    Value dilationW = createI64Const(dW);
    Value outputPaddingH = createI64Const(opH);
    Value outputPaddingW = createI64Const(opW);
    Value groupVal = createI64Const(group);

    int64_t dt = getHipdnnDataType(inputType.getElementType());
    if (dt < 0)
      return op.emitError()
             << "unsupported element type for hip.conv_transpose";
    Value dataType = createI64Const(dt);

    SmallVector<Type, 28> paramTypes = {
        ptrType, ptrType, i64Type, i64Type, i64Type, i64Type, ptrType,
        i64Type, ptrType, ptrType, i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapConvTransposeBwd, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 28> args = {
        statePtr,  inputPtr,       inputN,    inputC,
        inputH,    inputW,         weightsPtr, weightsK,
        biasPtr,   outputPtr,      outputH,   outputW,
        outputC,   kernelH,        kernelW,   strideH,
        strideW,   padTop,         padLeft,   padBottom,
        padRight,  dilationH,      dilationW, outputPaddingH,
        outputPaddingW, groupVal,  dataType};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void mlir::hip::populateConvTransposeLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvTransposeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
