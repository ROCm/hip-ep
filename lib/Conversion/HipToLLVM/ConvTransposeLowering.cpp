/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.conv_transpose(%ctx, %input, %weights, %bias, %output)
//   -> wrap_miopenConvolutionTranspose(ctx, input, input_n, input_c, input_h,
//                                       input_w, weights, bias, output,
//                                       output_c, output_h, output_w, kernel_h,
//                                       kernel_w, stride_h, stride_w, pad_top,
//                                       pad_left, pad_bottom, pad_right,
//                                       dilation_h, dilation_w,
//                                       output_padding_h, output_padding_w,
//                                       group, data_type)
//
// The runtime selects MIOpen's miopenTranspose convolution mode. The weight
// layout is ONNX ConvTranspose's [C, M/group, kH, kW] (input channels first);
// the runtime derives M/group from output_c and group. data_type is derived
// here from the output element type so the runtime sets the correct MIOpen
// dtype (f16/bf16/f32).
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
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value weightsPtr =
        extractContiguousMemRefPtr(adaptor.getWeights(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    Value biasPtr;
    if (adaptor.getBias())
      biasPtr = extractContiguousMemRefPtr(adaptor.getBias(), rewriter, loc);
    else
      biasPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    if (inputType.getRank() != 4)
      return op.emitError("Input must be rank-4 tensor [N, C, H, W]");
    if (weightsType.getRank() != 4)
      return op.emitError("Weights must be rank-4 tensor [C, M/group, R, S]");
    if (outputType.getRank() != 4)
      return op.emitError("Output must be rank-4 tensor [N, M, H', W']");

    // Input shape: [N, C, H, W] (static or dynamic dims both handled).
    Value inputN =
        getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value inputC =
        getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
    Value inputH =
        getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);
    Value inputW =
        getMemRefDimSize(inputType, 3, adaptor.getInput(), rewriter, loc);

    // Output shape: [N, M, H', W'] — M is the total output channel count.
    Value outputC =
        getMemRefDimSize(outputType, 1, adaptor.getOutput(), rewriter, loc);
    Value outputH =
        getMemRefDimSize(outputType, 2, adaptor.getOutput(), rewriter, loc);
    Value outputW =
        getMemRefDimSize(outputType, 3, adaptor.getOutput(), rewriter, loc);

    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    auto dilations = op.getDilations();
    auto outputPadding = op.getOutputPadding();
    auto group = op.getGroup();

    auto getI64 = [](Attribute attr) -> int64_t {
      return cast<IntegerAttr>(attr).getInt();
    };

    Value kernelH = createI64Const(getI64(kernelShape[0]));
    Value kernelW = createI64Const(getI64(kernelShape[1]));
    Value strideH = createI64Const(getI64(strides[0]));
    Value strideW = createI64Const(getI64(strides[1]));
    Value padTop = createI64Const(getI64(pads[0]));
    Value padLeft = createI64Const(getI64(pads[1]));
    Value padBottom = createI64Const(getI64(pads[2]));
    Value padRight = createI64Const(getI64(pads[3]));
    Value dilationH = createI64Const(getI64(dilations[0]));
    Value dilationW = createI64Const(getI64(dilations[1]));
    Value outPadH = createI64Const(getI64(outputPadding[0]));
    Value outPadW = createI64Const(getI64(outputPadding[1]));
    Value groupVal = createI64Const(group);

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return op.emitError("ConvTranspose: unsupported element type");
    Value dataTypeVal = createI64Const(dataType);

    SmallVector<Type, 27> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // input
        i64Type, // input_n
        i64Type, // input_c
        i64Type, // input_h
        i64Type, // input_w
        ptrType, // weights
        ptrType, // bias
        ptrType, // output
        i64Type, // output_c
        i64Type, // output_h
        i64Type, // output_w
        i64Type, // kernel_h
        i64Type, // kernel_w
        i64Type, // stride_h
        i64Type, // stride_w
        i64Type, // pad_top
        i64Type, // pad_left
        i64Type, // pad_bottom
        i64Type, // pad_right
        i64Type, // dilation_h
        i64Type, // dilation_w
        i64Type, // output_padding_h
        i64Type, // output_padding_w
        i64Type, // group
        i64Type  // data_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenConvolutionTranspose, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    auto opStateSlot = getOpStateSlotValue(op, rewriter, loc);
    SmallVector<Value, 27> args = {
        statePtr, opStateSlot, inputPtr,    inputN,    inputC,    inputH,
        inputW,   weightsPtr,  biasPtr,     outputPtr, outputC,   outputH,
        outputW,  kernelH,     kernelW,     strideH,   strideW,   padTop,
        padLeft,  padBottom,   padRight,    dilationH, dilationW, outPadH,
        outPadW,  groupVal,    dataTypeVal,
    };

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateConvTransposeLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns) {
  patterns.add<ConvTransposeOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
