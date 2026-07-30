/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"
#include <mlir/Dialect/LLVMIR/LLVMDialect.h>

namespace mlir {
namespace hip {
namespace {

// ===== Convolution ops ================================

// hip.conv(%ctx, %input, %weights, %bias, %output)
//   -> wrap_miopenConvolutionForward(ctx, input, input_n, input_c, input_h,
//                                     input_w, weights, weights_k, bias,
//                                     output, output_h, output_w, kernel_h,
//                                     kernel_w, stride_h, stride_w, pad_top,
//                                     pad_left, pad_bottom, pad_right,
//                                     dilation_h, dilation_w, group, data_type)
//
// Before:
//   %out = hip.conv(%ctx) ins(%in, %w, %b :
//                              memref<1x3x896x896xf16, 1>,
//                              memref<1152x3x14x14xf16, 1>,
//                              memref<1152xf16, 1>)
//                          outs(%o : memref<1x1152x64x64xf16, 1>)
//                          {kernel_shape=[14,14], strides=[14,14], ...}
// After:
//   llvm.call @wrap_miopenConvolutionForward(%ctx, %in, 1, 3, 896, 896,
//                                              %w, 1152, %b, %o, 64, 64,
//                                              14, 14, 14, 14, 0, 0, 0, 0,
//                                              1, 1, 1,
//                                              /*data_type=*/1 /* f16 */)
//
// The `data_type` value is derived from the OUTPUT memref's element type and
// applied uniformly to all three MIOpen tensor descriptors. MIOpen requires
// input / weights / output to share the same dtype; this is enforced by the
// host-side typing rule in OnnxToHip (`init` tensor allocated with
// `resultType.getElementType()`).
struct ConvOpLowering : public ConvertOpToLLVMPattern<ConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Generate call to runtime wrapper following opaque RuntimeState pattern.
    // The wrapper extracts handle/stream from state internally (no direct field
    // access!).
    //
    // Signature:
    // int wrap_miopenConvolutionForward(
    //     RuntimeState* state,    // Opaque pointer - extracts handle/stream
    //                             // internally
    //     void* input,            // Input tensor data pointer
    //     int64_t input_n,        // Input batch size
    //     int64_t input_c,        // Input channels
    //     int64_t input_h,        // Input height
    //     int64_t input_w,        // Input width
    //     void* weights,          // Weights tensor data pointer
    //     int64_t weights_k,      // Output channels (number of filters)
    //     void* bias,             // Bias tensor data pointer (nullable)
    //     void* output,           // Output tensor data pointer (in-place)
    //     int64_t output_h,       // Output height
    //     int64_t output_w,       // Output width
    //     int64_t kernel_h,       // Kernel height
    //     int64_t kernel_w,       // Kernel width
    //     int64_t stride_h,       // Stride height
    //     int64_t stride_w,       // Stride width
    //     int64_t pad_top,        // Padding top
    //     int64_t pad_left,       // Padding left
    //     int64_t pad_bottom,     // Padding bottom
    //     int64_t pad_right,      // Padding right
    //     int64_t dilation_h,     // Dilation height
    //     int64_t dilation_w,     // Dilation width
    //     int64_t group,          // Number of groups
    //     int64_t data_type       // HIPDNN_EP_DATATYPE_* for I/O+weights
    // );
    //
    // Returns: 0 on success, non-zero on error

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract memref pointers (aligned pointers from descriptors)
    Value statePtr = adaptor.getCtx(); // RuntimeState* (opaque)
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value weightsPtr =
        extractContiguousMemRefPtr(adaptor.getWeights(), rewriter, loc);
    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Handle optional bias
    Value biasPtr;
    if (adaptor.getBias()) {
      biasPtr = extractContiguousMemRefPtr(adaptor.getBias(), rewriter, loc);
    } else {
      // Pass null pointer if no bias
      biasPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    }

    // Extract shapes from memref types
    // Supports both static and dynamic dimensions using MemRefDescriptor
    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Verify ranks
    if (inputType.getRank() != 4) {
      return op.emitError("Input must be rank-4 tensor [N, C, H, W]");
    }
    if (weightsType.getRank() != 4) {
      return op.emitError("Weights must be rank-4 tensor [K, C, R, S]");
    }
    if (outputType.getRank() != 4) {
      return op.emitError("Output must be rank-4 tensor [N, K, H', W']");
    }

    // Input shape: [N, C, H, W]
    Value inputN =
        getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value inputC =
        getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
    Value inputH =
        getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);
    Value inputW =
        getMemRefDimSize(inputType, 3, adaptor.getInput(), rewriter, loc);

    // Weights shape: [K, C, R, S] where K=output channels
    Value weightsK =
        getMemRefDimSize(weightsType, 0, adaptor.getWeights(), rewriter, loc);

    // Output shape: [N, K, H', W']
    Value outputH =
        getMemRefDimSize(outputType, 2, adaptor.getOutput(), rewriter, loc);
    Value outputW =
        getMemRefDimSize(outputType, 3, adaptor.getOutput(), rewriter, loc);

    // Extract attributes
    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    auto dilations = op.getDilations();
    auto group = op.getGroup();

    // Extract integer values from attributes
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
    Value groupVal = createI64Const(group);

    // dtype: derive from the output memref's element type. All three buffers
    // (input, weights, output) must share this dtype — see the typing rule
    // in OnnxToHip::ConvConversion which uses resultType.getElementType()
    // for the allocated output. The runtime fails fast if the dtype is
    // unsupported.
    int64_t dataTypeEnum = getHipdnnDataType(outputType.getElementType());
    if (dataTypeEnum < 0)
      return op.emitError("hip.conv: unsupported output element type ")
             << outputType.getElementType();
    Value dataType = createI64Const(dataTypeEnum);

    // Build function signature
    SmallVector<Type, 25> paramTypes = {
        ptrType, // state
        ptrType, // input
        i64Type, // input_n
        i64Type, // input_c
        i64Type, // input_h
        i64Type, // input_w
        ptrType, // weights
        i64Type, // weights_k
        ptrType, // bias
        ptrType, // output
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
        i64Type, // group
        i64Type  // data_type
    };

    // Lookup or create the runtime function
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenConvolutionForward, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // Build argument list matching the signature
    SmallVector<Value, 25> args = {
        statePtr,   inputPtr, inputN,    inputC,    inputH,   inputW,
        weightsPtr, weightsK, biasPtr,   outputPtr, outputH,  outputW,
        kernelH,    kernelW,  strideH,   strideW,   padTop,   padLeft,
        padBottom,  padRight, dilationH, dilationW, groupVal, dataType};

    // Call the runtime function
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // Erase the HIP conv operation (it's in-place, no results)
    rewriter.eraseOp(op);
    return success();
  }
};

class Im2d2ColLowering : public ConvertOpToLLVMPattern<Im2d2ColOp> {
public:
  using ConvertOpToLLVMPattern<Im2d2ColOp>::ConvertOpToLLVMPattern;
  LogicalResult
  matchAndRewrite(Im2d2ColOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto loc = op->getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    SmallVector<Type> paramTypes = {
        ptrType, // state
        ptrType, // input
        i64Type, // data_type
        i64Type, // C
        i64Type, // H
        i64Type, // W
        i64Type, // kh
        i64Type, // kw
        i64Type, // pad_top
        i64Type, // pad_bottom
        i64Type, // pad_left
        i64Type, // pad_right
        i64Type, // stride_h
        i64Type, // stride_w
        i64Type, // dilation_h
        i64Type, // dilation_w
        ptrType, // output
        i64Type, // out_h
        i64Type, // out_w
    };

    // Lookup or create the runtime function
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kIm2d2Col, paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    auto inputType = cast<MemRefType>(op.getInput().getType());

    auto createI64Const = [&](Attribute value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      cast<IntegerAttr>(value));
    };

    Value statePtr = adaptor.getCtx();
    Value inputPtr =
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value dataType =
        createI64Const(rewriter.getI64IntegerAttr(getHipdnnDataType(
            cast<MemRefType>(op.getInput().getType()).getElementType())));
    auto _ = getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    auto C = getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
    auto H = getMemRefDimSize(inputType, 2, adaptor.getInput(), rewriter, loc);
    auto W = getMemRefDimSize(inputType, 3, adaptor.getInput(), rewriter, loc);
    auto kh = createI64Const(adaptor.getKernelShape()[0]);
    auto kw = createI64Const(adaptor.getKernelShape()[1]);
    auto pad_top = createI64Const(adaptor.getPads()[0]);
    auto pad_bottom = createI64Const(adaptor.getPads()[1]);
    auto pad_left = createI64Const(adaptor.getPads()[2]);
    auto pad_right = createI64Const(adaptor.getPads()[3]);
    auto stride_h = createI64Const(adaptor.getStrides()[0]);
    auto stride_w = createI64Const(adaptor.getStrides()[1]);
    auto dilation_h = createI64Const(adaptor.getDilations()[0]);
    auto dilation_w = createI64Const(adaptor.getDilations()[1]);

    Value outputPtr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);
    auto out_h = createI64Const(adaptor.getOutDims()[0]);
    auto out_w = createI64Const(adaptor.getOutDims()[1]);

    SmallVector<Value> args({statePtr, inputPtr, dataType, C, H, W, kh, kw,
                             pad_top, pad_bottom, pad_left, pad_right, stride_h,
                             stride_w, dilation_h, dilation_w, outputPtr, out_h,
                             out_w});

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);

    return success();
  }
};

} // namespace

void populateConvLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<ConvOpLowering>(converter);
  patterns.add<Im2d2ColLowering>(converter);
}

} // namespace hip
} // namespace mlir
