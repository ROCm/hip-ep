/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// ===== Fused quantized 1x1 convolution (W4A16) ==============================
//
// hip.qconv(%ctx, %in, %w, %wscale, %wzp, %bias, %out) -> wrap_qconv
//
// Before:
//   hip.qconv(%ctx) ins(%in, %w, %wscale, %wzp :
//                        memref<1x2048x1x128xui16, 1>,   // activation
//                        memref<1024x2048x1x1xi8, 1>,    // packed INT4 weights
//                        memref<1024xf32, 1>,            // per-channel scale
//                        memref<1024xi8, 1>)             // packed INT4 zp
//                   outs(%out : memref<1x1024x1x128xui16, 1>)
//                   {input_scale = ..., input_zp = ..., output_scale = ...,
//                    output_zp = ..., kernel_shape = [1, 1], ..., packed_int4}
// After:
//   llvm.call @wrap_qconv(%ctx, %in, %w, %wscale, %wzp, null, %out,
//                         1, 2048, 1024, 128,   // N, Cin, Cout, spatial_size
//                         9, 5, 4, -1,          // act dtype, w dtype, w bits,
//                                               //   bias dtype (absent)
//                         input_scale, input_zp, output_scale, output_zp)
//
// The ABI is deliberately narrower than wrap_conv's 19 geometry slots. A 1x1
// kernel with unit stride, unit dilation and no padding collapses to one dot
// product per output position down the channel axis, so the only extent the
// kernel needs beyond N/Cin/Cout is how many positions there are -- and since
// nothing is strided or padded, that count is the same on the input and the
// output. Everything wrap_conv passes per-axis would be a constant 1 or 0 here.
//
// `spatial_size` is built as a runtime product rather than folded on the host:
// the token dimension is dynamic on a shape-specialised LLM graph, so the
// extents are only known from the memref descriptor.
//
// The geometry attributes are still checked rather than assumed. hip.qconv can
// SPELL a general convolution, and only the fusion pattern guarantees it never
// does; anything else has to fail loudly here instead of being silently
// lowered as if it were 1x1. These are emitError, not notifyMatchFailure,
// because no other pattern can match hip.qconv -- declining would leave the op
// unlowered and surface much later as an unrelated-looking failure.
struct QConvOpLowering : public ConvertOpToLLVMPattern<QConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto scalesType = cast<MemRefType>(op.getWeightScales().getType());
    auto zeroPointsType = cast<MemRefType>(op.getWeightZeroPoints().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // NCHW throughout: a 1x1 kernel over two spatial dims, which is what the
    // two-entry kernel_shape checked below implies.
    if (inputType.getRank() != 4 || weightsType.getRank() != 4 ||
        outputType.getRank() != 4)
      return op.emitError("hip.qconv: input, weights and output must be rank 4,"
                          " got ")
             << inputType.getRank() << ", " << weightsType.getRank() << ", "
             << outputType.getRank();

    // --- geometry: only the degenerate 1x1 form the kernel implements -------
    auto allEqual = [](ArrayAttr attr, int64_t expected) {
      return llvm::all_of(attr, [&](Attribute entry) {
        return cast<IntegerAttr>(entry).getInt() == expected;
      });
    };
    if (op.getKernelShape().size() != 2 || !allEqual(op.getKernelShape(), 1))
      return op.emitError("hip.qconv: kernel_shape must be [1, 1]");
    if (op.getStrides().size() != 2 || !allEqual(op.getStrides(), 1))
      return op.emitError("hip.qconv: strides must be [1, 1]");
    if (op.getDilations().size() != 2 || !allEqual(op.getDilations(), 1))
      return op.emitError("hip.qconv: dilations must be [1, 1]");
    if (op.getPads().size() != 4 || !allEqual(op.getPads(), 0))
      return op.emitError("hip.qconv: pads must be [0, 0, 0, 0]");
    if (op.getGroup() != 1)
      return op.emitError("hip.qconv: group must be 1, got ") << op.getGroup();
    // Weight quantization is per output channel, which for a [Cout, Cin, k...]
    // filter is axis 0. The kernel indexes the scale by output channel, so any
    // other axis would pair each scale with the wrong reduction.
    if (op.getWeightAxis() != 0)
      return op.emitError("hip.qconv: weight_axis must be 0, got ")
             << op.getWeightAxis();

    // --- element types ------------------------------------------------------
    Type activationElem = inputType.getElementType();
    if (activationElem != outputType.getElementType())
      return op.emitError("hip.qconv: input and output element types must "
                          "match, got ")
             << activationElem << " and " << outputType.getElementType();
    if (!activationElem.isUnsignedInteger(16))
      return op.emitError("hip.qconv: activation must be ui16, got ")
             << activationElem;

    // ONNX guarantees zero_point.dtype == x.dtype, so one storage type and one
    // width describe both weight buffers.
    Type weightElem = weightsType.getElementType();
    if (weightElem != zeroPointsType.getElementType())
      return op.emitError("hip.qconv: weights and weight_zero_points element "
                          "types must match, got ")
             << weightElem << " and " << zeroPointsType.getElementType();
    if (weightElem.getIntOrFloatBitWidth() != 8 ||
        !isa<IntegerType>(weightElem))
      return op.emitError("hip.qconv: weights must have 8-bit integer storage, "
                          "got ")
             << weightElem;
    if (!scalesType.getElementType().isF32())
      return op.emitError("hip.qconv: weight_scales must be f32, got ")
             << scalesType.getElementType();
    if (scalesType.getRank() != 1 || zeroPointsType.getRank() != 1)
      return op.emitError("hip.qconv: weight_scales and weight_zero_points "
                          "must be rank 1");

    // A packed tensor keeps its 8-bit element type and its LOGICAL shape, so
    // the real value width has to travel as a separate parameter. Mirrors
    // resolveQuantBits in QdqLowering.cpp; 8-bit storage is what makes two
    // nibbles per byte meaningful in the first place.
    int64_t weightBits = 8;
    if (op.getPackedInt4()) {
      if (weightElem.getIntOrFloatBitWidth() != 8)
        return op.emitError(
            "hip.qconv: packed_int4 requires 8-bit weight storage");
      weightBits = 4;
    }

    int64_t activationDtype = getHipdnnDataType(activationElem);
    int64_t weightDtype = getHipdnnDataType(weightElem);
    if (activationDtype < 0 || weightDtype < 0)
      return op.emitError("hip.qconv: element type the runtime cannot name");

    float outputScale = op.getOutputScale().convertToFloat();
    if (outputScale == 0.0f)
      return op.emitError("hip.qconv: output_scale must be non-zero");

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    Value batch =
        getMemRefDimSize(inputType, 0, adaptor.getInput(), rewriter, loc);
    Value inChannels =
        getMemRefDimSize(inputType, 1, adaptor.getInput(), rewriter, loc);
    // Output channels are the weight tensor's first dim, matching hip.conv.
    Value outChannels =
        getMemRefDimSize(weightsType, 0, adaptor.getWeights(), rewriter, loc);
    // Output positions: the product of the spatial extents. Unit stride and no
    // padding make the input's spatial extents identical, so one count serves
    // both sides of the dot product.
    Value spatialSize = createI64Const(1);
    for (unsigned dim : llvm::seq<unsigned>(2, outputType.getRank()))
      spatialSize = LLVM::MulOp::create(rewriter, loc, spatialSize,
                                        getMemRefDimSize(outputType, dim,
                                                         adaptor.getOutput(),
                                                         rewriter, loc));

    Value biasPtr = extractOptionalMemRefPtr(adaptor.getBias(), rewriter, loc);
    // Meaningful only when biasPtr is non-null; the sentinel says "no bias
    // type" rather than naming an unsupported one.
    int64_t biasDtype = HIPDNN_EP_DATATYPE_UNSUPPORTED;
    if (op.getBias())
      biasDtype = getHipdnnDataType(
          cast<MemRefType>(op.getBias().getType()).getElementType());

    SmallVector<Type, 19> paramTypes = {
        ptrType,          // state
        ptrType,          // input
        ptrType,          // weights (packed)
        ptrType,          // weight_scales
        ptrType,          // weight_zero_points (packed)
        ptrType,          // bias (nullable)
        ptrType,          // output
        i64Type,          // batch
        i64Type,          // in_channels
        i64Type,          // out_channels
        i64Type,          // spatial_size
        i64Type,          // activation_dtype
        i64Type,          // weight_dtype
        i64Type,          // weight_bits
        i64Type,          // bias_dtype
        f32Type, i64Type, // input_scale, input_zp
        f32Type, i64Type, // output_scale, output_zp
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQConv, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 19> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getWeights(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getWeightScales(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getWeightZeroPoints(), rewriter,
                                   loc),
        biasPtr,
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc),
        batch,
        inChannels,
        outChannels,
        spatialSize,
        createI64Const(activationDtype),
        createI64Const(weightDtype),
        createI64Const(weightBits),
        createI64Const(biasDtype),
        createF32Const(op.getInputScale().convertToFloat()),
        createI64Const(op.getInputZp()),
        createF32Const(outputScale),
        createI64Const(op.getOutputZp()),
    };

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // hip.qconv writes into its `outs` operand and has no lowered result.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQConvLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns) {
  patterns.add<QConvOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
