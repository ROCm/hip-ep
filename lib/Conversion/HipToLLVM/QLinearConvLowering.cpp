/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

// hip.qlinear_conv -> wrap_qlinear_conv
//
// Before:
//   hip.qlinear_conv(%ctx) ins(%x, %xs, %xz, %w, %ws, %wz, %ys, %yz, %b :
//                        memref<1x3x8x8xui8, 1>, memref<f32, 1>,
//                        memref<ui8, 1>, memref<4x3x3x3xi8, 1>, ...)
//                   outs(%y : memref<1x4x4x4xui8, 1>)
//                   {kernel_shape = [3, 3], strides = [2, 2],
//                    pads = [1, 1, 1, 1], dilations = [1, 1], group = 1}
// After:
//   llvm.call @wrap_qlinear_conv(%ctx, %x, %xs, %xz, %w, %ws, %wz, %ys, %yz,
//                                %b, %y, 1, 3, 4, 8, 8, 4, 4, 3, 3, 2, 2,
//                                1, 1, 1, 1, 1, 7, 5, 7, 3, 1, 1, 1, 1, 1, 1)
//
// Extents come from the memref descriptors so a dynamic batch is legal. The
// window attributes are static. Scale and zero-point lengths are element
// counts: 1 is per-tensor, and a weight count equal to the output channels is
// per-channel. An absent bias is a null pointer with bias dtype -1.
struct QLinearConvOpLowering : public ConvertOpToLLVMPattern<QLinearConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QLinearConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto inputType = dyn_cast<MemRefType>(op.getInput().getType());
    auto weightsType = dyn_cast<MemRefType>(op.getWeights().getType());
    auto outputType = dyn_cast<MemRefType>(op.getOutput().getType());
    if (!inputType || !weightsType || !outputType || inputType.getRank() != 4 ||
        weightsType.getRank() != 4 || outputType.getRank() != 4)
      return op.emitError("hip.qlinear_conv: input, weights, and output must "
                          "be rank-4 memrefs");

    if (op.getKernelShape().size() != 2 || op.getStrides().size() != 2 ||
        op.getDilations().size() != 2 || op.getPads().size() != 4)
      return op.emitError(
          "hip.qlinear_conv: 2D window attributes are required");

    auto at = [](ArrayAttr attr, unsigned index) {
      return cast<IntegerAttr>(attr[index]).getInt();
    };

    int64_t inputDtype = getHipdnnDataType(inputType.getElementType());
    int64_t weightDtype = getHipdnnDataType(weightsType.getElementType());
    int64_t outputDtype = getHipdnnDataType(outputType.getElementType());
    if (inputDtype < 0 || weightDtype < 0 || outputDtype < 0)
      return op.emitError("hip.qlinear_conv: element type the runtime cannot "
                          "name");

    int64_t biasDtype = HIPDNN_EP_DATATYPE_UNSUPPORTED;
    if (op.getBias()) {
      auto biasType = dyn_cast<MemRefType>(op.getBias().getType());
      if (!biasType)
        return op.emitError("hip.qlinear_conv: bias must be a memref");
      biasDtype = getHipdnnDataType(biasType.getElementType());
      if (biasDtype != HIPDNN_EP_DATATYPE_INT32)
        return op.emitError("hip.qlinear_conv: bias must be i32");
    }

    auto createI64 = [&](int64_t value) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    Value input = adaptor.getInput();
    Value weights = adaptor.getWeights();
    Value output = adaptor.getOutput();

    SmallVector<Type> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType, ptrType,
        ptrType, ptrType, ptrType, i64Type, i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type, i64Type, i64Type,
        i64Type, i64Type, i64Type, i64Type, i64Type,
    };
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQLinearConv, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractContiguousMemRefPtr(input, rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getInputScale(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getInputZeroPoint(), rewriter, loc),
        extractContiguousMemRefPtr(weights, rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getWeightScale(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getWeightZeroPoint(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutputScale(), rewriter, loc),
        extractContiguousMemRefPtr(adaptor.getOutputZeroPoint(), rewriter, loc),
        extractOptionalMemRefPtr(adaptor.getBias(), rewriter, loc),
        extractContiguousMemRefPtr(output, rewriter, loc),
        getMemRefDimSize(inputType, 0, input, rewriter, loc),
        getMemRefDimSize(inputType, 1, input, rewriter, loc),
        getMemRefDimSize(weightsType, 0, weights, rewriter, loc),
        getMemRefDimSize(inputType, 2, input, rewriter, loc),
        getMemRefDimSize(inputType, 3, input, rewriter, loc),
        getMemRefDimSize(outputType, 2, output, rewriter, loc),
        getMemRefDimSize(outputType, 3, output, rewriter, loc),
        createI64(at(op.getKernelShape(), 0)),
        createI64(at(op.getKernelShape(), 1)),
        createI64(at(op.getStrides(), 0)),
        createI64(at(op.getStrides(), 1)),
        createI64(at(op.getPads(), 0)),
        createI64(at(op.getPads(), 1)),
        createI64(at(op.getDilations(), 0)),
        createI64(at(op.getDilations(), 1)),
        createI64(op.getGroup()),
        createI64(inputDtype),
        createI64(weightDtype),
        createI64(outputDtype),
        createI64(biasDtype),
        computeNumElements(cast<MemRefType>(op.getInputScale().getType()),
                           adaptor.getInputScale(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getWeightScale().getType()),
                           adaptor.getWeightScale(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getOutputScale().getType()),
                           adaptor.getOutputScale(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getInputZeroPoint().getType()),
                           adaptor.getInputZeroPoint(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getWeightZeroPoint().getType()),
                           adaptor.getWeightZeroPoint(), rewriter, loc),
        computeNumElements(cast<MemRefType>(op.getOutputZeroPoint().getType()),
                           adaptor.getOutputZeroPoint(), rewriter, loc),
    };

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateQLinearConvLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns) {
  patterns.add<QLinearConvOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
