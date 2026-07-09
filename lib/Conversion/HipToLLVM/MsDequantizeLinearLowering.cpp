/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// hip.ms_dequantize_linear lowering
//===----------------------------------------------------------------------===//
//
// Before:
//   hip.ms_dequantize_linear(%ctx)
//       ins(%input, %scale : memref<Bx512xi8, 1>, memref<1x512xf16, 1>)
//       zero_points(%zp : memref<Bx512xi8, 1>)
//       outs(%out : memref<Bx512xf16, 1>)
//       {axis=1, input_elem_size=1, scale_elem_size=2}
//
// After:
//   llvm.call @wrap_ms_dequantize_linear(
//       %state, %input_ptr, %scale_ptr, %zp_ptr (nullable), %out_ptr,
//       %n_elements, %axis, %n_channels,
//       %input_elem_size, %scale_elem_size)

struct MsDequantizeLinearOpLowering
    : public ConvertOpToLLVMPattern<MsDequantizeLinearOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MsDequantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();

    auto inputType  = cast<MemRefType>(op.getInput().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    Value statePtr  = adaptor.getCtx();
    Value inputPtr  = extractContiguousMemRefPtr(adaptor.getInput(),  rewriter, loc);
    Value scalePtr  = extractContiguousMemRefPtr(adaptor.getScale(),  rewriter, loc);
    Value zpPtr     = extractOptionalMemRefPtr(adaptor.getZeroPoint(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Total elements
    int64_t totalElems = 1;
    bool isDynamic = false;
    for (int64_t d : outputType.getShape()) {
      if (d == ShapedType::kDynamic) { isDynamic = true; break; }
      totalElems *= d;
    }

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value nElems;
    if (!isDynamic) {
      nElems = createI64(totalElems);
    } else {
      // Compute at runtime from output memref descriptor.
      nElems = createI64(1);
      for (int r = 0; r < outputType.getRank(); ++r) {
        Value dim = getMemRefDimSize(outputType, r, adaptor.getOutput(), rewriter, loc);
        nElems = LLVM::MulOp::create(rewriter, loc, i64Type, nElems, dim);
      }
    }

    // axis and n_channels (length of scale/zp along that axis)
    int64_t axisVal = op.getAxis();
    Value axisV     = createI64(axisVal);

    int64_t nChannels = 1;
    bool channelDynamic = false;
    if (axisVal >= 0 && axisVal < inputType.getRank()) {
      int64_t d = inputType.getShape()[axisVal];
      if (d == ShapedType::kDynamic) channelDynamic = true;
      else nChannels = d;
    }
    Value nChannelsV;
    if (!channelDynamic) {
      nChannelsV = createI64(nChannels);
    } else {
      nChannelsV = getMemRefDimSize(inputType, axisVal, adaptor.getInput(), rewriter, loc);
    }

    Value inputElemSize = createI64(op.getInputElemSize());
    Value scaleElemSize = createI64(op.getScaleElemSize());

    SmallVector<Type, 10> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType,
        i64Type, i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMsDequantizeLinear, paramTypes,
        rewriter.getI32Type());
    if (failed(funcOp)) return failure();

    SmallVector<Value, 10> args = {
        statePtr, inputPtr, scalePtr, zpPtr, outputPtr,
        nElems, axisV, nChannelsV, inputElemSize, scaleElemSize};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// hip.ms_quantize_linear lowering
//===----------------------------------------------------------------------===//

struct MsQuantizeLinearOpLowering
    : public ConvertOpToLLVMPattern<MsQuantizeLinearOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MsQuantizeLinearOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();

    auto inputType  = cast<MemRefType>(op.getInput().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    Value statePtr  = adaptor.getCtx();
    Value inputPtr  = extractContiguousMemRefPtr(adaptor.getInput(),  rewriter, loc);
    Value scalePtr  = extractContiguousMemRefPtr(adaptor.getScale(),  rewriter, loc);
    Value zpPtr     = extractOptionalMemRefPtr(adaptor.getZeroPoint(), rewriter, loc);
    Value outputPtr = extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    int64_t totalElems = 1;
    bool isDynamic = false;
    for (int64_t d : outputType.getShape()) {
      if (d == ShapedType::kDynamic) { isDynamic = true; break; }
      totalElems *= d;
    }

    auto createI64 = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    Value nElems;
    if (!isDynamic) {
      nElems = createI64(totalElems);
    } else {
      nElems = createI64(1);
      for (int r = 0; r < outputType.getRank(); ++r) {
        Value dim = getMemRefDimSize(outputType, r, adaptor.getOutput(), rewriter, loc);
        nElems = LLVM::MulOp::create(rewriter, loc, i64Type, nElems, dim);
      }
    }

    int64_t axisVal = op.getAxis();
    Value axisV     = createI64(axisVal);

    int64_t nChannels = 1;
    bool channelDynamic = false;
    if (axisVal >= 0 && axisVal < inputType.getRank()) {
      int64_t d = inputType.getShape()[axisVal];
      if (d == ShapedType::kDynamic) channelDynamic = true;
      else nChannels = d;
    }
    Value nChannelsV;
    if (!channelDynamic) {
      nChannelsV = createI64(nChannels);
    } else {
      nChannelsV = getMemRefDimSize(inputType, axisVal, adaptor.getInput(), rewriter, loc);
    }

    Value inputElemSize  = createI64(op.getInputElemSize());
    Value outputElemSize = createI64(op.getOutputElemSize());

    SmallVector<Type, 10> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType,
        i64Type, i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMsQuantizeLinear, paramTypes,
        rewriter.getI32Type());
    if (failed(funcOp)) return failure();

    SmallVector<Value, 10> args = {
        statePtr, inputPtr, scalePtr, zpPtr, outputPtr,
        nElems, axisV, nChannelsV, inputElemSize, outputElemSize};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMsDequantizeLinearLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<MsDequantizeLinearOpLowering>(converter);
  patterns.add<MsQuantizeLinearOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
