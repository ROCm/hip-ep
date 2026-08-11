/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {
namespace {

//===----------------------------------------------------------------------===//
// DynamicDispatch Backend Lowering Patterns
//===----------------------------------------------------------------------===//
//
// These patterns lower HIP dialect operations to DynamicDispatch (Vitis AI /
// XRT NPU backend) runtime wrappers instead of GPU backends (MIOpen/hipBLASLt).
//
// Enabled via --use-dynamic-dispatch pass option.
//
// Pattern:
//   hip.gemm → wrap_dd_matmul (NPU) instead of wrap_gemm (GPU/hipBLASLt)
//   hip.conv → wrap_dd_conv2d (NPU) instead of wrap_miopenConvolutionForward
//===----------------------------------------------------------------------===//

// DynamicDispatch runtime function names
inline constexpr const char *kWrapDDMatMul = "wrap_dd_matmul";
inline constexpr const char *kWrapDDConv2d = "wrap_dd_conv2d";

//===----------------------------------------------------------------------===//
// GemmOpDynamicDispatchLowering - GEMM via DynamicDispatch
//===----------------------------------------------------------------------===//
//
// Lowers hip.gemm to wrap_dd_matmul (NPU execution via DynamicDispatch)
//
// Before:
//   %out = hip.gemm(%ctx, %A, %B, %C, %init) {alpha=1.0, beta=0.0, transA=0,
//                                              transB=0}
//
// After:
//   llvm.call @wrap_dd_matmul(%ctx, %op_slot, %A, %B, %C, %out,
//                             M, N, K, alpha, beta, transA, transB, typeCode)
//===----------------------------------------------------------------------===//

struct GemmOpDynamicDispatchLowering : public ConvertOpToLLVMPattern<GemmOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f64Type = rewriter.getF64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract element type and map to HIPDNN_EP_DATATYPE_*
    auto AType = cast<MemRefType>(op.getInputA().getType());
    Type elemType = AType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);
    if (dataType < 0) {
      return op.emitError("Unsupported element type for DynamicDispatch GEMM");
    }
    Value dataTypeVal = createI64Const(dataType);

    // Extract memref pointers
    Value statePtr = adaptor.getCtx();
    Value input_A_ptr =
        extractContiguousMemRefPtr(adaptor.getInputA(), rewriter, loc);
    Value input_B_ptr =
        extractContiguousMemRefPtr(adaptor.getInputB(), rewriter, loc);
    Value input_C_ptr =
        extractOptionalMemRefPtr(adaptor.getInputC(), rewriter, loc);
    Value output_ptr =
        extractContiguousMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Convert alpha/beta to f64 for DynamicDispatch
    Value alpha = LLVM::ConstantOp::create(
        rewriter, loc, f64Type,
        rewriter.getF64FloatAttr(adaptor.getAlpha().convertToDouble()));
    Value beta = LLVM::ConstantOp::create(
        rewriter, loc, f64Type,
        rewriter.getF64FloatAttr(adaptor.getBeta().convertToDouble()));
    Value transA = createI64Const(adaptor.getTransA());
    Value transB = createI64Const(adaptor.getTransB());

    // Extract M, N, K dimensions
    // A.shape = transA ? [K, M] : [M, K]
    // B.shape = transB ? [N, K] : [K, N]
    Value M, K, N;
    if (adaptor.getTransA()) {
      K = getMemRefDimSize(AType, 0, adaptor.getInputA(), rewriter, loc);
      M = getMemRefDimSize(AType, 1, adaptor.getInputA(), rewriter, loc);
    } else {
      M = getMemRefDimSize(AType, 0, adaptor.getInputA(), rewriter, loc);
      K = getMemRefDimSize(AType, 1, adaptor.getInputA(), rewriter, loc);
    }
    auto BType = cast<MemRefType>(op.getInputB().getType());
    if (adaptor.getTransB()) {
      N = getMemRefDimSize(BType, 0, adaptor.getInputB(), rewriter, loc);
    } else {
      N = getMemRefDimSize(BType, 1, adaptor.getInputB(), rewriter, loc);
    }

    // Build function signature for wrap_dd_matmul
    SmallVector<Type, 16> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // input_a
        ptrType, // input_b
        ptrType, // bias (can be null)
        ptrType, // output
        i64Type, // M
        i64Type, // N
        i64Type, // K
        f64Type, // alpha
        f64Type, // beta
        i64Type, // transA
        i64Type, // transB
        i64Type, // data_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapDDMatMul, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 16> args = {
        statePtr,    getOpStateSlotValue(op, rewriter, loc),
        input_A_ptr, input_B_ptr,
        input_C_ptr, // bias (C matrix)
        output_ptr,  M,
        N,           K,
        alpha,       beta,
        transA,      transB,
        dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvOpDynamicDispatchLowering - Convolution via DynamicDispatch
//===----------------------------------------------------------------------===//
//
// Lowers hip.conv to wrap_dd_conv2d (NPU execution via DynamicDispatch)
//
// Before:
//   %out = hip.conv(%ctx) ins(%in, %w, %b) outs(%o)
//          {kernel_shape=[3,3], strides=[1,1], ...}
//
// After:
//   llvm.call @wrap_dd_conv2d(%ctx, %op_slot, %in, n, c, h, w,
//                             %w, k, %b, %o, out_h, out_w,
//                             kh, kw, sh, sw, pt, pl, pb, pr,
//                             dh, dw, group, dataType)
//===----------------------------------------------------------------------===//

struct ConvOpDynamicDispatchLowering : public ConvertOpToLLVMPattern<ConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConvOp op, OpAdaptor adaptor,
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

    // Extract memref pointers
    Value statePtr = adaptor.getCtx();
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
      biasPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);
    }

    // Extract shapes from memref types
    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Verify ranks
    if (inputType.getRank() != 4 || weightsType.getRank() != 4 ||
        outputType.getRank() != 4) {
      return op.emitError("DynamicDispatch Conv requires rank-4 tensors");
    }

    // Extract element type and map to HIPDNN_EP_DATATYPE_*
    Type elemType = outputType.getElementType();
    int64_t dataType = getHipdnnDataType(elemType);
    if (dataType < 0) {
      return op.emitError("Unsupported element type for DynamicDispatch Conv");
    }
    Value dataTypeVal = createI64Const(dataType);

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

    // Convert attributes to LLVM constants
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

    // Build function signature for wrap_dd_conv2d
    SmallVector<Type, 32> paramTypes = {
        ptrType, // state
        i32Type, // op_state_slot
        ptrType, // input
        i64Type, // n
        i64Type, // c
        i64Type, // h
        i64Type, // w
        ptrType, // weights
        i64Type, // k
        ptrType, // bias
        ptrType, // output
        i64Type, // out_h
        i64Type, // out_w
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
        i64Type, // data_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapDDConv2d, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 32> args = {
        statePtr,   getOpStateSlotValue(op, rewriter, loc),
        inputPtr,   inputN,
        inputC,     inputH,
        inputW,     weightsPtr,
        weightsK,   biasPtr,
        outputPtr,  outputH,
        outputW,    kernelH,
        kernelW,    strideH,
        strideW,    padTop,
        padLeft,    padBottom,
        padRight,   dilationH,
        dilationW,  groupVal,
        dataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern Population Functions
//===----------------------------------------------------------------------===//

void populateDynamicDispatchGemmLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GemmOpDynamicDispatchLowering>(converter);
}

void populateDynamicDispatchConvLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<ConvOpDynamicDispatchLowering>(converter);
}

} // namespace hip
} // namespace mlir
