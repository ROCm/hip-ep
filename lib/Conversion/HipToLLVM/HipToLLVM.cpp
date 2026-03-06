/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/ControlFlowToLLVM/ControlFlowToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Pass/PassRegistry.h"
#include "mlir/Transforms/DialectConversion.h"
#include "mlir/Transforms/Passes.h"
#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Dialect/IR/HipDialect.h"

namespace mlir {
namespace hip {

namespace {

static constexpr const char* kMiopenConvolutionForward =
    "wrap_miopenConvolutionForward";
static constexpr const char* kHipGetConstant = "hipdnn_ep_constant_get";
static constexpr const char* kHipGetPoolBase = "hipdnn_ep_get_pool_base";

// Maps MLIR element type to runtime data type enum (HIPDNN_EP_DATATYPE_*).
// Values must match the #defines in hipdnn_ep_runtime.h.
// Returns -1 for unsupported types.
static int64_t getHipdnnDataType(Type elemType) {
  if (elemType.isF32()) return 0;   // HIPDNN_EP_DATATYPE_FLOAT
  if (elemType.isF16()) return 1;   // HIPDNN_EP_DATATYPE_HALF
  if (elemType.isBF16()) return 2;  // HIPDNN_EP_DATATYPE_BFLOAT16
  return -1;
}

// --- GetPoolOp: hip.get_pool(%ctx) : memref<?xi8, 1>
//     -> llvm.call @hipdnn_ep_get_pool_base(state) + memref descriptor
struct GetPoolOpLowering : public ConvertOpToLLVMPattern<GetPoolOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    MemRefType memRefType = cast<MemRefType>(op.getPool().getType());

    // void* hipdnn_ep_get_pool_base(void* state)
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGetPoolBase, {ptrType}, ptrType);
    if (failed(funcOp))
      return failure();

    Value rawPtr =
        LLVM::CallOp::create(rewriter, loc, *funcOp, adaptor.getCtx())
            .getResult();

    // Cast to GPU address space (address space 1)
    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();

    Value gpuPtr = rawPtr;
    if (cast<LLVM::LLVMPointerType>(rawPtr.getType()).getAddressSpace() !=
        *addrSpace)
      gpuPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          rawPtr);

    // Use pool size from module metadata (set by memory-pooling pass)
    int64_t poolSizeVal = 0;
    if (auto attr = module->getAttrOfType<IntegerAttr>("hipdnn.pool_size"))
      poolSizeVal = attr.getInt();

    Value poolSize = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(poolSizeVal));
    Value stride1 = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(1));

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, gpuPtr, gpuPtr, {poolSize}, {stride1}, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// --- ConvOp: hip.conv(%handle, %input, %weights, %bias, %output) ->
//             llvm.call @miopenConvolutionForward(...)
struct ConvOpLowering : public ConvertOpToLLVMPattern<ConvOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ConvOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();

    // Generate call to runtime wrapper following opaque RuntimeState pattern.
    // The wrapper extracts handle/stream from state internally (no direct field
    // access!).
    //
    // Signature:
    // int wrap_miopenConvolutionForward(
    //     RuntimeState* state,    // Opaque pointer - extracts handle/stream
    //     internally void* input,            // Input tensor data pointer
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
    //     int64_t group           // Number of groups
    // );
    //
    // Returns: 0 on success, non-zero on error

    // Helper to create i64 constants
    Type i64Type = rewriter.getI64Type();
    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    // Extract memref pointers (aligned pointers from descriptors)
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      // Cast to void* (address space 0) if needed
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getHandle(); // RuntimeState* (opaque)
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value weightsPtr = getAlignedPtr(adaptor.getWeights());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    // Handle optional bias
    Value biasPtr;
    if (adaptor.getBias()) {
      biasPtr = getAlignedPtr(adaptor.getBias());
    } else {
      // Pass null pointer if no bias
      biasPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    }

    // Extract shapes from memref types (static shapes known at compile time)
    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto weightsType = cast<MemRefType>(op.getWeights().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Input shape: [N, C, H, W]
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return op.emitError("Input must be rank-4 tensor [N, C, H, W]");
    }
    Value inputN = createI64Const(inputShape[0]);
    Value inputC = createI64Const(inputShape[1]);
    Value inputH = createI64Const(inputShape[2]);
    Value inputW = createI64Const(inputShape[3]);

    // Weights shape: [K, C, R, S] where K=output channels
    auto weightsShape = weightsType.getShape();
    if (weightsShape.size() != 4) {
      return op.emitError("Weights must be rank-4 tensor [K, C, R, S]");
    }
    Value weightsK = createI64Const(weightsShape[0]);

    // Output shape: [N, K, H', W']
    auto outputShape = outputType.getShape();
    if (outputShape.size() != 4) {
      return op.emitError("Output must be rank-4 tensor [N, K, H', W']");
    }
    Value outputH = createI64Const(outputShape[2]);
    Value outputW = createI64Const(outputShape[3]);

    // Extract attributes
    auto kernelShape = op.getKernelShape();
    auto strides = op.getStrides();
    auto pads = op.getPads();
    auto dilations = op.getDilations();
    auto group = op.getGroup();

    // Extract integer values from attributes
    auto getI64 = [](mlir::Attribute attr) -> int64_t {
      return cast<mlir::IntegerAttr>(attr).getInt();
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

    // Build function signature
    SmallVector<Type, 24> paramTypes = {
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
        i64Type  // group
    };

    // Lookup or create the runtime function
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenConvolutionForward, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // Build argument list matching the signature
    SmallVector<Value, 24> args = {
        statePtr,   inputPtr, inputN,    inputC,    inputH,  inputW,
        weightsPtr, weightsK, biasPtr,   outputPtr, outputH, outputW,
        kernelH,    kernelW,  strideH,   strideW,   padTop,  padLeft,
        padBottom,  padRight, dilationH, dilationW, groupVal};

    // Call the runtime function
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // Erase the HIP conv operation (it's in-place, no results)
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GEMM (General Matrix Multiply) Lowering
//===----------------------------------------------------------------------===//

struct GemmOpLowering : public ConvertOpToLLVMPattern<GemmOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    // Helper to create constants
    auto createI32Const = [&](int32_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i32Type, rewriter.getI32IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, f32Type,
                                               rewriter.getF32FloatAttr(value));
    };

    // Extract memref pointers
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getHandle(); // RuntimeState*
    Value APtr = getAlignedPtr(adaptor.getA());
    Value BPtr = getAlignedPtr(adaptor.getB());
    Value CPtr = getAlignedPtr(adaptor.getC());
    Value resultPtr = getAlignedPtr(adaptor.getOutput());

    // Extract matrix dimensions from memref types
    // A: [M x K], B: [K x N], C: [M x N], result: [M x N]
    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());

    auto AShape = AType.getShape();
    auto BShape = BType.getShape();

    if (AShape.size() != 2 || BShape.size() != 2) {
      return op.emitError("GEMM requires 2D matrices");
    }

    int64_t M = AShape[0];
    int64_t K = AShape[1];
    int64_t N = BShape[1];

    Value m = createI32Const(M);
    Value n = createI32Const(N);
    Value k = createI32Const(K);

    // Get transpose flags and scalar multipliers
    int64_t transA = op.getTransA();
    int64_t transB = op.getTransB();
    float alpha = op.getAlpha().convertToFloat();
    float beta = op.getBeta().convertToFloat();

    Value transAVal = createI32Const(transA);
    Value transBVal = createI32Const(transB);
    Value alphaVal = createF32Const(alpha);
    Value betaVal = createF32Const(beta);

    // Leading dimensions (assuming row-major, no transpose)
    // For row-major: lda = K, ldb = N, ldc = N
    Value lda = createI32Const(transA == 0 ? K : M);
    Value ldb = createI32Const(transB == 0 ? N : K);
    Value ldc = createI32Const(N);

    // Build wrapper function signature
    // int wrap_hipblas_sgemm(
    //     RuntimeState* state,
    //     int transA, int transB,
    //     int m, int n, int k,
    //     float* alpha,
    //     float* A, int lda,
    //     float* B, int ldb,
    //     float* beta,
    //     float* C, int ldc,
    //     float* result
    // )
    SmallVector<Type, 15> paramTypes = {
        ptrType, // state
        i32Type, // transA
        i32Type, // transB
        i32Type, // m
        i32Type, // n
        i32Type, // k
        ptrType, // alpha (pointer to scalar)
        ptrType, // A
        i32Type, // lda
        ptrType, // B
        i32Type, // ldb
        ptrType, // beta (pointer to scalar)
        ptrType, // C
        i32Type, // ldc
        ptrType  // result
    };

    // Create wrapper function name
    static constexpr const char* kHipBlasSgemm = "wrap_hipblas_sgemm";

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipBlasSgemm, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // Allocate stack space for alpha and beta scalars
    Value one = createI32Const(1);
    Value alphaPtr =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, f32Type, one, 0);
    Value betaPtr =
        rewriter.create<LLVM::AllocaOp>(loc, ptrType, f32Type, one, 0);
    rewriter.create<LLVM::StoreOp>(loc, alphaVal, alphaPtr);
    rewriter.create<LLVM::StoreOp>(loc, betaVal, betaPtr);

    // Build argument list
    SmallVector<Value, 15> args = {
        statePtr, transAVal, transBVal, m,       n,    k,   alphaPtr, APtr,
        lda,      BPtr,      ldb,       betaPtr, CPtr, ldc, resultPtr};

    // Call the runtime function
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // Erase the HIP gemm operation
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReLU Activation Lowering
//===----------------------------------------------------------------------===//

struct ReluOpLowering : public ConvertOpToLLVMPattern<ReluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Runtime wrapper signature (following ConvOp pattern - MemRef-agnostic):
    // int wrap_miopenActivationForward_relu(
    //     RuntimeState* state,
    //     void* input_gpu_ptr,
    //     int64_t input_n, int64_t input_c, int64_t input_h, int64_t input_w,
    //     void* output_gpu_ptr,
    //     int64_t output_n, int64_t output_c, int64_t output_h, int64_t
    //     output_w
    // );

    // Helper: create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    // Extract GPU pointers (aligned pointers from descriptors)
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    // Extract shapes from memref types (static shapes)
    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto inputShape = inputType.getShape();
    if (inputShape.size() != 4) {
      return op.emitError("Input must be rank-4 tensor [N, C, H, W]");
    }
    Value inputN = createI64Const(inputShape[0]);
    Value inputC = createI64Const(inputShape[1]);
    Value inputH = createI64Const(inputShape[2]);
    Value inputW = createI64Const(inputShape[3]);

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    auto outputShape = outputType.getShape();
    if (outputShape.size() != 4) {
      return op.emitError("Output must be rank-4 tensor [N, C, H, W]");
    }
    Value outputN = createI64Const(outputShape[0]);
    Value outputC = createI64Const(outputShape[1]);
    Value outputH = createI64Const(outputShape[2]);
    Value outputW = createI64Const(outputShape[3]);

    // Build function signature with dimension parameters
    SmallVector<Type, 11> paramTypes = {
        ptrType,                            // state
        ptrType,                            // input_gpu_ptr
        i64Type, i64Type, i64Type, i64Type, // input_n, c, h, w
        ptrType,                            // output_gpu_ptr
        i64Type, i64Type, i64Type, i64Type  // output_n, c, h, w
    };

    // Lookup or create the runtime function
    StringRef funcName = "wrap_miopenActivationForward_relu";
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, funcName, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // Call with extracted values
    SmallVector<Value, 11> args = {statePtr, inputPtr, inputN,    inputC,
                                   inputH,   inputW,   outputPtr, outputN,
                                   outputC,  outputH,  outputW};
    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // Erase the HIP relu operation
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Elementwise Mul Lowering
//===----------------------------------------------------------------------===//

struct MulOpLowering : public ConvertOpToLLVMPattern<MulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value lhsPtr = getAlignedPtr(adaptor.getLhs());
    Value rhsPtr = getAlignedPtr(adaptor.getRhs());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    // Compute total number of elements from static shape
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    auto outputShape = outputType.getShape();
    int64_t numElements = 1;
    for (int64_t dim : outputShape) {
      if (dim == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in hip.mul lowering");
      }
      numElements *= dim;
    }

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.mul");

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value numElementsVal = createI64Const(numElements);
    Value dataTypeVal = createI64Const(dataType);
    Value tensorOpVal = createI64Const(0); // HIPDNN_EP_TENSOR_OP_MUL

    SmallVector<Type, 7> paramTypes = {
        ptrType, // state
        ptrType, // lhs
        ptrType, // rhs
        ptrType, // output
        i64Type, // num_elements
        i64Type, // data_type (HIPDNN_EP_DATATYPE_*)
        i64Type  // tensor_op (HIPDNN_EP_TENSOR_OP_*)
    };

    static constexpr const char* kWrapMiopenOpTensor = "wrap_miopenOpTensor";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenOpTensor, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,        lhsPtr,         rhsPtr,
                                  outputPtr, numElementsVal, dataTypeVal,
                                  tensorOpVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Elementwise Sub Lowering
//===----------------------------------------------------------------------===//

struct SubOpLowering : public ConvertOpToLLVMPattern<SubOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value lhsPtr = getAlignedPtr(adaptor.getLhs());
    Value rhsPtr = getAlignedPtr(adaptor.getRhs());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    auto outputShape = outputType.getShape();
    int64_t numElements = 1;
    for (int64_t dim : outputShape) {
      if (dim == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in hip.sub lowering");
      }
      numElements *= dim;
    }

    unsigned elementSizeBytes =
        outputType.getElementType().getIntOrFloatBitWidth() / 8;

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value numElementsVal = createI64Const(numElements);
    Value elemSizeVal = createI64Const(elementSizeBytes);

    SmallVector<Type, 6> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, i64Type, i64Type};

    static constexpr const char* kWrapElementwiseSub = "wrap_elementwise_sub";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapElementwiseSub, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,       lhsPtr,     rhsPtr,
                                  outputPtr, numElementsVal, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Gather Lowering
//===----------------------------------------------------------------------===//

struct GatherOpLowering : public ConvertOpToLLVMPattern<GatherOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value dataPtr = getAlignedPtr(adaptor.getData());
    Value indicesPtr = getAlignedPtr(adaptor.getIndices());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto dataType = cast<MemRefType>(op.getData().getType());
    int64_t dataNumElements = 1;
    for (int64_t dim : dataType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in hip.gather lowering");
      dataNumElements *= dim;
    }

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t outputNumElements = 1;
    for (int64_t dim : outputType.getShape())
      outputNumElements *= dim;

    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value axisVal = createI64Const(op.getAxis());
    Value dataNumElementsVal = createI64Const(dataNumElements);
    Value outputNumElementsVal = createI64Const(outputNumElements);
    Value elemSizeVal = createI64Const(elementSizeBytes);

    // int wrap_gather(RuntimeState* state, void* data, void* indices,
    //                 void* output, int64_t axis, int64_t data_num_elements,
    //                 int64_t output_num_elements, int64_t element_size_bytes)
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type, i64Type};

    static constexpr const char* kWrapGather = "wrap_gather";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGather, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr, dataPtr, indicesPtr, outputPtr,
                                  axisVal,  dataNumElementsVal,
                                  outputNumElementsVal, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ReduceSum Lowering
//===----------------------------------------------------------------------===//

struct ReduceSumOpLowering : public ConvertOpToLLVMPattern<ReduceSumOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value dataPtr = getAlignedPtr(adaptor.getData());
    Value axesPtr = getAlignedPtr(adaptor.getAxes());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto dataType = cast<MemRefType>(op.getData().getType());
    int64_t dataNumElements = 1;
    for (int64_t dim : dataType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in hip.reduce_sum lowering");
      dataNumElements *= dim;
    }

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t outputNumElements = 1;
    for (int64_t dim : outputType.getShape())
      outputNumElements *= dim;

    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value dataNumElementsVal = createI64Const(dataNumElements);
    Value outputNumElementsVal = createI64Const(outputNumElements);
    Value elemSizeVal = createI64Const(elementSizeBytes);
    Value keepdimsVal = createI64Const(op.getKeepdims());

    // int wrap_reduce_sum(RuntimeState* state, void* data, void* axes,
    //                     void* output, int64_t data_num_elements,
    //                     int64_t output_num_elements, int64_t element_size_bytes,
    //                     int64_t keepdims)
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type, i64Type};

    static constexpr const char* kWrapReduceSum = "wrap_reduce_sum";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapReduceSum, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {statePtr, dataPtr, axesPtr, outputPtr,
                                  dataNumElementsVal, outputNumElementsVal,
                                  elemSizeVal, keepdimsVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Cast Lowering
//===----------------------------------------------------------------------===//

struct CastOpLowering : public ConvertOpToLLVMPattern<CastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t numElements = 1;
    for (int64_t dim : inputType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in hip.cast lowering");
      numElements *= dim;
    }
    // For scalar (rank-0) memrefs, numElements stays 1

    unsigned inputElemSize =
        inputType.getElementType().getIntOrFloatBitWidth() / 8;
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    unsigned outputElemSize =
        outputType.getElementType().getIntOrFloatBitWidth() / 8;

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value numElementsVal = createI64Const(numElements);
    Value inputElemSizeVal = createI64Const(inputElemSize);
    Value outputElemSizeVal = createI64Const(outputElemSize);
    Value toVal = createI64Const(op.getTo());

    // int wrap_cast(RuntimeState* state, void* input, void* output,
    //               int64_t num_elements, int64_t input_element_size,
    //               int64_t output_element_size, int64_t to)
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type, i64Type};

    static constexpr const char* kWrapCast = "wrap_cast";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCast, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr, inputPtr, outputPtr,
                                  numElementsVal, inputElemSizeVal,
                                  outputElemSizeVal, toVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Sigmoid Activation Lowering
//===----------------------------------------------------------------------===//

struct SigmoidOpLowering : public ConvertOpToLLVMPattern<SigmoidOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto outputType = cast<MemRefType>(op.getOutput().getType());
    int64_t numElements = 1;
    for (int64_t dim : outputType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in hip.sigmoid lowering");
      numElements *= dim;
    }

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.sigmoid");

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value numElementsVal = createI64Const(numElements);
    Value dataTypeVal = createI64Const(dataType);
    Value activationModeVal = createI64Const(0); // HIPDNN_EP_ACTIVATION_SIGMOID

    // int wrap_miopenActivationForward(RuntimeState* state, void* input,
    //     void* output, int64_t num_elements, int64_t data_type,
    //     int64_t activation_mode)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType, i64Type,
                                       i64Type, i64Type};

    static constexpr const char* kWrapMiopenActivationForward =
        "wrap_miopenActivationForward";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenActivationForward, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,        inputPtr,    outputPtr,
                                  numElementsVal, dataTypeVal, activationModeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// RotaryEmbedding Lowering
//===----------------------------------------------------------------------===//

struct RotaryEmbeddingOpLowering
    : public ConvertOpToLLVMPattern<RotaryEmbeddingOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RotaryEmbeddingOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      return ptr;
    };

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getHandle();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value posIdsPtr = getAlignedPtr(adaptor.getPositionIds());
    Value cosCachePtr = getAlignedPtr(adaptor.getCosCache());
    Value sinCachePtr = getAlignedPtr(adaptor.getSinCache());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    Value interleaved = createI64Const(op.getInterleaved());
    Value numHeads = createI64Const(op.getNumHeads());
    Value rotaryDim = createI64Const(op.getRotaryEmbeddingDim());

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t inputNumElements = 1;
    for (int64_t dim : inputType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");
      inputNumElements *= dim;
    }
    Value inputNumElementsVal = createI64Const(inputNumElements);

    auto cosCacheType = cast<MemRefType>(op.getCosCache().getType());
    int64_t cosCacheNumElements = 1;
    for (int64_t dim : cosCacheType.getShape())
      cosCacheNumElements *= dim;
    Value cosCacheNumElementsVal = createI64Const(cosCacheNumElements);

    unsigned elementSizeBytes =
        inputType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    // int wrap_rotary_embedding(RuntimeState* state,
    //     void* input, void* position_ids, void* cos_cache, void* sin_cache,
    //     void* output,
    //     int64_t interleaved, int64_t num_heads, int64_t rotary_dim,
    //     int64_t input_num_elements, int64_t cos_cache_num_elements,
    //     int64_t element_size_bytes)
    SmallVector<Type, 12> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, ptrType, ptrType,
        i64Type, i64Type, i64Type, i64Type, i64Type, i64Type};

    static constexpr const char* kWrapRotaryEmbedding =
        "wrap_rotary_embedding";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapRotaryEmbedding, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr, inputPtr, posIdsPtr, cosCachePtr, sinCachePtr, outputPtr,
        interleaved, numHeads, rotaryDim,
        inputNumElementsVal, cosCacheNumElementsVal, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SimplifiedLayerNorm Lowering
//===----------------------------------------------------------------------===//

struct SimplifiedLayerNormOpLowering
    : public ConvertOpToLLVMPattern<SimplifiedLayerNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SimplifiedLayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      return ptr;
    };

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getHandle();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value scalePtr = getAlignedPtr(adaptor.getScale());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t inputNumElements = 1;
    for (int64_t dim : inputType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");
      inputNumElements *= dim;
    }

    auto scaleType = cast<MemRefType>(op.getScale().getType());
    int64_t scaleNumElements = 1;
    for (int64_t dim : scaleType.getShape())
      scaleNumElements *= dim;

    unsigned elementSizeBytes =
        inputType.getElementType().getIntOrFloatBitWidth() / 8;

    Value inputNumElementsVal = createI64Const(inputNumElements);
    Value scaleNumElementsVal = createI64Const(scaleNumElements);
    Value elemSizeVal = createI64Const(elementSizeBytes);
    Value axisVal = createI64Const(op.getAxis());
    Value stashTypeVal = createI64Const(op.getStashType());
    Value epsilonVal = rewriter.create<LLVM::ConstantOp>(
        loc, f32Type,
        rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()));

    // int wrap_miopenT5LayerNormForward(RuntimeState* state,
    //     void* input, void* scale, void* output,
    //     int64_t input_num_elements, int64_t scale_num_elements,
    //     int64_t element_size_bytes,
    //     int64_t axis, float epsilon, int64_t stash_type)
    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                         i64Type, i64Type, i64Type,
                                         i64Type, f32Type, i64Type};

    static constexpr const char* kWrapMiopenT5LayerNormForward =
        "wrap_miopenT5LayerNormForward";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenT5LayerNormForward, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 10> args = {
        statePtr, inputPtr, scalePtr, outputPtr,
        inputNumElementsVal, scaleNumElementsVal, elemSizeVal,
        axisVal, epsilonVal, stashTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// SkipSimplifiedLayerNorm Lowering
//===----------------------------------------------------------------------===//

struct SkipSimplifiedLayerNormOpLowering
    : public ConvertOpToLLVMPattern<SkipSimplifiedLayerNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SkipSimplifiedLayerNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      return ptr;
    };

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    Value statePtr = adaptor.getHandle();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value skipPtr = getAlignedPtr(adaptor.getSkip());
    Value gammaPtr = getAlignedPtr(adaptor.getGamma());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());
    Value skipOutputPtr = getAlignedPtr(adaptor.getSkipOutput());

    auto inputType = cast<MemRefType>(op.getInput().getType());
    int64_t inputNumElements = 1;
    for (int64_t dim : inputType.getShape()) {
      if (dim == ShapedType::kDynamic)
        return rewriter.notifyMatchFailure(op, "dynamic shapes not supported");
      inputNumElements *= dim;
    }

    auto gammaType = cast<MemRefType>(op.getGamma().getType());
    int64_t gammaNumElements = 1;
    for (int64_t dim : gammaType.getShape())
      gammaNumElements *= dim;

    unsigned elementSizeBytes =
        inputType.getElementType().getIntOrFloatBitWidth() / 8;

    Value inputNumElementsVal = createI64Const(inputNumElements);
    Value gammaNumElementsVal = createI64Const(gammaNumElements);
    Value elemSizeVal = createI64Const(elementSizeBytes);
    Value epsilonVal = rewriter.create<LLVM::ConstantOp>(
        loc, f32Type,
        rewriter.getF32FloatAttr(op.getEpsilon().convertToFloat()));

    // int wrap_skip_simplified_layer_norm(RuntimeState* state,
    //     void* input, void* skip, void* gamma,
    //     void* output, void* skip_output,
    //     int64_t input_num_elements, int64_t gamma_num_elements,
    //     int64_t element_size_bytes, float epsilon)
    SmallVector<Type, 10> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                         ptrType, ptrType,
                                         i64Type, i64Type, i64Type, f32Type};

    static constexpr const char* kWrapSkipSimplifiedLayerNorm =
        "wrap_skip_simplified_layer_norm";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapSkipSimplifiedLayerNorm, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 10> args = {
        statePtr, inputPtr, skipPtr, gammaPtr, outputPtr, skipOutputPtr,
        inputNumElementsVal, gammaNumElementsVal, elemSizeVal, epsilonVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// Constant Management Operations Lowering
//===----------------------------------------------------------------------===//

// --- GetConstantOp: %mem = hip.get_constant(%ctx, %index) : memref<...>
//     -> %ptr = hip_get_constant(%ctx, %index) + build memref descriptor
struct GetConstantOpLowering : public ConvertOpToLLVMPattern<GetConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    MemRefType memRefType = op.getResult().getType();

    // Function signature: void* hip_get_constant(void* state, i64 index)
    SmallVector<Type, 2> paramTypes = {ptrType, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGetConstant, paramTypes, ptrType);
    if (failed(funcOp))
      return failure();

    // Call runtime function to get GPU pointer
    SmallVector<Value, 2> args = {
        adaptor.getCtx(),  // state pointer
        adaptor.getIndex() // constant index
    };

    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    Value gpuPtr = callOp.getResult();

    // Cast to GPU address space (address space 1)
    // hip_get_constant returns !llvm.ptr, but memref needs !llvm.ptr<1>
    Value gpuPtrWithAddrSpace = rewriter.create<LLVM::AddrSpaceCastOp>(
        loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 1), gpuPtr);

    // Build sizes and strides for memref descriptor
    auto shape = memRefType.getShape();
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;

    // Calculate sizes (all static for constants)
    for (int64_t dim : shape) {
      Value size = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(dim));
      sizes.push_back(size);
    }

    // Calculate strides (row-major layout)
    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      Value strideVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(stride));
      strides.insert(strides.begin(), strideVal);
      stride *= shape[i];
    }

    // Create memref descriptor from GPU pointer (with address space)
    MemRefDescriptor desc =
        createMemRefDescriptor(loc, memRefType, gpuPtrWithAddrSpace,
                               gpuPtrWithAddrSpace, sizes, strides, rewriter);

    // Replace with the constructed memref descriptor
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GroupQueryAttention Lowering
//===----------------------------------------------------------------------===//

struct GroupQueryAttentionOpLowering
    : public ConvertOpToLLVMPattern<GroupQueryAttentionOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GroupQueryAttentionOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(loc, f32Type,
                                               rewriter.getF32FloatAttr(value));
    };

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value queryPtr = getAlignedPtr(adaptor.getQuery());
    Value keyPtr = getAlignedPtr(adaptor.getKey());
    Value valuePtr = getAlignedPtr(adaptor.getValue());
    Value pastKeyPtr = getAlignedPtr(adaptor.getPastKey());
    Value pastValuePtr = getAlignedPtr(adaptor.getPastValue());
    Value seqlensKPtr = getAlignedPtr(adaptor.getSeqlensK());
    Value totalSeqLenPtr = getAlignedPtr(adaptor.getTotalSeqLen());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());
    Value presentKeyPtr = getAlignedPtr(adaptor.getPresentKey());
    Value presentValuePtr = getAlignedPtr(adaptor.getPresentValue());

    // cos/sin cache: NULL pointers for now (RoPE done via separate op)
    Value nullPtr = rewriter.create<LLVM::ZeroOp>(loc, ptrType);
    Value cosCachePtr = nullPtr;
    Value sinCachePtr = nullPtr;

    Value numHeads = createI64Const(op.getNumHeads());
    Value kvNumHeads = createI64Const(op.getKvNumHeads());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value softcap = createF32Const(op.getSoftcap().convertToFloat());
    Value doRotary = createI64Const(op.getDoRotary());
    Value rotaryInterleaved = createI64Const(op.getRotaryInterleaved());

    // Extract shape info from query memref: [batch, seq_q, num_heads * head_dim]
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto queryShape = queryType.getShape();
    int64_t batchSize = queryShape[0];
    int64_t seqLenQ = queryShape[1];
    int64_t queryHidden = queryShape[2];
    int64_t headDim = queryHidden / op.getNumHeads();
    unsigned elementSizeBytes =
        queryType.getElementType().getIntOrFloatBitWidth() / 8;

    // Extract seq_len_kv from present_key shape.
    // ONNX GQA uses BNSD layout: [batch, kv_num_heads, total_seq, head_dim]
    auto presentKeyType = cast<MemRefType>(op.getPresentKey().getType());
    auto pkShape = presentKeyType.getShape();
    int64_t seqLenKV = (pkShape.size() == 4) ? pkShape[2] : pkShape[1];

    Value batchSizeVal = createI64Const(batchSize);
    Value seqLenQVal = createI64Const(seqLenQ);
    Value seqLenKVVal = createI64Const(seqLenKV);
    Value headDimVal = createI64Const(headDim);
    Value elemSizeVal = createI64Const(elementSizeBytes);

    SmallVector<Type, 24> paramTypes = {
        ptrType,  // state
        ptrType,  // query
        ptrType,  // key
        ptrType,  // value
        ptrType,  // past_key
        ptrType,  // past_value
        ptrType,  // seqlens_k
        ptrType,  // total_seq_len
        ptrType,  // cos_cache
        ptrType,  // sin_cache
        ptrType,  // output
        ptrType,  // present_key
        ptrType,  // present_value
        i64Type,  // num_heads
        i64Type,  // kv_num_heads
        f32Type,  // scale
        f32Type,  // softcap
        i64Type,  // do_rotary
        i64Type,  // rotary_interleaved
        i64Type,  // batch_size
        i64Type,  // seq_len_q
        i64Type,  // seq_len_kv
        i64Type,  // head_dim
        i64Type   // element_size_bytes
    };

    static constexpr const char* kWrapGQA = "wrap_group_query_attention";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGQA, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 24> args = {
        statePtr, queryPtr, keyPtr, valuePtr,
        pastKeyPtr, pastValuePtr, seqlensKPtr, totalSeqLenPtr,
        cosCachePtr, sinCachePtr,
        outputPtr, presentKeyPtr, presentValuePtr,
        numHeads, kvNumHeads, scale, softcap,
        doRotary, rotaryInterleaved,
        batchSizeVal, seqLenQVal, seqLenKVVal, headDimVal, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// MatMul Lowering
//===----------------------------------------------------------------------===//

struct MatMulOpLowering : public ConvertOpToLLVMPattern<MatMulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatMulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(value));
    };

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getHandle();
    Value APtr = getAlignedPtr(adaptor.getA());
    Value BPtr = getAlignedPtr(adaptor.getB());
    Value resultPtr = getAlignedPtr(adaptor.getOutput());

    // Extract shapes from memref types
    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());
    auto AShape = AType.getShape();
    auto BShape = BType.getShape();

    // Compute M, N, K and batch_count from the shapes
    // A: [..., M, K], B: [..., K, N] or B: [K, N] (broadcast)
    int64_t ARank = AShape.size();
    int64_t BRank = BShape.size();

    if (ARank < 2 || BRank < 2) {
      // For rank-1 cases, ONNX has special rules, but for LLM inference
      // we always have rank >= 2
      if (ARank < 1 || BRank < 1)
        return op.emitError("MatMul requires at least rank-1 inputs");
    }

    int64_t M = (ARank >= 2) ? AShape[ARank - 2] : 1;
    int64_t K = AShape[ARank - 1];
    int64_t N = BShape[BRank - 1];

    // Compute batch count from leading dimensions of A
    int64_t batchCount = 1;
    for (int64_t i = 0; i < ARank - 2; ++i) {
      batchCount *= AShape[i];
    }

    // Element size in bytes
    int64_t elemSize =
        AType.getElementType().getIntOrFloatBitWidth() / 8;

    Value m = createI64Const(M);
    Value n = createI64Const(N);
    Value k = createI64Const(K);
    Value batch = createI64Const(batchCount);
    Value elemSizeVal = createI64Const(elemSize);

    // int wrap_hipblasLtMatmul(RuntimeState* state,
    //                          const void* A, const void* B, void* output,
    //                          int64_t M, int64_t N, int64_t K,
    //                          int64_t batch_count, int64_t elem_size)
    SmallVector<Type, 9> paramTypes = {
        ptrType,  // state
        ptrType,  // A
        ptrType,  // B
        ptrType,  // output
        i64Type,  // M
        i64Type,  // N
        i64Type,  // K
        i64Type,  // batch_count
        i64Type   // elem_size
    };

    static constexpr const char* kWraphipblasLtMatmul = "wrap_hipblasLtMatmul";
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWraphipblasLtMatmul, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {
        statePtr, APtr, BPtr, resultPtr,
        m, n, k, batch, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

// NOTE: Main function transformation is handled post-conversion in
// runOnOperation

// --- Pass
struct ConvertHipToLLVMPass
    : public PassWrapper<ConvertHipToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertHipToLLVMPass)

  StringRef getArgument() const final { return "convert-hip-to-llvm"; }
  StringRef getDescription() const final {
    return "Convert HIP dialect to LLVM dialect";
  }

  void getDependentDialects(DialectRegistry& registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<memref::MemRefDialect>();
    registry.insert<arith::ArithDialect>();
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext* ctx = module.getContext();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);

    // Convert !hip.context to !llvm.ptr
    typeConverter.addConversion([ctx](ContextType type) -> Type {
      return LLVM::LLVMPointerType::get(ctx, 0);
    });

    RewritePatternSet patterns(ctx);

    // Add HIP-specific conversion patterns
    patterns
        .add<GetPoolOpLowering, ConvOpLowering, GemmOpLowering, ReluOpLowering,
             MatMulOpLowering, MulOpLowering, SubOpLowering, GatherOpLowering,
             ReduceSumOpLowering, CastOpLowering, SigmoidOpLowering,
             RotaryEmbeddingOpLowering, SimplifiedLayerNormOpLowering,
             SkipSimplifiedLayerNormOpLowering, GetConstantOpLowering,
             GroupQueryAttentionOpLowering>(typeConverter);

    // Add standard MLIR→LLVM conversion patterns.
    // populateFinalizeMemRefToLLVMConversionPatterns is intentionally bundled
    // here rather than run as a separate pipeline stage. The HIP op patterns
    // and the memref patterns share one applyPartialConversion call, so the
    // type converter resolves hip.get_pool → memref.view → hip.conv value
    // chains atomically without inserting unrealized_conversion_cast at the
    // memref/LLVM boundary. Running them as separate stages would require a
    // reconcile-unrealized-casts cleanup pass.
    // See: doc/design/mlir/passes/WHY-BUNDLE-FINALIZE-MEMREF.md
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
    cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<HipDialect>();
    // memref is illegal: populateFinalizeMemRefToLLVMConversionPatterns above
    // converts all memref ops in this same pass. Do not move it to a separate
    // pipeline stage — see WHY-BUNDLE-FINALIZE-MEMREF.md.
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<arith::ArithDialect>();
    target.addIllegalDialect<cf::ControlFlowDialect>();
    // arith.constant with a tensor-typed result must NOT reach this pass.
    // OnnxToHip must convert every ONNX constant to hip.get_constant so that
    // no tensor-typed arith.constant survives bufferization. If one does,
    // applyPartialConversion will fail with a clear diagnostic — which is the
    // correct behaviour: it surfaces a bug in convert-onnx-to-hip rather than
    // silently wrapping the stray constant in unrealized_conversion_cast.

    target.addLegalOp<ModuleOp>();

    // FuncOp is legal only if types are converted
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp op) {
      return typeConverter.isSignatureLegal(op.getFunctionType());
    });

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();

    // Post-processing: Transform @main_graph function signature
    // After standard conversion
    // (populateFinalizeMemRefToLLVMConversionPatterns),
    // @main_graph has memrefs unpacked to scalar parameters (23 params for 1
    // input + 1 output rank-4) We wrap it: @main_graph (3 params, struct
    // arrays) →
    // @main_graph_internal (23 params, scalars)
    if (failed(transformMainFunction(module)))
      signalPassFailure();
  }

private:
  /// Returns LLVM struct type for memref: (ptr, ptr, i64, array<rank x i64>,
  /// array<rank x i64>)
  Type getMemRefStructType(OpBuilder& builder, int64_t rank,
                           unsigned addrSpace) {
    MLIRContext* ctx = builder.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, addrSpace);
    Type i64Type = builder.getI64Type();
    Type sizeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Type strideArrayType = LLVM::LLVMArrayType::get(i64Type, rank);

    return LLVM::LLVMStructType::getLiteral(
        ctx, {ptrType, ptrType, i64Type, sizeArrayType, strideArrayType});
  }

  /// Unpacks memref struct into scalar values (2 + 1 + rank + rank)
  void unpackMemRefStruct(OpBuilder& builder, Location loc, Value memrefStruct,
                          int64_t rank, SmallVectorImpl<Value>& args) {
    // Extract allocated pointer (field 0)
    args.push_back(builder.create<LLVM::ExtractValueOp>(loc, memrefStruct,
                                                        ArrayRef<int64_t>{0}));

    // Extract aligned pointer (field 1)
    args.push_back(builder.create<LLVM::ExtractValueOp>(loc, memrefStruct,
                                                        ArrayRef<int64_t>{1}));

    // Extract offset (field 2)
    args.push_back(builder.create<LLVM::ExtractValueOp>(loc, memrefStruct,
                                                        ArrayRef<int64_t>{2}));

    // Extract sizes (field 3, array elements 0..rank-1)
    for (int64_t dim = 0; dim < rank; dim++) {
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{3, dim}));
    }

    // Extract strides (field 4, array elements 0..rank-1)
    for (int64_t dim = 0; dim < rank; dim++) {
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{4, dim}));
    }
  }

  /// Unpack a memref struct, casting the two pointer fields to AS 0.
  /// Used when the struct was built with AS 1 (GPU) pointers but the callee
  /// expects AS 0 (one-shot-bufferize function-boundary convention).
  void unpackMemRefStructWithAddrCast(OpBuilder& builder, Location loc,
                                      Value memrefStruct, int64_t rank,
                                      SmallVectorImpl<Value>& args) {
    Type as0PtrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);

    auto castToAs0 = [&](Value ptr) -> Value {
      auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
      if (ptrTy.getAddressSpace() == 0)
        return ptr;
      return builder.create<LLVM::AddrSpaceCastOp>(loc, as0PtrType, ptr);
    };

    // Field 0: allocated pointer — cast AS 1 → AS 0
    Value allocPtr = builder.create<LLVM::ExtractValueOp>(
        loc, memrefStruct, ArrayRef<int64_t>{0});
    args.push_back(castToAs0(allocPtr));

    // Field 1: aligned pointer — cast AS 1 → AS 0
    Value alignedPtr = builder.create<LLVM::ExtractValueOp>(
        loc, memrefStruct, ArrayRef<int64_t>{1});
    args.push_back(castToAs0(alignedPtr));

    // Field 2: offset (i64, no cast needed)
    args.push_back(builder.create<LLVM::ExtractValueOp>(
        loc, memrefStruct, ArrayRef<int64_t>{2}));

    // Fields 3 & 4: sizes and strides arrays
    for (int64_t dim = 0; dim < rank; dim++)
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{3, dim}));
    for (int64_t dim = 0; dim < rank; dim++)
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{4, dim}));
  }

  /// Transform @main_graph from unpacked memrefs to array-based interface
  LogicalResult transformMainFunction(ModuleOp module) {
    // Find @main_graph function
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!mainFunc) {
      return success(); // No main_graph function - this is fine
    }

    // Read metadata
    auto inputCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.input_count");
    auto outputCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.output_count");
    auto inputShapesAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.input_shapes");
    auto outputShapesAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");

    if (!inputCountAttr || !outputCountAttr || !inputShapesAttr ||
        !outputShapesAttr) {
      llvm::errs()
          << "[HipToLLVM] Warning: No metadata found, skipping @main_graph "
             "transformation\n";
      return success(); // Graceful degradation
    }

    int64_t inputCount = inputCountAttr.getInt();
    int64_t outputCount = outputCountAttr.getInt();

    // Validate metadata
    if ((int64_t)inputShapesAttr.size() != inputCount ||
        (int64_t)outputShapesAttr.size() != outputCount) {
      return module.emitError("Metadata mismatch: shapes array size != count");
    }

    // Calculate expected parameter count (1 context + unpacked memrefs)
    // Rank of tensor i is derived from the length of its shape array.
    unsigned expectedParams = 1; // context
    for (auto shapeAttr : inputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedParams +=
          2 + 1 + rank + rank; // 2 ptrs + offset + sizes + strides
    }
    for (auto shapeAttr : outputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedParams += 2 + 1 + rank + rank;
    }

    unsigned actualParams = mainFunc.getFunctionType().getNumParams();
    if (actualParams != expectedParams) {
      return module.emitError()
             << "[HipToLLVM] Parameter count mismatch: expected "
             << expectedParams << ", got " << actualParams;
    }

    OpBuilder builder(module.getContext());
    Location loc = mainFunc.getLoc();

    // Rename @main_graph → @main_graph_internal (make private)
    mainFunc.setName("main_graph_internal");
    mainFunc.setLinkage(LLVM::Linkage::Private);

    // Create new @main_graph with array-based interface
    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> newParamTypes = {ptrType, ptrType, ptrType};
    auto newFuncType = LLVM::LLVMFunctionType::get(i32Type, newParamTypes);

    builder.setInsertionPoint(mainFunc);
    auto newMainFunc =
        builder.create<LLVM::LLVMFuncOp>(loc, "main_graph", newFuncType);
    newMainFunc.setLinkage(LLVM::Linkage::Private);

    // CRITICAL WORKAROUND: Prevent LLVM SROA optimizer bug
    //
    // Without noinline, LLVM inlines @main_graph into inference_compute, then
    // SROA (Scalar Replacement of Aggregates) tries to optimize memref
    // structures. SROA fails to track GPU pointers through double indirection
    // pattern:
    //   array (AS0) → struct (AS0) → GPU pointer (AS1)
    // and produces 'undef' instead, causing segfault at runtime.
    //
    // This is a known LLVM limitation that persists even in LLVM 22 with
    // opaque pointers. The noinline attribute prevents inlining, so SROA
    // never sees the full pattern and doesn't attempt the broken optimization.
    //
    // Performance impact: One function call overhead (~5-10 cycles), negligible
    // compared to GPU operations (millions of cycles).
    //
    // See: doc/design/mlir/passes/LLVM-SROA-WORKAROUND.md for full explanation
    newMainFunc->setAttr(
        "passthrough",
        builder.getArrayAttr({builder.getStringAttr("noinline")}));

    Block* entryBlock = newMainFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value ctxArg = entryBlock->getArgument(0);     // %context
    Value inputsArg = entryBlock->getArgument(1);  // %inputs
    Value outputsArg = entryBlock->getArgument(2); // %outputs

    // Build arguments for @main_graph_internal
    SmallVector<Value> mainInternalArgs;
    mainInternalArgs.push_back(ctxArg); // arg0: context

    // Unpack inputs — the arrays contain pointers to memref structs (ptr[]),
    // so we load the pointer first, then load the struct through it.
    for (int64_t i = 0; i < inputCount; i++) {
      int64_t rank = cast<DenseI64ArrayAttr>(inputShapesAttr[i]).size();

      // GEP to get pointer to inputs[i] (the array stores pointers to
      // stack-allocated memref structs, so two loads are needed)
      Value inputIdxVal = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(i));
      Value inputSlotPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, inputsArg, ValueRange{inputIdxVal});

      Value inputStructPtr =
          builder.create<LLVM::LoadOp>(loc, ptrType, inputSlotPtr);

      // GenerateInterface builds GPU (AS 1) memref structs from hipMalloc
      // pointers, so we must load them as AS 1 here.
      Type memrefStructType = getMemRefStructType(builder, rank, 1);
      Value inputMemref =
          builder.create<LLVM::LoadOp>(loc, memrefStructType, inputStructPtr);

      // @main_graph_internal has AS 0 params (one-shot-bufferize function
      // boundary), so cast the two pointer fields from AS 1 → AS 0.
      unpackMemRefStructWithAddrCast(builder, loc, inputMemref, rank,
                                     mainInternalArgs);
    }

    // Unpack outputs — same ptr[] indirection as inputs.
    // GenerateInterface builds GPU (AS 1) memref structs, so load as AS 1.
    for (int64_t i = 0; i < outputCount; i++) {
      int64_t rank = cast<DenseI64ArrayAttr>(outputShapesAttr[i]).size();

      Value outputIdxVal = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(i));
      Value outputSlotPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, outputsArg, ValueRange{outputIdxVal});

      Value outputStructPtr =
          builder.create<LLVM::LoadOp>(loc, ptrType, outputSlotPtr);

      Type memrefStructType = getMemRefStructType(builder, rank, 1);
      Value outputMemref =
          builder.create<LLVM::LoadOp>(loc, memrefStructType, outputStructPtr);

      unpackMemRefStructWithAddrCast(builder, loc, outputMemref, rank,
                                     mainInternalArgs);
    }

    // Call @main_graph_internal with unpacked arguments.
    // If the internal function is void (buffer-results-to-out-params converted
    // outputs to out-params), return i32 0 (success) from the wrapper.
    auto internalRetTy = mainFunc.getFunctionType().getReturnType();
    if (isa<LLVM::LLVMVoidType>(internalRetTy)) {
      builder.create<LLVM::CallOp>(loc, mainFunc, mainInternalArgs);
      Value zero = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(0));
      builder.create<LLVM::ReturnOp>(loc, zero);
    } else {
      auto callOp =
          builder.create<LLVM::CallOp>(loc, mainFunc, mainInternalArgs);
      builder.create<LLVM::ReturnOp>(loc, callOp.getResult());
    }

    llvm::errs() << "[HipToLLVM] Transformed @main_graph signature: "
                 << actualParams << " params → 3 params\n";
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertHipToLLVMPass() {
  return std::make_unique<ConvertHipToLLVMPass>();
}

} // namespace hip
} // namespace mlir
