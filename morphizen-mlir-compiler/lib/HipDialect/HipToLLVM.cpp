/*
 * Copyright (C) 2023 - 2025 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipDialect.h"
#include "HipPasses.h"
#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/FuncToLLVM/ConvertFuncToLLVM.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Conversion/MemRefToLLVM/MemRefToLLVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/Transforms/Passes.h"
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

namespace mlir {
namespace hip {

namespace {

static constexpr const char* kHipMalloc = "hipMalloc";
static constexpr const char* kHipFree = "hipFree";
static constexpr const char* kWrapHipMemcpyAsync = "wrap_hipMemcpyAsync";
static constexpr const char* kMiopenConvolutionForward =
    "wrap_miopenConvolutionForward";
static constexpr const char* kHipGetConstant = "hipdnn_ep_constant_get";
static constexpr const char* kHipGetBufferFromPool =
    "hipdnn_ep_get_buffer_from_pool";

// Helper to get buffer index for an AllocOp from module metadata
// Returns the index assigned by MemoryPoolingPass, or -1 if not found
static int64_t getBufferIndexForAlloc(ModuleOp module, AllocOp allocOp) {
  // Count allocations in deterministic order (same as MemoryPoolingPass)
  int64_t currentIndex = 0;
  int64_t targetIndex = -1;

  for (auto funcOp : module.getOps<func::FuncOp>()) {
    funcOp.walk([&](AllocOp op) {
      if (op == allocOp) {
        targetIndex = currentIndex;
        return WalkResult::interrupt();
      }
      currentIndex++;
      return WalkResult::advance();
    });
    if (targetIndex >= 0) {
      break;
    }
  }

  return targetIndex;
}

// --- AllocOp: hip.alloc(%handle, %dyn...) -> hipMalloc(bytes) + memref
// descriptor
struct AllocOpLowering : public ConvertOpToLLVMPattern<AllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MemRefType memRefType = op.getMemref().getType();

    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    Type indexType = getIndexType();
    Type ptrType = getPtrType();
    Type i32Type = IntegerType::get(getContext(), 32);
    Type i64Type = IntegerType::get(getContext(), 64);

    // Compute sizes and strides for memref descriptor
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, sizeBytes, true);

    Value allocatedPtr;

    // Check if memory pooling is enabled (module has pool metadata)
    auto poolSizeAttr = module->getAttrOfType<IntegerAttr>("hipdnn.pool_size");
    if (poolSizeAttr) {
      llvm::errs()
          << "[HipToLLVM] Pool metadata found, using pool-based allocation\n";
      // Pool-based allocation
      // Get buffer index from the operation's attribute (set by
      // MemoryPoolingPass)
      auto bufferIndexAttr =
          op->getAttrOfType<IntegerAttr>("hipdnn.buffer_index");
      if (!bufferIndexAttr) {
        llvm::errs() << "[HipToLLVM] ERROR: hip.alloc missing "
                        "hipdnn.buffer_index attribute!\n";
        return rewriter.notifyMatchFailure(
            op, "hip.alloc operation missing hipdnn.buffer_index attribute");
      }
      int64_t bufferIndex = bufferIndexAttr.getInt();
      llvm::errs() << "[HipToLLVM] Buffer index for this alloc: " << bufferIndex
                   << "\n";

      // Call hipdnn_ep_get_buffer_from_pool(state, index)
      FailureOr<LLVM::LLVMFuncOp> getBufferFn = LLVM::lookupOrCreateFn(
          rewriter, module, kHipGetBufferFromPool, {ptrType, i64Type}, ptrType);
      if (failed(getBufferFn))
        return failure();

      Value indexValue = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(bufferIndex));
      Value statePtr = adaptor.getHandle(); // RuntimeState*

      allocatedPtr = LLVM::CallOp::create(rewriter, loc, *getBufferFn,
                                          {statePtr, indexValue})
                         .getResult();
    } else {
      // Direct allocation (fallback)
      // Declare hipMalloc with CORRECT signature: (ptr, i64) -> i32
      FailureOr<LLVM::LLVMFuncOp> mallocFn = LLVM::lookupOrCreateFn(
          rewriter, module, kHipMalloc, {ptrType, indexType}, i32Type);
      if (failed(mallocFn))
        return failure();

      // Allocate stack space for the returned pointer
      Value one = rewriter.create<LLVM::ConstantOp>(loc, indexType,
                                                    rewriter.getIndexAttr(1));
      Value ptrStorage = rewriter.create<LLVM::AllocaOp>(loc, ptrType, ptrType,
                                                         one, /*alignment=*/8);

      // Call hipMalloc(&ptrStorage, sizeBytes)
      Value mallocResult = LLVM::CallOp::create(rewriter, loc, *mallocFn,
                                                {ptrStorage, sizeBytes})
                               .getResult();

      // TODO: Check mallocResult for errors (hipSuccess == 0)
      // For now, assume success

      // Load the allocated pointer from ptrStorage
      allocatedPtr = rewriter.create<LLVM::LoadOp>(loc, ptrType, ptrStorage);
    }

    // Cast to memref address space if needed
    Type elementPtrType = getElementPtrType(memRefType);
    if (!elementPtrType)
      return rewriter.notifyMatchFailure(op,
                                         "could not compute element ptr type");
    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();
    if (cast<LLVM::LLVMPointerType>(allocatedPtr.getType()).getAddressSpace() !=
        *addrSpace)
      allocatedPtr = rewriter.create<LLVM::AddrSpaceCastOp>(
          loc, LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          allocatedPtr);

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, allocatedPtr, allocatedPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// --- FreeOp: hip.free(%handle, %memref) -> llvm.call @hipFree(allocated_ptr)
struct FreeOpLowering : public ConvertOpToLLVMPattern<FreeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    // Check if memory pooling is enabled
    auto poolSizeAttr = module->getAttrOfType<IntegerAttr>("hipdnn.pool_size");
    if (poolSizeAttr) {
      // Pooling enabled - free is a nop (pool freed in cleanup)
      rewriter.eraseOp(op);
      return success();
    }

    // Direct allocation - call hipFree
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kHipFree, ptrType, voidType);
    if (failed(funcOp))
      return failure();

    Value memrefDesc = adaptor.getMemref();
    Value allocatedPtr =
        MemRefDescriptor(memrefDesc).allocatedPtr(rewriter, loc);
    // hipFree expects void*; if memref is in non-default address space, cast
    auto ptrTy = allocatedPtr.getType();
    if (cast<LLVM::LLVMPointerType>(ptrTy).getAddressSpace() != 0)
      allocatedPtr =
          rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, allocatedPtr);

    LLVM::CallOp::create(rewriter, loc, *funcOp, allocatedPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

// --- CopyOp: hip.copy(%ctx, %src, %dst) ->
//             llvm.call @wrap_hipMemcpyAsync(ctx, src_ptr, dst_ptr, size)
struct CopyOpLowering : public ConvertOpToLLVMPattern<CopyOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter& rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();

    Type ptrType = getPtrType();
    Type i32Type = IntegerType::get(getContext(), 32);
    Type i64Type = IntegerType::get(getContext(), 64);

    // Get source and destination memref descriptors
    Value srcMemrefDesc = adaptor.getSource();
    Value dstMemrefDesc = adaptor.getDestination();

    MemRefDescriptor srcDesc(srcMemrefDesc);
    MemRefDescriptor dstDesc(dstMemrefDesc);

    // Extract aligned pointers
    Value srcPtr = srcDesc.alignedPtr(rewriter, loc);
    Value dstPtr = dstDesc.alignedPtr(rewriter, loc);

    // Cast to generic pointer (address space 0) if needed
    auto srcPtrType = cast<LLVM::LLVMPointerType>(srcPtr.getType());
    auto dstPtrType = cast<LLVM::LLVMPointerType>(dstPtr.getType());
    if (srcPtrType.getAddressSpace() != 0)
      srcPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, srcPtr);
    if (dstPtrType.getAddressSpace() != 0)
      dstPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, dstPtr);

    // Calculate size: multiply all dimensions by element size
    MemRefType srcMemRefType = op.getSource().getType();
    Type elementType = srcMemRefType.getElementType();
    unsigned elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;

    Value sizeBytes = rewriter.create<LLVM::ConstantOp>(
        loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    for (int64_t dim : srcMemRefType.getShape()) {
      if (dim == ShapedType::kDynamic) {
        return rewriter.notifyMatchFailure(
            op, "dynamic shapes not yet supported in hip.copy lowering");
      }
      Value dimVal = rewriter.create<LLVM::ConstantOp>(
          loc, i64Type, rewriter.getI64IntegerAttr(dim));
      sizeBytes = rewriter.create<LLVM::MulOp>(loc, i64Type, sizeBytes, dimVal);
    }

    // Declare/lookup runtime function
    // int wrap_hipMemcpyAsync(void* state, void* dst, void* src, size_t size)
    SmallVector<Type> argTypes = {ptrType, ptrType, ptrType, i64Type};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipMemcpyAsync, argTypes, i32Type);
    if (failed(funcOp))
      return failure();

    // Call runtime function
    Value state = adaptor.getCtx();
    SmallVector<Value> callOperands = {state, dstPtr, srcPtr, sizeBytes};
    // hip.copy has no results, but we still call the runtime function
    // (ignoring the return value since copy is a void operation)
    rewriter.create<LLVM::CallOp>(op.getLoc(), *funcOp, callOperands);
    rewriter.eraseOp(op);

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
    Value resultPtr = getAlignedPtr(adaptor.getResult());

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
        .add<AllocOpLowering, FreeOpLowering, CopyOpLowering, ConvOpLowering,
             GemmOpLowering, ReluOpLowering, GetConstantOpLowering>(
            typeConverter);

    // Add standard MLIR→LLVM conversion patterns
    populateFuncToLLVMConversionPatterns(typeConverter, patterns);
    populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
    arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);

    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<HipDialect>();
    target.addIllegalDialect<memref::MemRefDialect>();
    target.addIllegalDialect<arith::ArithDialect>();

    // Allow only tensor-typed arith.constant operations
    // Scalar arith.constant (i32, i64, etc.) must be converted to
    // llvm.mlir.constant Tensor constants are wrapped by
    // unrealized_conversion_cast and can stay as-is
    target.addDynamicallyLegalOp<arith::ConstantOp>([&](arith::ConstantOp op) {
      Type resultType = op.getType();
      // Legal only if it's a tensor constant (wrapped by casts)
      // Scalar constants must be converted to llvm.mlir.constant
      return isa<TensorType>(resultType);
    });

    // Allow unrealized_conversion_cast (will be cleaned up later)
    target.addLegalOp<UnrealizedConversionCastOp>();

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
    auto inputRanksAttr =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.input_ranks");
    auto outputRanksAttr =
        module->getAttrOfType<DenseI64ArrayAttr>("hipdnn.output_ranks");

    if (!inputCountAttr || !outputCountAttr || !inputRanksAttr ||
        !outputRanksAttr) {
      llvm::errs()
          << "[HipToLLVM] Warning: No metadata found, skipping @main_graph "
             "transformation\n";
      return success(); // Graceful degradation
    }

    int64_t inputCount = inputCountAttr.getInt();
    int64_t outputCount = outputCountAttr.getInt();
    auto inputRanks = inputRanksAttr.asArrayRef();
    auto outputRanks = outputRanksAttr.asArrayRef();

    // Validate metadata
    if (inputRanks.size() != inputCount || outputRanks.size() != outputCount) {
      return module.emitError("Metadata mismatch: ranks array size != count");
    }

    // Calculate expected parameter count (1 context + unpacked memrefs)
    unsigned expectedParams = 1; // context
    for (int64_t rank : inputRanks) {
      expectedParams +=
          2 + 1 + rank + rank; // 2 ptrs + offset + sizes + strides
    }
    for (int64_t rank : outputRanks) {
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

    // Unpack inputs
    for (int64_t i = 0; i < inputCount; i++) {
      int64_t rank = inputRanks[i];

      // GEP to get pointer to inputs[i]
      Value inputIdxVal = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(i));
      Value inputStructPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, inputsArg, ValueRange{inputIdxVal});

      // Load memref struct from array
      Type memrefStructType =
          getMemRefStructType(builder, rank, 1); // addr space 1 (GPU)
      Value inputMemref =
          builder.create<LLVM::LoadOp>(loc, memrefStructType, inputStructPtr);

      // Extract fields (for rank-4: 2 ptrs + offset + 4 sizes + 4 strides = 11)
      unpackMemRefStruct(builder, loc, inputMemref, rank, mainInternalArgs);
    }

    // Unpack outputs
    for (int64_t i = 0; i < outputCount; i++) {
      int64_t rank = outputRanks[i];

      Value outputIdxVal = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(i));
      Value outputStructPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, outputsArg, ValueRange{outputIdxVal});

      Type memrefStructType = getMemRefStructType(builder, rank, 1);
      Value outputMemref =
          builder.create<LLVM::LoadOp>(loc, memrefStructType, outputStructPtr);

      unpackMemRefStruct(builder, loc, outputMemref, rank, mainInternalArgs);
    }

    // Call @main_graph_internal with unpacked arguments
    auto callOp = builder.create<LLVM::CallOp>(loc, mainFunc, mainInternalArgs);
    Value result = callOp.getResult();

    // Return the result
    builder.create<LLVM::ReturnOp>(loc, result);

    llvm::errs() << "[HipToLLVM] Transformed @main_graph signature: "
                 << actualParams << " params → 3 params\n";
    return success();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertHipToLLVMPass() {
  return std::make_unique<ConvertHipToLLVMPass>();
}

void registerHipPasses() {
  // Register all HIP-related conversion passes
  // Note: Pass registration uses the getArgument() string from each pass class

  // ConvertOnnxToHipPass (defined in OnnxToHip.cpp)
  // Registered via: --convert-onnx-to-hip
  registerPass(
      []() -> std::unique_ptr<Pass> { return createConvertOnnxToHipPass(); });

  // ConvertHipToLLVMPass (defined in this file)
  // Registered via: --convert-hip-to-llvm
  PassRegistration<ConvertHipToLLVMPass>();

  // MemoryPoolingPass (defined in MemoryPoolingPass.cpp)
  // Registered via: --memory-pooling
  registerPass(
      []() -> std::unique_ptr<Pass> { return createMemoryPoolingPass(); });

  // GenerateInterfacePass (defined in GenerateInterfacePass.cpp)
  // Registered via: --generate-interface
  registerPass(
      []() -> std::unique_ptr<Pass> { return createGenerateInterfacePass(); });

  // HipBufferDeallocationPass (defined in HipBufferDeallocationPass.cpp)
  // Registered via: --hip-buffer-deallocation
  registerPass([]() -> std::unique_ptr<Pass> {
    return createHipBufferDeallocationPass();
  });

  // Complete ONNX→HIP→LLVM→Interface pipeline
  // Registered via PassPipelineRegistration as: --all-passes
  PassPipelineRegistration<>(
      "all-passes",
      "Run complete ONNX→HIP→BufferDeallocation→LLVM→Interface pipeline",
      populateCompleteOnnxToLLVMPipeline);
}

void populateCompleteOnnxToLLVMPipeline(OpPassManager& pm) {
  // ONNX → HIP conversion
  pm.addPass(createConvertOnnxToHipPass());

  // BufferDeallocation pipeline (nested under func.func)
  auto& funcPM = pm.nest<func::FuncOp>();
  funcPM.addPass(bufferization::createBufferLoopHoistingPass());
  funcPM.addPass(bufferization::createOwnershipBasedBufferDeallocationPass());
  funcPM.addPass(bufferization::createOptimizeAllocationLivenessPass());

  // Canonicalization (back at module level)
  pm.addPass(createCanonicalizerPass());

  // Memory pooling optimization
  pm.addPass(createMemoryPoolingPass());

  // HIP → LLVM conversion
  pm.addPass(createConvertHipToLLVMPass());

  // Generate interface
  pm.addPass(createGenerateInterfacePass());
}

} // namespace hip
} // namespace mlir
