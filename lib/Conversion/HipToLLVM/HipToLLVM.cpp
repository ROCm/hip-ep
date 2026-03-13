/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
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
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTHIPTOLLVMPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static constexpr const char *kHipMalloc = "hip_device_malloc";
static constexpr const char *kHipFree = "hip_device_free";
static constexpr const char *kHipGetPoolBase = "hipdnn_ep_get_pool_base";

static constexpr const char *kMiopenConvolutionForward =
    "wrap_miopenConvolutionForward";
static constexpr const char *kHipblasltMatmul = "hip_hipblaslt_matmul";
static constexpr const char *kWrapMiopenT5LayerNormForward =
    "wrap_miopenT5LayerNormForward";
static constexpr const char *kWrapMiopenAddT5LayerNormForward =
    "wrap_miopenAddT5LayerNormForward";
static constexpr const char *kMiopenAdd = "hip_miopen_add";
static constexpr const char *kMiopenMul = "hip_miopen_mul";
static constexpr const char *kMiopenSoftmax = "hip_miopen_softmax";
static constexpr const char *kHipTranspose = "hip_transpose";
static constexpr const char *kHipGather = "hip_gather";
static constexpr const char *kHipSilu = "hip_silu";
static constexpr const char *kWrapMiopenActivationForward =
    "wrap_miopenActivationForward"; // hip.sigmoid
static constexpr const char *kWrapElementwiseSub = "wrap_elementwise_sub";
static constexpr const char *kWrapMiopenOpTensor =
    "wrap_miopenOpTensor"; // hip.mul
static constexpr const char *kWrapMiopenCast = "wrap_miopenCast";
static constexpr const char *kWrapReduceSum = "wrap_reduce_sum";
static constexpr const char *kHipGqa = "hip_gqa";

// Maps MLIR element type to runtime data type enum (HIPDNN_EP_DATATYPE_*).
// Values must match the #defines in hipdnn_ep_runtime.h.
// Returns -1 for unsupported types.
static int64_t getHipdnnDataType(Type elemType) {
  if (elemType.isF32())
    return 0; // HIPDNN_EP_DATATYPE_FLOAT
  if (elemType.isF16())
    return 1; // HIPDNN_EP_DATATYPE_HALF
  if (elemType.isBF16())
    return 2; // HIPDNN_EP_DATATYPE_BFLOAT16
  return -1;
}

// Tensor operation types (must match runtime enum).
// HIPDNN_EP_TENSOR_OP_* values.
enum class TensorOp : int64_t {
  Sub = 0, // Subtraction: C = A - B
  Add = 1, // Addition: C = A + B
  Mul = 2, // Multiplication: C = A * B
};

// Helper: extract the aligned data pointer from a converted memref descriptor,
// casting to address space 0 if needed.
// Uses alignedPtr (not allocatedPtr) so that memref.view offsets into a memory
// pool are respected -- each view has the same allocatedPtr but a distinct
// alignedPtr.
static Value extractMemRefPtr(Value memrefDesc,
                              ConversionPatternRewriter &rewriter,
                              Location loc) {
  Value ptr = MemRefDescriptor(memrefDesc).alignedPtr(rewriter, loc);
  auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
  if (ptrTy.getAddressSpace() != 0)
    ptr = LLVM::AddrSpaceCastOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 0),
        ptr);
  return ptr;
}

// Helper: compute total number of elements in a memref, handling both static
// and dynamic dimensions.
static Value computeNumElements(MemRefType type, Value descriptor,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {
  MemRefDescriptor desc(descriptor);
  Type i64Type = rewriter.getI64Type();
  Value num = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));

  for (auto dimIdx : llvm::seq<int64_t>(type.getRank())) {
    Value dimSize;
    if (type.isDynamicDim(dimIdx)) {
      dimSize = desc.size(rewriter, loc, dimIdx);
    } else {
      dimSize = LLVM::ConstantOp::create(
          rewriter, loc, i64Type,
          rewriter.getI64IntegerAttr(type.getDimSize(dimIdx)));
    }
    num = LLVM::MulOp::create(rewriter, loc, num, dimSize);
  }
  return num;
}

// ===== Convolution ops ================================

// hip.conv(%ctx, %input, %weights, %bias, %output)
//   -> wrap_miopenConvolutionForward(ctx, input, input_n, input_c, input_h,
//                                     input_w, weights, weights_k, bias,
//                                     output, output_h, output_w, kernel_h,
//                                     kernel_w, stride_h, stride_w, pad_top,
//                                     pad_left, pad_bottom, pad_right,
//                                     dilation_h, dilation_w, group)
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
    //     int64_t group           // Number of groups
    // );
    //
    // Returns: 0 on success, non-zero on error

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract memref pointers (aligned pointers from descriptors)
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      Value ptr = MemRefDescriptor(memrefDesc).alignedPtr(rewriter, loc);
      // Cast to void* (address space 0) if needed
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getCtx(); // RuntimeState* (opaque)
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value weightsPtr = getAlignedPtr(adaptor.getWeights());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    // Handle optional bias
    Value biasPtr;
    if (adaptor.getBias()) {
      biasPtr = getAlignedPtr(adaptor.getBias());
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

    // Helper: Get dimension value (static constant or dynamic descriptor size)
    auto getDim = [&](MemRefDescriptor &desc, MemRefType type,
                      unsigned dimIdx) -> Value {
      int64_t dimSize = type.getShape()[dimIdx];
      if (!ShapedType::isDynamic(dimSize)) {
        // Static dimension: use compile-time constant
        return createI64Const(dimSize);
      }
      // Dynamic dimension: extract from runtime descriptor
      return desc.size(rewriter, loc, dimIdx);
    };

    // Create descriptors for accessing runtime sizes
    MemRefDescriptor inputDesc(adaptor.getInput());
    MemRefDescriptor weightsDesc(adaptor.getWeights());
    MemRefDescriptor outputDesc(adaptor.getOutput());

    // Input shape: [N, C, H, W]
    Value inputN = getDim(inputDesc, inputType, 0);
    Value inputC = getDim(inputDesc, inputType, 1);
    Value inputH = getDim(inputDesc, inputType, 2);
    Value inputW = getDim(inputDesc, inputType, 3);

    // Weights shape: [K, C, R, S] where K=output channels
    Value weightsK = getDim(weightsDesc, weightsType, 0);

    // Output shape: [N, K, H', W']
    Value outputH = getDim(outputDesc, outputType, 2);
    Value outputW = getDim(outputDesc, outputType, 3);

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

// --- AllocOp: hip.alloc(%ctx, %dyn...) -> hipMalloc(bytes) + memref descriptor
struct AllocOpLowering : public ConvertOpToLLVMPattern<AllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MemRefType memRefType = op.getMemref().getType();

    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    // Declare hipMalloc(size: i64) -> ptr
    Type indexType = getIndexType();
    Type ptrType = getPtrType();
    FailureOr<LLVM::LLVMFuncOp> mallocFn = LLVM::lookupOrCreateFn(
        rewriter, module, kHipMalloc, indexType, ptrType);
    if (failed(mallocFn))
      return failure();

    // Compute sizes and sizeBytes (dynamic sizes are after the ctx).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, sizeBytes, true);

    Value allocatedPtr =
        LLVM::CallOp::create(rewriter, loc, *mallocFn, sizeBytes).getResult();

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
      allocatedPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          allocatedPtr);

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, allocatedPtr, allocatedPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// --- FreeOp: hip.free(%ctx, %memref) -> llvm.call @hipFree(allocated_ptr)
struct FreeOpLowering : public ConvertOpToLLVMPattern<FreeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(FreeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
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
          LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, allocatedPtr);

    LLVM::CallOp::create(rewriter, loc, *funcOp, allocatedPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

// --- GetPoolOp: hip.get_pool(%ctx, %pool_size) : memref<?xi8>
//     -> llvm.call @hipdnn_ep_get_pool_base(state, size) + memref descriptor
struct GetPoolOpLowering : public ConvertOpToLLVMPattern<GetPoolOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    MemRefType memRefType = cast<MemRefType>(op.getPool().getType());

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGetPoolBase, {ptrType, i64Type}, ptrType);
    if (failed(funcOp))
      return failure();

    Value poolSize = adaptor.getPoolSize();
    Value rawPtr = LLVM::CallOp::create(rewriter, loc, *funcOp,
                                        ValueRange{adaptor.getCtx(), poolSize})
                       .getResult();

    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();

    Value gpuPtr = rawPtr;
    if (cast<LLVM::LLVMPointerType>(rawPtr.getType()).getAddressSpace() !=
        *addrSpace)
      gpuPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          rawPtr);

    Value stride1 = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, gpuPtr, gpuPtr, {poolSize}, {stride1}, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// ===== Region ops: inline body and erase =====================================

template <typename OpTy>
struct GraphRegionOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Block &body = op.getBody().front();
    rewriter.inlineBlockBefore(&body, op);
    rewriter.eraseOp(op);
    return success();
  }
};

using MiopenGraphOpLowering = GraphRegionOpLowering<MiopenGraphOp>;
using HipblasltGraphOpLowering = GraphRegionOpLowering<HipblasltGraphOp>;

// ===== hipBLASLt ops =========================================================

// hip.hipblaslt.matmul(handle) ins(A, B) outs(C)
//   -> hip_hipblaslt_matmul(handle, A, B, C, rankA, rankB, batch, M, K, N)
// Rank-generic: batch from A if 3D, B broadcast if rankB < rankA.
struct HipblasltMatmulOpLowering
    : public ConvertOpToLLVMPattern<HipblasltMatmulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(HipblasltMatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    // (handle, A, B, C, rankA, rankB, batch, M, K, N)
    SmallVector<Type> paramTypes = {ptrType,   ptrType,   ptrType,   ptrType,
                                    indexType, indexType, indexType, indexType,
                                    indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipblasltMatmul, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    int rankA = cast<MemRefType>(op.getA().getType()).getRank();
    int rankB = cast<MemRefType>(op.getB().getType()).getRank();
    MemRefDescriptor aDesc(adaptor.getA());
    MemRefDescriptor bDesc(adaptor.getB());

    Value one = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                         rewriter.getIndexAttr(1));
    Value rankAVal = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                              rewriter.getIndexAttr(rankA));
    Value rankBVal = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                              rewriter.getIndexAttr(rankB));

    Value batch = (rankA == 3) ? aDesc.size(rewriter, loc, 0) : one;
    Value M = aDesc.size(rewriter, loc, rankA - 2);
    Value K = aDesc.size(rewriter, loc, rankA - 1);
    Value N = bDesc.size(rewriter, loc, rankB - 1);

    SmallVector<Value> args = {adaptor.getCtx(),
                               extractMemRefPtr(adaptor.getA(), rewriter, loc),
                               extractMemRefPtr(adaptor.getB(), rewriter, loc),
                               extractMemRefPtr(adaptor.getC(), rewriter, loc),
                               rankAVal,
                               rankBVal,
                               batch,
                               M,
                               K,
                               N};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// ===== MIOpen ops ============================================================

// hip.miopen.rms_norm(%handle) ins(%input, %weight) outs(%output)
//   -> hip_miopen_rms_norm(handle, input, weight, output, N, D)
// Rank-generic: N = product of all dims except last, D = last dim.
// For 3D [B,S,D]: N = B*S, D = D.
struct RmsNormOpLowering : public ConvertOpToLLVMPattern<RmsNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value scalePtr = extractMemRefPtr(adaptor.getScale(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Compute num_elements with dynamic shape support
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    auto scaleType = cast<MemRefType>(op.getScale().getType());
    Value scaleNumElements =
        computeNumElements(scaleType, adaptor.getScale(), rewriter, loc);

    // Compute element_size_bytes based on element type
    Type elementType = inputType.getElementType();
    unsigned elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;
    Value elementSizeBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // Extract attributes
    Value axisVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getAxis()));
    Value epsilonVal =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());
    Value stashTypeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getStashType()));

    // Runtime function signature (10 params)
    SmallVector<Type> paramTypes = {
        ptrType, ptrType, ptrType, ptrType, // state, input, scale, output
        i64Type, i64Type, i64Type, // input_num_elements, scale_num_elements,
                                   // element_size_bytes
        i64Type, f32Type, i64Type  // axis, epsilon, stash_type
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapMiopenT5LayerNormForward,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,
                               inputPtr,
                               scalePtr,
                               outputPtr,
                               inputNumElements,
                               scaleNumElements,
                               elementSizeBytesVal,
                               axisVal,
                               epsilonVal,
                               stashTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.skip_rms_norm lowering with dynamic shape support
struct SkipRmsNormOpLowering : public ConvertOpToLLVMPattern<SkipRmsNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SkipRmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value skipPtr = extractMemRefPtr(adaptor.getSkip(), rewriter, loc);
    Value gammaPtr = extractMemRefPtr(adaptor.getGamma(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value skipOutputPtr =
        extractMemRefPtr(adaptor.getSkipOutput(), rewriter, loc);

    // Compute num_elements for input and gamma
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    auto gammaType = cast<MemRefType>(op.getGamma().getType());
    Value gammaNumElements =
        computeNumElements(gammaType, adaptor.getGamma(), rewriter, loc);

    // Compute element_size_bytes based on element type
    Type elementType = inputType.getElementType();
    unsigned elementSizeBytes = elementType.getIntOrFloatBitWidth() / 8;
    Value elementSizeBytesVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // Extract epsilon attribute
    Value epsilonVal =
        LLVM::ConstantOp::create(rewriter, loc, f32Type, op.getEpsilonAttr());

    // Runtime function signature (10 params)
    SmallVector<Type> paramTypes = {
        ptrType, ptrType, ptrType, ptrType,
        ptrType, ptrType, // state, input, skip, gamma, output, skip_output
        i64Type, i64Type, i64Type, // input_num, gamma_num, element_size_bytes
        f32Type                    // epsilon
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenAddT5LayerNormForward, paramTypes,
        rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        statePtr,         inputPtr,         skipPtr,
        gammaPtr,         outputPtr,        skipOutputPtr,
        inputNumElements, gammaNumElements, elementSizeBytesVal,
        epsilonVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.rope(handle, q, k, cos_cache, sin_cache, start_pos)
struct RopeOpLowering : public ConvertOpToLLVMPattern<RopeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(RopeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value posIdsPtr = extractMemRefPtr(adaptor.getPositionIds(), rewriter, loc);
    Value cosCachePtr = extractMemRefPtr(adaptor.getCosCache(), rewriter, loc);
    Value sinCachePtr = extractMemRefPtr(adaptor.getSinCache(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Extract attributes as constants
    Value interleaved = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getInterleaved()));
    Value numHeads = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getNumHeads()));
    Value rotaryDim = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(op.getRotaryEmbeddingDim()));

    // Compute num_elements (supports dynamic shapes)
    auto inputType = cast<MemRefType>(op.getInput().getType());
    Value inputNumElements =
        computeNumElements(inputType, adaptor.getInput(), rewriter, loc);

    auto cosCacheType = cast<MemRefType>(op.getCosCache().getType());
    Value cosCacheNumElements =
        computeNumElements(cosCacheType, adaptor.getCosCache(), rewriter, loc);

    // Compute element_size_bytes
    unsigned elementSizeBits =
        inputType.getElementType().getIntOrFloatBitWidth();
    Value elemSizeBytes = LLVM::ConstantOp::create(
        rewriter, loc, i64Type,
        rewriter.getI64IntegerAttr(elementSizeBits / 8));

    // Function signature: wrap_rotary_embedding(
    //     RuntimeState* state, void* input, void* position_ids,
    //     void* cos_cache, void* sin_cache, void* output,
    //     int64_t interleaved, int64_t num_heads, int64_t rotary_dim,
    //     int64_t input_num_elements, int64_t cos_cache_num_elements,
    //     int64_t element_size_bytes)
    SmallVector<Type, 12> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                        ptrType, ptrType, i64Type, i64Type,
                                        i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, "wrap_rotary_embedding", paramTypes, i32Type);

    if (failed(funcOp))
      return failure();

    SmallVector<Value, 12> args = {
        statePtr,    inputPtr,         posIdsPtr,           cosCachePtr,
        sinCachePtr, outputPtr,        interleaved,         numHeads,
        rotaryDim,   inputNumElements, cosCacheNumElements, elemSizeBytes};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.add / hip.mul  (element-wise binary ops)
// Lowered call: hip_miopen_{add,mul}(handle, A_ptr, B_ptr, C_ptr, numA, numB)
// numA/numB are computed as the product of all memref dimensions for each
// operand.  When numB == 1 (scalar broadcast), the runtime broadcasts B
// over all elements of A.
template <typename OpTy>
struct MiopenBinaryOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;
  const char *funcName;

  MiopenBinaryOpLowering(const LLVMTypeConverter &converter, const char *name)
      : ConvertOpToLLVMPattern<OpTy>(converter), funcName(name) {}

  Value computeNumElements(MemRefType type, Value descriptor,
                           ConversionPatternRewriter &rewriter,
                           Location loc) const {
    Type indexType = this->getIndexType();
    int rank = type.getRank();
    Value num = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                         rewriter.getIndexAttr(1));
    for (int dimIdx : llvm::seq<int>(rank))
      num = LLVM::MulOp::create(
          rewriter, loc, num,
          MemRefDescriptor(descriptor).size(rewriter, loc, dimIdx));
    return num;
  }

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type voidType = this->getVoidType();
    Type ptrType = this->getPtrType();
    Type indexType = this->getIndexType();

    SmallVector<Type> paramTypes = {ptrType, ptrType,   ptrType,
                                    ptrType, indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, funcName, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    auto aType = cast<MemRefType>(op.getA().getType());
    auto bType = cast<MemRefType>(op.getB().getType());
    Value numA = computeNumElements(aType, adaptor.getA(), rewriter, loc);
    Value numB = computeNumElements(bType, adaptor.getB(), rewriter, loc);

    SmallVector<Value> args = {adaptor.getCtx(),
                               extractMemRefPtr(adaptor.getA(), rewriter, loc),
                               extractMemRefPtr(adaptor.getB(), rewriter, loc),
                               extractMemRefPtr(adaptor.getC(), rewriter, loc),
                               numA,
                               numB};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.softmax(%handle) ins(%input) outs(%output)
//   -> hip_miopen_softmax(handle, input, output, rows, cols)
// Rank-generic: softmax over last dim. For 3D [B,S,D], rows = B*S, cols = D.
struct MiopenSoftmaxOpLowering
    : public ConvertOpToLLVMPattern<MiopenSoftmaxOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MiopenSoftmaxOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType, indexType,
                                    indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenSoftmax, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    int rank = cast<MemRefType>(op.getInput().getType()).getRank();
    MemRefDescriptor inputDesc(adaptor.getInput());

    // cols = last dim; rows = product of all other dims
    Value cols = inputDesc.size(rewriter, loc, rank - 1);
    Value rows = inputDesc.size(rewriter, loc, 0);
    for (int i = 1; i < rank - 1; i++)
      rows = LLVM::MulOp::create(rewriter, loc, rows,
                                 inputDesc.size(rewriter, loc, i));

    SmallVector<Value> args = {
        adaptor.getCtx(), extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc), rows, cols};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// ===== Custom HIP kernel ops ================================================

// hip.transpose(%handle, %dim0, %dim1) ins(%input) outs(%output)
//   -> hip_transpose(handle, input, output, rank, dim0, dim1, s0, s1, s2)
// Swaps the two specified dimensions. Pads shape to 3 dims (trailing 1s).
struct TransposeOpLowering : public ConvertOpToLLVMPattern<TransposeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(TransposeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    // (handle, input, output, rank, dim0, dim1, s0, s1, s2)
    SmallVector<Type> paramTypes = {ptrType,   ptrType,   ptrType,
                                    indexType, indexType, indexType,
                                    indexType, indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipTranspose, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    int rank = cast<MemRefType>(op.getInput().getType()).getRank();
    if (rank > 3)
      return op.emitOpError("hip.transpose lowering supports rank <= 3, got ")
             << rank;

    MemRefDescriptor inputDesc(adaptor.getInput());
    Value rankVal = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                             rewriter.getIndexAttr(rank));
    Value one = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                         rewriter.getIndexAttr(1));

    SmallVector<Value, 3> shape;
    for (int dimIdx : llvm::seq<int>(3))
      shape.push_back(dimIdx < rank ? inputDesc.size(rewriter, loc, dimIdx)
                                    : one);

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        rankVal,
        adaptor.getDim0(),
        adaptor.getDim1(),
        shape[0],
        shape[1],
        shape[2]};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.gather(handle, indices, table, output)
struct GatherOpLowering : public ConvertOpToLLVMPattern<GatherOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GatherOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    SmallVector<Type> paramTypes(4, ptrType);
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGather, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getCtx(), extractMemRefPtr(adaptor.getIndices(), rewriter, loc),
        extractMemRefPtr(adaptor.getTable(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.silu(handle, input, output)
struct SiluOpLowering : public ConvertOpToLLVMPattern<SiluOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SiluOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    SmallVector<Type> paramTypes(3, ptrType);
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipSilu, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getCtx(), extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.sigmoid(ctx, input, output)
//   -> wrap_miopenActivationForward(state, input, output, num_elements,
//                                    data_type, activation_mode=0)
// Supports both static and dynamic shapes (computes num_elements at runtime).
struct SigmoidOpLowering : public ConvertOpToLLVMPattern<SigmoidOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SigmoidOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract pointers using alignedPtr (respects memref.view offsets)
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, ptr);
      return ptr;
    };

    Value statePtr = adaptor.getCtx();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes)
    // Start with constant 1, multiply by each dimension (static or dynamic)
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        // Dynamic dimension: extract from runtime descriptor
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        // Static dimension: use compile-time constant
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // Get data type enum (f32=0, f16=1, bf16=2)
    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.sigmoid");

    Value dataTypeVal = createI64Const(dataType);
    Value activationModeVal = createI64Const(0); // HIPDNN_EP_ACTIVATION_SIGMOID

    // int wrap_miopenActivationForward(RuntimeState* state, void* input,
    //     void* output, int64_t num_elements, int64_t data_type,
    //     int64_t activation_mode)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenActivationForward, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    inputPtr,    outputPtr,
                                  numElements, dataTypeVal, activationModeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.mul(handle, lhs, rhs, output)
//   -> wrap_miopenOpTensor(state, lhs, rhs, output, num_elements, data_type,
//   tensor_op=0)
// Supports both static and dynamic shapes.
struct MulOpLowering : public ConvertOpToLLVMPattern<MulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Helper to compute num_elements for a memref (static or dynamic)
    auto computeNumElements = [&](MemRefType type, Value descriptor) -> Value {
      Value num = createI64Const(1);
      MemRefDescriptor desc(descriptor);
      for (auto dimIdx : llvm::seq<int64_t>(type.getRank())) {
        Value dimSize;
        if (type.isDynamicDim(dimIdx)) {
          dimSize = desc.size(rewriter, loc, dimIdx);
        } else {
          dimSize = createI64Const(type.getDimSize(dimIdx));
        }
        num = LLVM::MulOp::create(rewriter, loc, num, dimSize);
      }
      return num;
    };

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = getAlignedPtr(adaptor.getLhs());
    Value rhsPtr = getAlignedPtr(adaptor.getRhs());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElementsVal = computeNumElements(outputType, adaptor.getOutput());

    // Get data type enum (f32=0, f16=1, bf16=2)
    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.mul");

    Value dataTypeVal = createI64Const(dataType);
    Value tensorOpVal = createI64Const(0); // HIPDNN_EP_TENSOR_OP_MUL

    // int wrap_miopenOpTensor(RuntimeState* state, void* lhs, void* rhs, void*
    // output,
    //     int64_t num_elements, int64_t data_type, int64_t tensor_op)
    SmallVector<Type, 7> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenOpTensor, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 7> args = {statePtr,   lhsPtr,         rhsPtr,
                                  outputPtr,  numElementsVal, dataTypeVal,
                                  tensorOpVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.sub(handle, lhs, rhs, output)
//   -> wrap_miopenTensorOp(state, lhs, rhs, output, num_lhs, num_rhs,
//                          data_type, tensor_op=0)
// Supports both static and dynamic shapes (computes num_lhs, num_rhs at
// runtime).
struct SubOpLowering : public ConvertOpToLLVMPattern<SubOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(SubOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Helper to compute num_elements for a memref (static or dynamic)
    auto computeNumElements = [&](MemRefType type, Value descriptor) -> Value {
      Value num = createI64Const(1);
      MemRefDescriptor desc(descriptor);
      for (auto dimIdx : llvm::seq<int64_t>(type.getRank())) {
        Value dimSize;
        if (type.isDynamicDim(dimIdx)) {
          dimSize = desc.size(rewriter, loc, dimIdx);
        } else {
          dimSize = createI64Const(type.getDimSize(dimIdx));
        }
        num = LLVM::MulOp::create(rewriter, loc, num, dimSize);
      }
      return num;
    };

    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0) {
        ptr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, ptr);
      }
      return ptr;
    };

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = getAlignedPtr(adaptor.getLhs());
    Value rhsPtr = getAlignedPtr(adaptor.getRhs());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElementsVal = computeNumElements(outputType, adaptor.getOutput());

    unsigned elementSizeBytes =
        outputType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       ptrType, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapElementwiseSub, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,  lhsPtr,         rhsPtr,
                                  outputPtr, numElementsVal, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    rewriter.eraseOp(op);
    return success();
  }
};

// hip.cast(ctx, input, output)
//   -> wrap_miopenCast(state, input, output, num_elements,
//                      src_data_type, dst_data_type)
// Supports both static and dynamic shapes (computes num_elements at runtime).
struct CastOpLowering : public ConvertOpToLLVMPattern<CastOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract pointers using alignedPtr (respects memref.view offsets)
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, ptr);
      return ptr;
    };

    Value statePtr = adaptor.getCtx();
    Value inputPtr = getAlignedPtr(adaptor.getInput());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute num_elements (supports dynamic shapes)
    Value numElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      numElements = LLVM::MulOp::create(rewriter, loc, numElements, dimSize);
    }

    // Get source and destination data type enums
    int64_t srcDataType = getHipdnnDataType(inputType.getElementType());
    int64_t dstDataType = getHipdnnDataType(outputType.getElementType());

    if (srcDataType < 0 || dstDataType < 0)
      return rewriter.notifyMatchFailure(
          op, "unsupported element type for hip.cast");

    Value srcDataTypeVal = createI64Const(srcDataType);
    Value dstDataTypeVal = createI64Const(dstDataType);

    // int wrap_miopenCast(RuntimeState* state, void* input, void* output,
    //     int64_t num_elements, int64_t src_data_type, int64_t dst_data_type)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenCast, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 6> args = {statePtr,    inputPtr,       outputPtr,
                                  numElements, srcDataTypeVal, dstDataTypeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.reduce_sum(ctx, input, output) {axes = [...], keepdims = ...}
//   -> wrap_miopenReduceSum(state, input, output, num_elements,
//                           axes_ptr, num_axes, keepdims, data_type)
// Supports both static and dynamic shapes (computes num_elements at runtime).
struct ReduceSumOpLowering : public ConvertOpToLLVMPattern<ReduceSumOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ReduceSumOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper to create i64 constants
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract pointers using alignedPtr
    auto getAlignedPtr = [&](Value memrefDesc) -> Value {
      MemRefDescriptor desc(memrefDesc);
      Value ptr = desc.alignedPtr(rewriter, loc);
      if (cast<LLVM::LLVMPointerType>(ptr.getType()).getAddressSpace() != 0)
        ptr = LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, ptr);
      return ptr;
    };

    Value statePtr = adaptor.getCtx();
    Value dataPtr = getAlignedPtr(adaptor.getData());
    Value axesPtr = getAlignedPtr(adaptor.getAxes());
    Value outputPtr = getAlignedPtr(adaptor.getOutput());

    auto dataType = cast<MemRefType>(op.getData().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    // Compute data_num_elements (supports dynamic shapes)
    Value dataNumElements = createI64Const(1);
    MemRefDescriptor dataDesc(adaptor.getData());

    for (auto dimIdx : llvm::seq<int64_t>(dataType.getRank())) {
      Value dimSize;
      if (dataType.isDynamicDim(dimIdx)) {
        dimSize = dataDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(dataType.getDimSize(dimIdx));
      }
      dataNumElements =
          LLVM::MulOp::create(rewriter, loc, dataNumElements, dimSize);
    }

    // Compute output_num_elements (supports dynamic shapes)
    Value outputNumElements = createI64Const(1);
    MemRefDescriptor outputDesc(adaptor.getOutput());

    for (auto dimIdx : llvm::seq<int64_t>(outputType.getRank())) {
      Value dimSize;
      if (outputType.isDynamicDim(dimIdx)) {
        dimSize = outputDesc.size(rewriter, loc, dimIdx);
      } else {
        dimSize = createI64Const(outputType.getDimSize(dimIdx));
      }
      outputNumElements =
          LLVM::MulOp::create(rewriter, loc, outputNumElements, dimSize);
    }

    // Element size in bytes
    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = createI64Const(elementSizeBytes);

    Value keepdimsVal = createI64Const(op.getKeepdims());

    // int wrap_reduce_sum(RuntimeState* state, void* data, void* axes,
    //                     void* output, int64_t data_num_elements,
    //                     int64_t output_num_elements, int64_t
    //                     element_size_bytes, int64_t keepdims)
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapReduceSum, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 8> args = {
        statePtr,        dataPtr,           axesPtr,     outputPtr,
        dataNumElements, outputNumElements, elemSizeVal, keepdimsVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.gqa(handle, q, k, v, kv_cache, output, layer, start_pos, seq_len)
struct GqaOpLowering : public ConvertOpToLLVMPattern<GqaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GqaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    SmallVector<Type> paramTypes = {ptrType,   ptrType,   ptrType,
                                    ptrType,   ptrType,   ptrType,
                                    indexType, indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kHipGqa, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getCtx(),
        extractMemRefPtr(adaptor.getQ(), rewriter, loc),
        extractMemRefPtr(adaptor.getK(), rewriter, loc),
        extractMemRefPtr(adaptor.getV(), rewriter, loc),
        extractMemRefPtr(adaptor.getKvCache(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        adaptor.getLayer(),
        adaptor.getStartPos(),
        adaptor.getSeqLen()};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// --- memref.alloc -> hip_device_malloc (produced by bufferization for
// tensor.empty)
struct MemRefAllocOpLowering : public ConvertOpToLLVMPattern<memref::AllocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::AllocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MemRefType memRefType = op.getType();

    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    Type indexType = getIndexType();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    FailureOr<LLVM::LLVMFuncOp> mallocFn = LLVM::lookupOrCreateFn(
        rewriter, module, kHipMalloc, indexType, ptrType);
    if (failed(mallocFn))
      return failure();

    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, sizeBytes, true);

    Value allocatedPtr =
        LLVM::CallOp::create(rewriter, loc, *mallocFn, sizeBytes).getResult();

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, allocatedPtr, allocatedPtr, sizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// --- memref.dealloc -> hip_device_free (produced by bufferization)
struct MemRefDeallocOpLowering
    : public ConvertOpToLLVMPattern<memref::DeallocOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(memref::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = LLVM::LLVMVoidType::get(rewriter.getContext());
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kHipFree, ptrType, voidType);
    if (failed(funcOp))
      return failure();

    // Must use allocatedPtr (not alignedPtr) -- hipFree requires the original
    // allocation base.  With memref.view, alignedPtr points into the pool
    // interior while allocatedPtr is the pool base.
    Value allocatedPtr =
        MemRefDescriptor(adaptor.getMemref()).allocatedPtr(rewriter, loc);
    if (cast<LLVM::LLVMPointerType>(allocatedPtr.getType()).getAddressSpace() !=
        0)
      allocatedPtr =
          LLVM::AddrSpaceCastOp::create(rewriter, loc, ptrType, allocatedPtr);
    LLVM::CallOp::create(rewriter, loc, *funcOp, allocatedPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertHipToLLVM Pass
//===----------------------------------------------------------------------===//

struct ConvertHipToLLVMPass
    : public impl::ConvertHipToLLVMPassBase<ConvertHipToLLVMPass> {
  void runOnOperation() override;
};

void ConvertHipToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  LowerToLLVMOptions options(ctx);
  LLVMTypeConverter typeConverter(ctx, options);

  // !hip.context -> !llvm.ptr (opaque pointer to runtime context)
  typeConverter.addConversion([ctx](ContextType type) -> Type {
    return LLVM::LLVMPointerType::get(ctx, 0);
  });

  RewritePatternSet patterns(ctx);

  // HIP dialect-specific lowerings
  patterns
      .add<AllocOpLowering, FreeOpLowering, MiopenGraphOpLowering,
           GetPoolOpLowering, HipblasltGraphOpLowering, ConvOpLowering,
           HipblasltMatmulOpLowering, RmsNormOpLowering, SkipRmsNormOpLowering,
           RopeOpLowering, MiopenSoftmaxOpLowering, TransposeOpLowering,
           GatherOpLowering, SiluOpLowering, SigmoidOpLowering, MulOpLowering,
           SubOpLowering, CastOpLowering, ReduceSumOpLowering, GqaOpLowering>(
          typeConverter);
  patterns.insert<MiopenBinaryOpLowering<MiopenAddOp>>(typeConverter,
                                                       kMiopenAdd);
  patterns.add<MemRefAllocOpLowering, MemRefDeallocOpLowering>(typeConverter);

  // Standard dialect lowerings
  // Bundle func/memref/arith/cf lowering with HIP lowering to minimize
  // unrealized casts at the memref/LLVM boundary. Running them as separate
  // stages would require a reconcile-unrealized-casts cleanup pass.
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  arith::populateArithToLLVMConversionPatterns(typeConverter, patterns);
  cf::populateControlFlowToLLVMConversionPatterns(typeConverter, patterns);

  LLVMConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<HipDialect>();
  target.addIllegalOp<memref::AllocOp, memref::DeallocOp>();
  target.addLegalOp<ModuleOp>();

  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    signalPassFailure();
}

} // namespace

} // namespace hip
} // namespace mlir
