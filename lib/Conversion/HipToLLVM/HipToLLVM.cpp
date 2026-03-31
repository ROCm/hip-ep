/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "hip/debug_log.h"
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
static constexpr const char *kWrapHipblasltMatmul = "wrap_hipblasLtMatmul";
static constexpr const char *kWrapMiopenT5LayerNormForward =
    "wrap_miopenT5LayerNormForward";
static constexpr const char *kWrapSkipSimplifiedLayerNorm =
    "wrap_skip_simplified_layer_norm";
static constexpr const char *kMiopenAdd = "hip_miopen_add";
static constexpr const char *kMiopenMul = "hip_miopen_mul";
static constexpr const char *kMiopenSoftmax = "hip_miopen_softmax";
static constexpr const char *kHipTranspose = "hip_transpose";
static constexpr const char *kWrapGather = "wrap_gather";
static constexpr const char *kHipSilu = "hip_silu";
static constexpr const char *kWrapMiopenActivationForward =
    "wrap_miopenActivationForward"; // hip.sigmoid
static constexpr const char *kWrapElementwiseSub = "wrap_elementwise_sub";
static constexpr const char *kWrapRotaryEmbedding = "wrap_rotary_embedding";
static constexpr const char *kWrapMiopenOpTensor =
    "wrap_miopenOpTensor"; // hip.mul, hip.add (with 4D shape for broadcasting)
static constexpr const char *kWrapCast = "wrap_cast";
static constexpr const char *kWrapReduceSum = "wrap_reduce_sum";
static constexpr const char *kWrapGQA = "wrap_group_query_attention";
static constexpr const char *kWrapMatMulNBits = "wrap_matmul_nbits";
static constexpr const char *kWrapQMoE = "wrap_qmoe";
static constexpr const char *kHipGetConstant = "hipdnn_ep_constant_get";

// LLVM memref descriptor struct field indices.
// Layout: { allocatedPtr, alignedPtr, offset, sizes[rank], strides[rank] }
static constexpr int64_t kAllocPtrIdx = 0;
static constexpr int64_t kAlignedPtrIdx = 1;
static constexpr int64_t kOffsetIdx = 2;
static constexpr int64_t kSizesIdx = 3;
static constexpr int64_t kStridesIdx = 4;

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
  if (elemType.isInteger(32))
    return 3; // HIPDNN_EP_DATATYPE_INT32
  if (elemType.isInteger(64))
    return 4; // HIPDNN_EP_DATATYPE_INT64
  if (elemType.isInteger(8))
    return 5; // HIPDNN_EP_DATATYPE_INT8
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

// Returns the aligned pointer for an optional memref operand, or a null
// pointer if the operand is absent.
static Value extractOptionalMemRefPtr(Value memrefDesc,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc) {
  Value result;
  if (memrefDesc) {
    result = extractMemRefPtr(memrefDesc, rewriter, loc);
  } else {
    result = LLVM::ZeroOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 0));
  }
  return result;
}

// Helper: get a single memref dimension as an i64 Value, using a compile-time
// constant for static dims and extracting from the descriptor for dynamic dims.
static Value getMemRefDimSize(MemRefType type, unsigned dimIdx,
                              Value descriptor,
                              ConversionPatternRewriter &rewriter,
                              Location loc) {
  Value result;
  if (type.isDynamicDim(dimIdx)) {
    result = MemRefDescriptor(descriptor).size(rewriter, loc, dimIdx);
  } else {
    result = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(),
        rewriter.getI64IntegerAttr(type.getDimSize(dimIdx)));
  }
  return result;
}

// Helper: compute total number of elements in a memref, handling both static
// and dynamic dimensions.
static Value computeNumElements(MemRefType type, Value descriptor,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {
  Type i64Type = rewriter.getI64Type();
  Value num = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                       rewriter.getI64IntegerAttr(1));
  for (auto dimIdx : llvm::seq<int64_t>(type.getRank())) {
    num = LLVM::MulOp::create(
        rewriter, loc, num,
        getMemRefDimSize(type, dimIdx, descriptor, rewriter, loc));
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
    Value statePtr = adaptor.getCtx(); // RuntimeState* (opaque)
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value weightsPtr = extractMemRefPtr(adaptor.getWeights(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Handle optional bias
    Value biasPtr;
    if (adaptor.getBias()) {
      biasPtr = extractMemRefPtr(adaptor.getBias(), rewriter, loc);
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
struct MatmulOpLowering : public ConvertOpToLLVMPattern<MatmulOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatmulOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    // Helper: create i64 constant
    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value APtr = extractMemRefPtr(adaptor.getA(), rewriter, loc);
    Value BPtr = extractMemRefPtr(adaptor.getB(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Get memref types and shapes
    auto AType = cast<MemRefType>(op.getA().getType());
    auto BType = cast<MemRefType>(op.getB().getType());
    int64_t ARank = AType.getRank();
    int64_t BRank = BType.getRank();

    // === DYNAMIC SHAPE SUPPORT ===
    // For dynamic shapes, we compute dimensions at runtime
    MemRefDescriptor ADesc(adaptor.getA());
    MemRefDescriptor BDesc(adaptor.getB());

    // Compute M, K, N from runtime dimensions
    // A: [..., M, K], B: [..., K, N] or B: [K, N]
    Value M =
        (ARank >= 2) ? ADesc.size(rewriter, loc, ARank - 2) : createI64Const(1);
    Value K = ADesc.size(rewriter, loc, ARank - 1);
    Value N = BDesc.size(rewriter, loc, BRank - 1);

    // Compute batch count from leading dimensions of A
    Value batchCount;
    if (ARank == 2) {
      batchCount = createI64Const(1);
    } else {
      batchCount = ADesc.size(rewriter, loc, 0);
      for (int64_t i = 1; i < ARank - 2; ++i) {
        Value dim = ADesc.size(rewriter, loc, i);
        batchCount = LLVM::MulOp::create(rewriter, loc, batchCount, dim);
      }
    }

    // Compute element size in bytes
    unsigned elemBits = AType.getElementType().getIntOrFloatBitWidth();
    Value elemSize = createI64Const(elemBits / 8);

    // Runtime signature:
    // int wrap_hipblasLtMatmul(RuntimeState* state,
    //                          const void* A, const void* B, void* output,
    //                          int64_t M, int64_t N, int64_t K,
    //                          int64_t batch_count, int64_t elem_size)
    SmallVector<Type, 9> paramTypes = {
        ptrType, // state
        ptrType, // A
        ptrType, // B
        ptrType, // output
        i64Type, // M
        i64Type, // N
        i64Type, // K
        i64Type, // batch_count
        i64Type  // elem_size
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipblasltMatmul, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 9> args = {statePtr, APtr, BPtr,       outputPtr, M,
                                  N,        K,    batchCount, elemSize};

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

    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref)
        return nullPtr;
      return extractMemRefPtr(memref, rewriter, loc);
    };

    // Extract pointers
    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value skipPtr = extractMemRefPtr(adaptor.getSkip(), rewriter, loc);
    Value gammaPtr = extractMemRefPtr(adaptor.getGamma(), rewriter, loc);
    Value biasPtr = getMemRefPtrOrNull(adaptor.getBias());
    // DPS outputs: outputs[0]=output, outputs[1]=input_skip_bias_sum (optional)
    auto outputs = adaptor.getOutputs();
    Value outputPtr = extractMemRefPtr(outputs[0], rewriter, loc);
    Value skipOutputPtr = outputs.size() > 1
                              ? extractMemRefPtr(outputs[1], rewriter, loc)
                              : nullPtr;

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

    // Runtime function signature (11 params)
    SmallVector<Type> paramTypes = {
        ptrType, // state
        ptrType, // input
        ptrType, // skip
        ptrType, // gamma
        ptrType, // bias (may be nullptr)
        ptrType, // output
        ptrType, // input_skip_bias_sum (may be nullptr)
        i64Type, // input_num_elements
        i64Type, // gamma_num_elements
        i64Type, // element_size_bytes
        f32Type  // epsilon
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapSkipSimplifiedLayerNorm,
                               paramTypes, rewriter.getI32Type());
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,         inputPtr,
                               skipPtr,          gammaPtr,
                               biasPtr,          outputPtr,
                               skipOutputPtr,    inputNumElements,
                               gammaNumElements, elementSizeBytesVal,
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
        rewriter, module, kWrapRotaryEmbedding, paramTypes, i32Type);

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
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    Value statePtr = adaptor.getCtx();
    Value dataPtr = extractMemRefPtr(adaptor.getData(), rewriter, loc);
    Value indicesPtr = extractMemRefPtr(adaptor.getIndices(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    // Compute data_num_elements
    auto dataType = cast<MemRefType>(op.getData().getType());
    Value dataNumElementsVal =
        computeNumElements(dataType, adaptor.getData(), rewriter, loc);

    // Compute output_num_elements
    auto outputType = cast<MemRefType>(op.getOutput().getType());
    Value outputNumElementsVal =
        computeNumElements(outputType, adaptor.getOutput(), rewriter, loc);

    // element_size_bytes
    unsigned elementSizeBytes =
        dataType.getElementType().getIntOrFloatBitWidth() / 8;
    Value elemSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elementSizeBytes));

    // axis attribute
    Value axisVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getAxis()));

    // int wrap_gather(RuntimeState* state, void* data, void* indices,
    //                 void* output, int64_t axis, int64_t data_num_elements,
    //                 int64_t output_num_elements, int64_t element_size_bytes)
    SmallVector<Type, 8> paramTypes = {ptrType, ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapGather, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {statePtr,
                               dataPtr,
                               indicesPtr,
                               outputPtr,
                               axisVal,
                               dataNumElementsVal,
                               outputNumElementsVal,
                               elemSizeVal};

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

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

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

// Extract the 4D shape (N, C, H, W) of a memref as LLVM i64 values.
// miopenSet4dTensorDescriptor requires exactly 4 dimensions, so ranks 1-3
// are left-padded with 1:
//   rank 1: [W]       → [1, 1, 1, W]
//   rank 2: [H, W]    → [1, 1, H, W]
//   rank 3: [C, H, W] → [1, C, H, W]
//   rank 4: [N, C, H, W] as-is
// This preserves ONNX broadcasting semantics: a dim of 1 tells MIOpen
// to broadcast that dimension against the corresponding dim of the other
// operand.
static SmallVector<Value, 4> extractShape4D(MemRefType type, Value descriptor,
                                            ConversionPatternRewriter &rewriter,
                                            Location loc, Type i64Type) {
  auto createConst = [&](int64_t v) {
    return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                    rewriter.getI64IntegerAttr(v));
  };
  MemRefDescriptor desc(descriptor);
  int rank = type.getRank();
  SmallVector<Value, 4> dims;
  for (int i : llvm::seq(4 - rank))
    dims.push_back(createConst(1));
  for (int i : llvm::seq(rank)) {
    if (type.isDynamicDim(i))
      dims.push_back(desc.size(rewriter, loc, i));
    else
      dims.push_back(createConst(type.getDimSize(i)));
  }
  return dims;
}

// Must match HIPDNN_EP_TENSOR_OP_* in lib/Runtime/hipdnn_ep_runtime.h
enum HipdnnTensorOp : int64_t {
  kTensorOpMul = 0,
  kTensorOpAdd = 1,
  kTensorOpMin = 2,
  kTensorOpMax = 3,
};

// Unified lowering for elementwise binary ops (hip.mul, hip.add, ...)
//   → wrap_miopenOpTensor(state, lhs_ptr, rhs_ptr, out_ptr,
//       lhs_n, lhs_c, lhs_h, lhs_w,
//       rhs_n, rhs_c, rhs_h, rhs_w,
//       out_n, out_c, out_h, out_w,
//       data_type, tensor_op)
//
// Full 4D shapes are passed to enable MIOpen-native broadcasting.
// E.g. Add(memref<1x128x32xf16>, memref<32xf16>) passes:
//   lhs=[1,1,128,32], rhs=[1,1,1,32], out=[1,1,128,32]
// MIOpen broadcasts rhs dims that are 1 against lhs automatically.
//
// NOTE: The 4D shape passing is a workaround for MIOpen's
// miopenSet4dTensorDescriptor API. Will be replaced when hipdnn
// elementwise support is available.
template <typename OpTy, HipdnnTensorOp tensorOpEnum>
struct ElementwiseOpLowering : public ConvertOpToLLVMPattern<OpTy> {
  using ConvertOpToLLVMPattern<OpTy>::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(OpTy op, typename OpTy::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->template getParentOfType<ModuleOp>();
    Type ptrType = this->getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();

    auto lhsType = cast<MemRefType>(op.getLhs().getType());
    auto rhsType = cast<MemRefType>(op.getRhs().getType());
    auto outputType = cast<MemRefType>(op.getOutput().getType());

    if (lhsType.getRank() > 4 || rhsType.getRank() > 4 ||
        outputType.getRank() > 4)
      return rewriter.notifyMatchFailure(
          op, "rank > 4 unsupported by MIOpen 4D descriptor API");

    auto lhsDims =
        extractShape4D(lhsType, adaptor.getLhs(), rewriter, loc, i64Type);
    auto rhsDims =
        extractShape4D(rhsType, adaptor.getRhs(), rewriter, loc, i64Type);
    auto outDims =
        extractShape4D(outputType, adaptor.getOutput(), rewriter, loc, i64Type);

    int64_t dataType = getHipdnnDataType(outputType.getElementType());
    if (dataType < 0)
      return rewriter.notifyMatchFailure(op, "unsupported element type");

    auto createI64Const = [&](int64_t v) {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    // 18 params: state + 3 data ptrs + 12 shape dims + data_type + tensor_op
    SmallVector<Type, 18> paramTypes(4, ptrType);
    paramTypes.append(14, i64Type);

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMiopenOpTensor, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 18> args = {
        adaptor.getCtx(), extractMemRefPtr(adaptor.getLhs(), rewriter, loc),
        extractMemRefPtr(adaptor.getRhs(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc)};
    args.append(lhsDims.begin(), lhsDims.end());
    args.append(rhsDims.begin(), rhsDims.end());
    args.append(outDims.begin(), outDims.end());
    args.push_back(createI64Const(dataType));
    args.push_back(createI64Const(tensorOpEnum));

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

    Value statePtr = adaptor.getCtx();
    Value lhsPtr = extractMemRefPtr(adaptor.getLhs(), rewriter, loc);
    Value rhsPtr = extractMemRefPtr(adaptor.getRhs(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

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
//   -> wrap_cast(state, input, output, num_elements,
//                src_data_type, dst_data_type)
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

    Value statePtr = adaptor.getCtx();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

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

    // int wrap_cast(RuntimeState* state, void* input, void* output,
    //     int64_t num_elements, int64_t src_data_type, int64_t dst_data_type)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       i64Type, i64Type, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapCast, paramTypes, i32Type);
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

    Value statePtr = adaptor.getCtx();
    Value dataPtr = extractMemRefPtr(adaptor.getData(), rewriter, loc);
    Value axesPtr = extractMemRefPtr(adaptor.getAxes(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

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

// hip.gqa(ctx, query, key, value, past_key, past_value, seqlens_k,
// total_seq_len,
//         output, present_key, present_value) {attributes...}
struct GqaOpLowering : public ConvertOpToLLVMPattern<GqaOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GqaOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    // Helper: create nullptr
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrType);

    // Helper: extract memref pointer or nullptr for optional operands
    auto getMemRefPtrOrNull = [&](Value memref) -> Value {
      if (!memref)
        return nullPtr;
      return extractMemRefPtr(memref, rewriter, loc);
    };

    // === Extract all inputs (required + optional) ===

    Value statePtr = adaptor.getCtx();

    // Required inputs
    Value queryPtr = extractMemRefPtr(adaptor.getQuery(), rewriter, loc);
    Value seqlensKPtr = extractMemRefPtr(adaptor.getSeqlensK(), rewriter, loc);
    Value totalSeqLenPtr =
        extractMemRefPtr(adaptor.getTotalSeqLen(), rewriter, loc);

    // Optional inputs (may be nullptr)
    Value keyPtr = getMemRefPtrOrNull(adaptor.getKey());
    Value valuePtr = getMemRefPtrOrNull(adaptor.getValue());
    Value pastKeyPtr = getMemRefPtrOrNull(adaptor.getPastKey());
    Value pastValuePtr = getMemRefPtrOrNull(adaptor.getPastValue());
    Value cosCachePtr = getMemRefPtrOrNull(adaptor.getCosCache());
    Value sinCachePtr = getMemRefPtrOrNull(adaptor.getSinCache());
    Value positionIdsPtr = getMemRefPtrOrNull(adaptor.getPositionIds());
    Value attentionBiasPtr = getMemRefPtrOrNull(adaptor.getAttentionBias());
    Value headSinkPtr = getMemRefPtrOrNull(adaptor.getHeadSink());
    Value kScalePtr = getMemRefPtrOrNull(adaptor.getKScale());
    Value vScalePtr = getMemRefPtrOrNull(adaptor.getVScale());

    // Output pointers
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);
    Value presentKeyPtr =
        extractMemRefPtr(adaptor.getPresentKey(), rewriter, loc);
    Value presentValuePtr =
        extractMemRefPtr(adaptor.getPresentValue(), rewriter, loc);
    Value outputQkPtr = getMemRefPtrOrNull(adaptor.getOutputQk());

    // === Extract attributes ===

    Value numHeads = createI64Const(op.getNumHeads());
    Value kvNumHeads = createI64Const(op.getKvNumHeads());
    Value scale = createF32Const(op.getScale().convertToFloat());
    Value doRotary = createI64Const(op.getDoRotary());
    Value rotaryInterleaved = createI64Const(op.getRotaryInterleaved());
    Value softcap = createF32Const(op.getSoftcap().convertToFloat());
    Value localWindowSize = createI64Const(op.getLocalWindowSize());
    Value smoothSoftmax = createI64Const(op.getSmoothSoftmax());
    Value qkOutput = createI64Const(op.getQkOutput());
    Value kvCacheBitWidth = createI64Const(op.getKvCacheBitWidth());

    // String attributes need to be converted to enum integers
    // "NONE"=0, "PER_TENSOR"=1, "PER_CHANNEL"=2
    auto quantTypeToEnum = [](llvm::StringRef str) -> int64_t {
      if (str == "NONE")
        return 0;
      if (str == "PER_TENSOR")
        return 1;
      if (str == "PER_CHANNEL")
        return 2;
      return 0; // default to NONE
    };
    Value kQuantType = createI64Const(quantTypeToEnum(op.getKQuantType()));
    Value vQuantType = createI64Const(quantTypeToEnum(op.getVQuantType()));

    // Extract shape info from query memref: [batch, seq_q, num_heads *
    // head_dim]
    // NOTE: Currently only supports static shapes. Dynamic shape support would
    // require extracting dimensions at runtime using MemRefDescriptor::size()
    // and computing headDim dynamically.
    auto queryType = cast<MemRefType>(op.getQuery().getType());
    auto queryShape = queryType.getShape();
    int64_t batchSize = queryShape[0];
    int64_t seqLenQ = queryShape[1];
    int64_t queryHidden = queryShape[2];
    // Packed QKV: query shape is [B, S, (H + 2*G)*d] instead of [B, S, H*d].
    // Derive head_dim accordingly: d = hidden / (H + 2*G) vs hidden / H.
    bool packedQKV = !op.getKey();
    int64_t headDim =
        packedQKV ? queryHidden / (op.getNumHeads() + 2 * op.getKvNumHeads())
                  : queryHidden / op.getNumHeads();
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

    // Function signature matches Task 4 runtime wrapper
    SmallVector<Type, 37> paramTypes = {
        ptrType, // state
        // Inputs (14 pointers - some may be nullptr)
        ptrType, // query
        ptrType, // key
        ptrType, // value
        ptrType, // past_key
        ptrType, // past_value
        ptrType, // seqlens_k
        ptrType, // total_seq_len
        ptrType, // cos_cache
        ptrType, // sin_cache
        ptrType, // position_ids
        ptrType, // attention_bias
        ptrType, // head_sink
        ptrType, // k_scale
        ptrType, // v_scale
        // Outputs (4 pointers - output_qk may be nullptr)
        ptrType, // output
        ptrType, // present_key
        ptrType, // present_value
        ptrType, // output_qk
        // Attributes (13 values)
        i64Type, // num_heads
        i64Type, // kv_num_heads
        f32Type, // scale
        i64Type, // do_rotary
        i64Type, // rotary_interleaved
        f32Type, // softcap
        i64Type, // local_window_size
        i64Type, // smooth_softmax
        i64Type, // qk_output
        i64Type, // k_quant_type
        i64Type, // v_quant_type
        i64Type, // kv_cache_bit_width
        // Shape info (5 values)
        i64Type, // batch_size
        i64Type, // seq_len_q
        i64Type, // seq_len_kv
        i64Type, // head_dim
        i64Type  // element_size_bytes
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kWrapGQA, paramTypes, i32Type);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 37> args = {
        statePtr,
        // Inputs (14 pointers)
        queryPtr, keyPtr, valuePtr, pastKeyPtr, pastValuePtr, seqlensKPtr,
        totalSeqLenPtr, cosCachePtr, sinCachePtr, positionIdsPtr,
        attentionBiasPtr, headSinkPtr, kScalePtr, vScalePtr,
        // Outputs (4 pointers)
        outputPtr, presentKeyPtr, presentValuePtr, outputQkPtr,
        // Attributes (13 values)
        numHeads, kvNumHeads, scale, doRotary, rotaryInterleaved, softcap,
        localWindowSize, smoothSoftmax, qkOutput, kQuantType, vQuantType,
        kvCacheBitWidth,
        // Shape info (5 values)
        batchSizeVal, seqLenQVal, seqLenKVVal, headDimVal, elemSizeVal};

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
// MatMulNBits Lowering
//===----------------------------------------------------------------------===//

struct MatMulNBitsOpLowering : public ConvertOpToLLVMPattern<MatMulNBitsOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MatMulNBitsOp op, OpAdaptor adaptor,
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

    Value statePtr = adaptor.getHandle();
    Value APtr = extractMemRefPtr(adaptor.getA(), rewriter, loc);
    Value BPtr = extractMemRefPtr(adaptor.getB(), rewriter, loc);
    Value scalesPtr = extractMemRefPtr(adaptor.getScales(), rewriter, loc);
    Value zeroPointsPtr =
        extractOptionalMemRefPtr(adaptor.getZeroPoints(), rewriter, loc);
    Value gIdxPtr = extractOptionalMemRefPtr(adaptor.getGIdx(), rewriter, loc);
    Value biasPtr = extractOptionalMemRefPtr(adaptor.getBias(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto AType = cast<MemRefType>(op.getA().getType());
    int64_t ARank = AType.getRank();
    int64_t elemSize = AType.getElementType().getIntOrFloatBitWidth() / 8;

    // A shape: [..., M, K] — M is the second-to-last dim
    Value m = (ARank >= 2) ? getMemRefDimSize(AType, ARank - 2, adaptor.getA(),
                                              rewriter, loc)
                           : createI64Const(1);
    // batch_count = product of all leading dimensions before M
    Value batch = createI64Const(1);
    for (int64_t i = 0; i < ARank - 2; ++i) {
      batch = LLVM::MulOp::create(
          rewriter, loc, batch,
          getMemRefDimSize(AType, i, adaptor.getA(), rewriter, loc));
    }

    Value n = createI64Const(op.getN());
    Value k = createI64Const(op.getK());
    Value bits = createI64Const(op.getBits());
    Value blockSize = createI64Const(op.getBlockSize());
    Value elemSizeVal = createI64Const(elemSize);

    SmallVector<Type, 15> paramTypes = {
        ptrType, // state
        ptrType, // A
        ptrType, // B
        ptrType, // scales
        ptrType, // zero_points (nullable)
        ptrType, // g_idx (nullable)
        ptrType, // bias (nullable)
        ptrType, // output
        i64Type, // M
        i64Type, // N
        i64Type, // K
        i64Type, // batch_count
        i64Type, // bits
        i64Type, // block_size
        i64Type  // elem_size
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapMatMulNBits, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 15> args = {
        statePtr, APtr,    BPtr,      scalesPtr, zeroPointsPtr,
        gIdxPtr,  biasPtr, outputPtr, m,         n,
        k,        batch,   bits,      blockSize, elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// QMoE Lowering
//===----------------------------------------------------------------------===//

struct QMoEOpLowering : public ConvertOpToLLVMPattern<QMoEOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(QMoEOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    Type f32Type = rewriter.getF32Type();

    auto createI64Const = [&](int64_t value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(value));
    };
    auto createF32Const = [&](float value) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, f32Type,
                                      rewriter.getF32FloatAttr(value));
    };

    Value statePtr = adaptor.getHandle();
    Value inputPtr = extractMemRefPtr(adaptor.getInput(), rewriter, loc);
    Value routerPtr = extractMemRefPtr(adaptor.getRouterProbs(), rewriter, loc);
    Value fc1WeightsPtr =
        extractMemRefPtr(adaptor.getFc1ExpertsWeights(), rewriter, loc);
    Value fc1ScalesPtr =
        extractMemRefPtr(adaptor.getFc1Scales(), rewriter, loc);
    Value fc1BiasPtr =
        extractOptionalMemRefPtr(adaptor.getFc1ExpertsBias(), rewriter, loc);
    Value fc2WeightsPtr =
        extractMemRefPtr(adaptor.getFc2ExpertsWeights(), rewriter, loc);
    Value fc2ScalesPtr =
        extractMemRefPtr(adaptor.getFc2Scales(), rewriter, loc);
    Value fc2BiasPtr =
        extractOptionalMemRefPtr(adaptor.getFc2ExpertsBias(), rewriter, loc);
    Value fc3WeightsPtr =
        extractOptionalMemRefPtr(adaptor.getFc3ExpertsWeights(), rewriter, loc);
    Value fc3ScalesPtr =
        extractOptionalMemRefPtr(adaptor.getFc3Scales(), rewriter, loc);
    Value fc3BiasPtr =
        extractOptionalMemRefPtr(adaptor.getFc3ExpertsBias(), rewriter, loc);
    Value fc1ZpPtr =
        extractOptionalMemRefPtr(adaptor.getFc1ZeroPoints(), rewriter, loc);
    Value fc2ZpPtr =
        extractOptionalMemRefPtr(adaptor.getFc2ZeroPoints(), rewriter, loc);
    Value fc3ZpPtr =
        extractOptionalMemRefPtr(adaptor.getFc3ZeroPoints(), rewriter, loc);
    Value outputPtr = extractMemRefPtr(adaptor.getOutput(), rewriter, loc);

    auto inputType = cast<MemRefType>(op.getInput().getType());
    auto routerType = cast<MemRefType>(op.getRouterProbs().getType());
    auto fc1Type = cast<MemRefType>(op.getFc1ExpertsWeights().getType());
    int64_t elemSize = inputType.getElementType().getIntOrFloatBitWidth() / 8;

    // input shape: [batch, seq, ..., hidden] — numTokens = product of all
    // dims except the last (hidden), supporting dynamic batch/seq dimensions.
    int64_t inputRank = inputType.getRank();
    Value numTokensVal = createI64Const(1);
    for (int64_t i = 0; i < inputRank - 1; ++i) {
      numTokensVal = LLVM::MulOp::create(
          rewriter, loc, numTokensVal,
          getMemRefDimSize(inputType, i, adaptor.getInput(), rewriter, loc));
    }
    Value hiddenSizeVal = getMemRefDimSize(inputType, inputRank - 1,
                                           adaptor.getInput(), rewriter, loc);
    Value numExpertsVal =
        getMemRefDimSize(routerType, routerType.getRank() - 1,
                         adaptor.getRouterProbs(), rewriter, loc);

    int64_t swigluFusion = op.getSwigluFusion();
    int64_t fusionSize = (swigluFusion > 0) ? 2 : 1;
    Value interSizeVal = getMemRefDimSize(
        fc1Type, 1, adaptor.getFc1ExpertsWeights(), rewriter, loc);
    if (fusionSize > 1) {
      interSizeVal = LLVM::SDivOp::create(rewriter, loc, interSizeVal,
                                          createI64Const(fusionSize));
    }

    StringRef activationType = op.getActivationType();
    int64_t activationTypeEnum = 0;
    if (activationType == "relu") {
      activationTypeEnum = 0;
    } else if (activationType == "gelu") {
      activationTypeEnum = 1;
    } else if (activationType == "silu") {
      activationTypeEnum = 2;
    } else if (activationType == "swiglu") {
      activationTypeEnum = 3;
    } else if (activationType == "identity") {
      activationTypeEnum = 4;
    }

    Value kVal = createI64Const(op.getK());
    Value expertWeightBitsVal = createI64Const(op.getExpertWeightBits());
    Value blockSizeVal = createI64Const(op.getBlockSize());
    Value swigluFusionVal = createI64Const(swigluFusion);
    Value activationTypeVal = createI64Const(activationTypeEnum);
    Value activationAlphaVal =
        createF32Const(op.getActivationAlpha().convertToFloat());
    Value activationBetaVal =
        createF32Const(op.getActivationBeta().convertToFloat());
    Value swigluLimitVal = createF32Const(op.getSwigluLimit().convertToFloat());
    Value normalizeVal = createI64Const(op.getNormalizeRoutingWeights());
    Value elemSizeVal = createI64Const(elemSize);

    SmallVector<Type, 30> paramTypes = {
        ptrType, // state
        ptrType, // input
        ptrType, // router_probs
        ptrType, // fc1_weights
        ptrType, // fc1_scales
        ptrType, // fc1_bias (nullable)
        ptrType, // fc2_weights
        ptrType, // fc2_scales
        ptrType, // fc2_bias (nullable)
        ptrType, // fc3_weights (nullable)
        ptrType, // fc3_scales (nullable)
        ptrType, // fc3_bias (nullable)
        ptrType, // fc1_zero_points (nullable)
        ptrType, // fc2_zero_points (nullable)
        ptrType, // fc3_zero_points (nullable)
        ptrType, // output
        i64Type, // num_tokens
        i64Type, // hidden_size
        i64Type, // inter_size
        i64Type, // num_experts
        i64Type, // k
        i64Type, // expert_weight_bits
        i64Type, // block_size
        i64Type, // swiglu_fusion
        i64Type, // activation_type
        f32Type, // activation_alpha
        f32Type, // activation_beta
        f32Type, // swiglu_limit
        i64Type, // normalize_routing_weights
        i64Type  // elem_size
    };

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapQMoE, paramTypes, i32Type);
    if (failed(funcOp)) {
      return failure();
    }

    SmallVector<Value, 30> args = {statePtr,
                                   inputPtr,
                                   routerPtr,
                                   fc1WeightsPtr,
                                   fc1ScalesPtr,
                                   fc1BiasPtr,
                                   fc2WeightsPtr,
                                   fc2ScalesPtr,
                                   fc2BiasPtr,
                                   fc3WeightsPtr,
                                   fc3ScalesPtr,
                                   fc3BiasPtr,
                                   fc1ZpPtr,
                                   fc2ZpPtr,
                                   fc3ZpPtr,
                                   outputPtr,
                                   numTokensVal,
                                   hiddenSizeVal,
                                   interSizeVal,
                                   numExpertsVal,
                                   kVal,
                                   expertWeightBitsVal,
                                   blockSizeVal,
                                   swigluFusionVal,
                                   activationTypeVal,
                                   activationAlphaVal,
                                   activationBetaVal,
                                   swigluLimitVal,
                                   normalizeVal,
                                   elemSizeVal};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

//===----------------------------------------------------------------------===//
// GetConstantOp Lowering
//===----------------------------------------------------------------------===//

struct GetConstantOpLowering : public ConvertOpToLLVMPattern<GetConstantOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetConstantOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i64Type = IntegerType::get(rewriter.getContext(), 64);
    MemRefType memRefType = op.getResult().getType();

    SmallVector<Type, 2> paramTypes = {ptrType, i64Type};

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGetConstant, paramTypes, ptrType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value, 2> args = {adaptor.getCtx(), adaptor.getIndex()};
    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp, args);

    // The runtime always returns a generic pointer (AS 0).  Cast to the
    // memref's address space (e.g. AS 1 = AMDGPU global memory) if needed.
    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();

    Value dataPtr = callOp.getResult();
    if (*addrSpace != 0)
      dataPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          dataPtr);

    auto shape = memRefType.getShape();
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;

    for (int64_t dim : shape) {
      Value size = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                            rewriter.getI64IntegerAttr(dim));
      sizes.push_back(size);
    }

    int64_t stride = 1;
    for (int i = shape.size() - 1; i >= 0; --i) {
      Value strideVal = LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(stride));
      strides.insert(strides.begin(), strideVal);
      stride *= shape[i];
    }

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, dataPtr, dataPtr, sizes, strides, rewriter);

    rewriter.replaceOp(op, {desc});
    return success();
  }
};

//===----------------------------------------------------------------------===//
// ConvertHipToLLVM Pass
//===----------------------------------------------------------------------===//

struct ConvertHipToLLVMPass
    : public impl::ConvertHipToLLVMPassBase<ConvertHipToLLVMPass> {
  void runOnOperation() override;

private:
  Type getMemRefStructType(OpBuilder &builder, int64_t rank,
                           unsigned addrSpace) {
    MLIRContext *ctx = builder.getContext();
    Type ptrType = LLVM::LLVMPointerType::get(ctx, addrSpace);
    Type i64Type = builder.getI64Type();
    Type sizeArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    Type strideArrayType = LLVM::LLVMArrayType::get(i64Type, rank);
    return LLVM::LLVMStructType::getLiteral(
        ctx, {ptrType, ptrType, i64Type, sizeArrayType, strideArrayType});
  }

  void unpackMemRefStructWithAddrCast(OpBuilder &builder, Location loc,
                                      Value memrefStruct, int64_t rank,
                                      SmallVectorImpl<Value> &args) {
    Type as0PtrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);

    auto castToAs0 = [&](Value ptr) -> Value {
      auto ptrTy = cast<LLVM::LLVMPointerType>(ptr.getType());
      if (ptrTy.getAddressSpace() == 0)
        return ptr;
      return builder.create<LLVM::AddrSpaceCastOp>(loc, as0PtrType, ptr);
    };

    Value allocPtr = builder.create<LLVM::ExtractValueOp>(
        loc, memrefStruct, ArrayRef<int64_t>{kAllocPtrIdx});
    args.push_back(castToAs0(allocPtr));

    Value alignedPtr = builder.create<LLVM::ExtractValueOp>(
        loc, memrefStruct, ArrayRef<int64_t>{kAlignedPtrIdx});
    args.push_back(castToAs0(alignedPtr));

    args.push_back(builder.create<LLVM::ExtractValueOp>(
        loc, memrefStruct, ArrayRef<int64_t>{kOffsetIdx}));

    for (int64_t dim : llvm::seq<int64_t>(0, rank))
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{kSizesIdx, dim}));
    for (int64_t dim : llvm::seq<int64_t>(0, rank))
      args.push_back(builder.create<LLVM::ExtractValueOp>(
          loc, memrefStruct, ArrayRef<int64_t>{kStridesIdx, dim}));
  }

  LogicalResult transformMainFunction(ModuleOp module) {
    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!mainFunc)
      return success();

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
      COMPILER_DEBUG_LOG(
          "[HipToLLVM] Warning: No metadata found, skipping @main_graph "
          "transformation\n");
      return success();
    }

    int64_t inputCount = inputCountAttr.getInt();
    int64_t outputCount = outputCountAttr.getInt();

    if ((int64_t)inputShapesAttr.size() != inputCount ||
        (int64_t)outputShapesAttr.size() != outputCount)
      return module.emitError("Metadata mismatch: shapes array size != count");

    constexpr unsigned kMemRefPtrs = 2;   // allocatedPtr + alignedPtr
    constexpr unsigned kMemRefOffset = 1; // offset scalar
    unsigned expectedParams = 1;          // context
    for (auto shapeAttr : inputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedParams += kMemRefPtrs + kMemRefOffset + rank + rank;
    }
    for (auto shapeAttr : outputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedParams += kMemRefPtrs + kMemRefOffset + rank + rank;
    }

    unsigned actualParams = mainFunc.getFunctionType().getNumParams();
    if (actualParams != expectedParams) {
      return module.emitError()
             << "[HipToLLVM] Parameter count mismatch: expected "
             << expectedParams << ", got " << actualParams;
    }

    OpBuilder builder(module.getContext());
    Location loc = mainFunc.getLoc();

    // Rename the original main_graph (with unpacked memref params) so we can
    // create a new wrapper that takes the runtime's (ctx, inputs, outputs)
    // signature and unpacks memref structs before forwarding the call.
    mainFunc.setName("main_graph_internal");
    mainFunc.setLinkage(LLVM::Linkage::Private);

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    SmallVector<Type> newParamTypes = {ptrType, ptrType, ptrType};
    auto newFuncType = LLVM::LLVMFunctionType::get(i32Type, newParamTypes);

    // Create the new main_graph wrapper with the simplified (ctx, inputs,
    // outputs) signature that the runtime expects.
    builder.setInsertionPoint(mainFunc);
    auto newMainFunc =
        builder.create<LLVM::LLVMFuncOp>(loc, "main_graph", newFuncType);
    newMainFunc.setLinkage(LLVM::Linkage::Private);

    newMainFunc->setAttr(
        "passthrough",
        builder.getArrayAttr({builder.getStringAttr("noinline")}));

    Block *entryBlock = newMainFunc.addEntryBlock(builder);
    builder.setInsertionPointToStart(entryBlock);

    Value ctxArg = entryBlock->getArgument(0);
    Value inputsArg = entryBlock->getArgument(1);
    Value outputsArg = entryBlock->getArgument(2);

    SmallVector<Value> mainInternalArgs;
    mainInternalArgs.push_back(ctxArg);

    for (int64_t i = 0; i < inputCount; i++) {
      int64_t rank = cast<DenseI64ArrayAttr>(inputShapesAttr[i]).size();
      Value inputIdxVal = builder.create<LLVM::ConstantOp>(
          loc, i32Type, builder.getI32IntegerAttr(i));
      Value inputSlotPtr = builder.create<LLVM::GEPOp>(
          loc, ptrType, ptrType, inputsArg, ValueRange{inputIdxVal});
      Value inputStructPtr =
          builder.create<LLVM::LoadOp>(loc, ptrType, inputSlotPtr);
      Type memrefStructType = getMemRefStructType(builder, rank, 1);
      Value inputMemref =
          builder.create<LLVM::LoadOp>(loc, memrefStructType, inputStructPtr);
      unpackMemRefStructWithAddrCast(builder, loc, inputMemref, rank,
                                     mainInternalArgs);
    }

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

    COMPILER_DEBUG_LOG("[HipToLLVM] Transformed @main_graph signature: "
                       << actualParams << " params -> 3 params\n");
    return success();
  }
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
      .add<AllocOpLowering, FreeOpLowering, GetConstantOpLowering,
           MiopenGraphOpLowering, GetPoolOpLowering, HipblasltGraphOpLowering,
           ConvOpLowering, MatmulOpLowering, RmsNormOpLowering,
           SkipRmsNormOpLowering, RopeOpLowering, MiopenSoftmaxOpLowering,
           TransposeOpLowering, GatherOpLowering, SiluOpLowering,
           SigmoidOpLowering, ElementwiseOpLowering<MulOp, kTensorOpMul>,
           ElementwiseOpLowering<AddOp, kTensorOpAdd>, SubOpLowering,
           CastOpLowering, ReduceSumOpLowering, GqaOpLowering,
           MatMulNBitsOpLowering, QMoEOpLowering>(typeConverter);
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

  if (failed(transformMainFunction(module)))
    signalPassFailure();
}

} // namespace

} // namespace hip
} // namespace mlir
