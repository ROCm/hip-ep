/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPTOLLVM_UTILS_H
#define HIP_CONVERSION_HIPTOLLVM_UTILS_H

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
#include "mlir/Dialect/Arith/Transforms/Passes.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/Sequence.h"

namespace mlir {
namespace hip {

inline constexpr const char *kHipMalloc = "hip_device_malloc";
inline constexpr const char *kHipFree = "hip_device_free";
inline constexpr const char *kHipGetPoolBase = "hipdnn_ep_get_pool_base";
inline constexpr const char *kHipGetHostScratch =
    "hipdnn_ep_get_host_scratch_base";
inline constexpr const char *kHipAllocOutput = "hipdnn_ep_alloc_output";

inline constexpr const char *kWrapHipMemcpyAsync = "wrap_hipMemcpyAsync";
inline constexpr const char *kWrapHipMemcpy2DAsync = "wrap_hipMemcpy2DAsync";
inline constexpr const char *kWrapStridedCopy = "wrap_strided_copy";

inline constexpr const char *kIm2d2Col = "wrap_im2d2col";
inline constexpr const char *kMiopenConvolutionForward =
    "wrap_miopenConvolutionForward";
inline constexpr const char *kMiopenConvolutionTranspose =
    "wrap_miopenConvolutionTranspose";
inline constexpr const char *kWrapHipblasltMatmul = "wrap_hipblasLtMatmul";
inline constexpr const char *kWrapMiopenT5LayerNormForward =
    "wrap_miopenT5LayerNormForward";
inline constexpr const char *kWrapSkipSimplifiedLayerNorm =
    "wrap_skip_simplified_layer_norm";
inline constexpr const char *kWrapLayerNormalization =
    "wrap_layer_normalization";
inline constexpr const char *kMiopenAdd = "hip_miopen_add";
inline constexpr const char *kMiopenMul = "hip_miopen_mul";
inline constexpr const char *kMiopenSoftmax = "hip_miopen_softmax";
inline constexpr const char *kWrapTranspose = "wrap_transpose";
inline constexpr const char *kWrapGather = "wrap_gather";
inline constexpr const char *kWrapGatherElements = "wrap_gather_elements";
inline constexpr const char *kWrapTopK = "wrap_top_k";
inline constexpr const char *kWrapScatterElements = "wrap_scatter_elements";
inline constexpr const char *kWrapCompress = "wrap_compress";
inline constexpr const char *kWrapOneHot = "wrap_one_hot";
inline constexpr const char *kHipSilu = "hip_silu";
inline constexpr const char *kWrapMiopenActivationForward =
    "wrap_miopenActivationForward";                            // hip.sigmoid
inline constexpr const char *kWrapGelu = "wrap_gelu";          // hip.gelu
inline constexpr const char *kWrapBiasGelu = "wrap_bias_gelu"; // hip.bias_gelu
inline constexpr const char *kWrapFastGelu = "wrap_fast_gelu"; // hip.fast_gelu
inline constexpr const char *kWrapLeakyRelu =
    "wrap_leaky_relu"; // hip.leaky_relu
inline constexpr const char *kWrapElementwiseSub = "wrap_elementwise_sub";
inline constexpr const char *kWrapRotaryEmbedding = "wrap_rotary_embedding";
inline constexpr const char *kWrapMiopenOpTensor =
    "wrap_miopenOpTensor"; // hip.mul, hip.add (with 4D shape for broadcasting)
inline constexpr const char *kWrapCast = "wrap_cast";
inline constexpr const char *kWrapPower = "wrap_power";
inline constexpr const char *kWrapRange = "wrap_range";
inline constexpr const char *kWrapReduceSum = "wrap_reduce_sum";
inline constexpr const char *kWrapReduceMean = "wrap_reduce_mean";
inline constexpr const char *kWrapReduceL2 = "wrap_reduce_l2";
inline constexpr const char *kWrapReduceMax = "wrap_reduce_max";
inline constexpr const char *kWrapReduceMin = "wrap_reduce_min";
inline constexpr const char *kWrapGQA = "wrap_group_query_attention";
inline constexpr const char *kWrapMultiHeadAttention =
    "wrap_multi_head_attention";
inline constexpr const char *kWrapMatMulNBits = "wrap_matmul_nbits";
inline constexpr const char *kWrapQMoE = "wrap_qmoe";
inline constexpr const char *kWrapGatherBlockQuantized =
    "wrap_gather_block_quantized";
inline constexpr const char *kWrapGemm = "wrap_gemm";
inline constexpr const char *kWrapLinearAttention = "wrap_linear_attention";
inline constexpr const char *kHipGetConstant = "hipdnn_ep_constant_get";
inline constexpr const char *kHipDNNGraphExecute = "hipdnn_graph_execute";
inline constexpr const char *kWrapCausalConvWithState =
    "wrap_causal_conv_with_state";
inline constexpr const char *kWrapWhere = "wrap_where";
inline constexpr const char *kWrapEqual = "wrap_equal";
inline constexpr const char *kWrapOr = "wrap_or";
inline constexpr const char *kWrapAnd = "wrap_and";
inline constexpr const char *kWrapAbs = "wrap_abs";
inline constexpr const char *kWrapNeg = "wrap_neg";
inline constexpr const char *kWrapNot = "wrap_not";
inline constexpr const char *kWrapCos = "wrap_cos";
inline constexpr const char *kWrapSin = "wrap_sin";
inline constexpr const char *kWrapCeil = "wrap_ceil";
inline constexpr const char *kWrapExp = "wrap_exp";
inline constexpr const char *kWrapLog = "wrap_log";
inline constexpr const char *kWrapDiv = "wrap_div";
inline constexpr const char *kWrapCumSum = "wrap_cumsum";
inline constexpr const char *kWrapPad = "wrap_pad";
inline constexpr const char *kWrapTile = "wrap_tile";
inline constexpr const char *kWrapExpand = "wrap_expand";
inline constexpr const char *kWrapReduceProd = "wrap_reduce_prod";
inline constexpr const char *kWrapPool = "wrap_pool";
inline constexpr const char *kWrapResize = "wrap_resize";
inline constexpr const char *kWrapGlobalPool = "wrap_global_pool";
inline constexpr const char *kWrapLess = "wrap_less";
inline constexpr const char *kWrapGatherND = "wrap_gather_nd";
inline constexpr const char *kWrapSign = "wrap_sign";
inline constexpr const char *kWrapMod = "wrap_mod";
inline constexpr const char *kWrapSlice = "wrap_slice";
inline constexpr const char *kWrapScatterND = "wrap_scatter_nd";
inline constexpr const char *kWrapNonZero = "wrap_nonzero";
inline constexpr const char *kWrapSize = "wrap_size";
// Synchronize the stream and read a device i32 scalar back to the host
// (used by hip.readback_dim to materialise a data-dependent dynamic dim).
inline constexpr const char *kHipReadbackI32 = "hipdnn_ep_readback_i32";

// Synchronize the stream and copy a small device scalar of arbitrary byte
// width into a host buffer (used by hip.readback_scalar for non-i32 scalars
// such as the i64/f32/f16 operands of a data-dependent onnx.Range).
inline constexpr const char *kHipReadbackScalar = "hipdnn_ep_readback_scalar";

// LLVM memref descriptor struct field indices.
// Layout: { allocatedPtr, alignedPtr, offset, sizes[rank], strides[rank] }
inline constexpr int64_t kAllocPtrIdx = 0;
inline constexpr int64_t kAlignedPtrIdx = 1;
inline constexpr int64_t kOffsetIdx = 2;
inline constexpr int64_t kSizesIdx = 3;
inline constexpr int64_t kStridesIdx = 4;

// Activation mode constants.
// Values must match HIPDNN_EP_ACTIVATION_* in lib/Runtime/hipdnn_ep_runtime.h.
inline constexpr int64_t kActivationSigmoid = 0;
inline constexpr int64_t kActivationRelu = 1;
inline constexpr int64_t kActivationTanh = 2;
inline constexpr int64_t kActivationSoftplus = 3;

// Window-pool reduction mode constants (hip.pool / wrap_pool).
// Values must match HIPDNN_EP_POOL_* in lib/Runtime/hipdnn_ep_runtime.h
// and the `pool_mode` constants used in OnnxToHip/PoolConversion.cpp.
inline constexpr int64_t kPoolAverage = 0;
inline constexpr int64_t kPoolMax = 1;
inline constexpr int64_t kPoolLp = 2;
// Global-pool reduction mode constants.
// Values must match HIPDNN_EP_GLOBAL_POOL_* in lib/Runtime/hipdnn_ep_runtime.h
// and the `mode` constants used in OnnxToHip/GlobalPoolConversion.cpp.
inline constexpr int64_t kGlobalPoolAverage = 0;
inline constexpr int64_t kGlobalPoolMax = 1;
inline constexpr int64_t kGlobalPoolLp = 2;

// Maps MLIR element type to runtime data type enum (HIPDNN_EP_DATATYPE_*).
// Values must match the #defines in hipdnn_ep_runtime.h.
// Returns -1 for unsupported types.
inline int64_t getHipdnnDataType(Type elemType) {
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
  if (elemType.isUnsignedInteger(8))
    return 7; // HIPDNN_EP_DATATYPE_UINT8
  if (elemType.isSignedInteger(8) || elemType.isSignlessInteger(8))
    return 5; // HIPDNN_EP_DATATYPE_INT8
  if (elemType.isF64())
    return 6; // HIPDNN_EP_DATATYPE_DOUBLE
  if (elemType.isInteger(16))
    return 8; // HIPDNN_EP_DATATYPE_INT16
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
//
// PRECONDITION: the source memref must have an identity layout (zero offset,
// contiguous strides).  The returned pointer is alignedPtr only — offset and
// strides are dropped on the floor.  Calling this on a strided/offset memref
// (e.g., the result of memref.subview) silently produces a pointer to the
// base of the parent buffer, not the slice.
//
// The --hip-promote-strided-operands pass enforces this precondition for
// hip.* DPS-input operands by materializing contiguous temporaries upstream.
// Direct callers (outside the standard hip.* lowering path) must guarantee
// it themselves; if you need the descriptor's offset / strides, use
// extractMemRefDescriptor below.
//
// Uses alignedPtr (not allocatedPtr) so that memref.view offsets into a memory
// pool are respected -- each view has the same allocatedPtr but a distinct
// alignedPtr.
inline Value extractContiguousMemRefPtr(Value memrefDesc,
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

// First logical element: alignedPtr + offset (elements), then cast to AS 0.
// Use for HIP/MIOpen entry points when the memref may be a subview with a
// non-zero descriptor offset (same base alignedPtr as parent, distinct offset).
inline Value extractMemRefDataPtr(Value memrefDesc, MemRefType memrefType,
                                  const TypeConverter *typeConverter,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
  SmallVector<Type, 1> llvmElemTypes;
  if (failed(typeConverter->convertType(memrefType.getElementType(),
                                        llvmElemTypes)) ||
      llvmElemTypes.empty())
    return Value();
  Type llvmElemTy = llvmElemTypes.front();

  MemRefDescriptor desc(memrefDesc);
  Value aligned = desc.alignedPtr(rewriter, loc);
  Value offset = desc.offset(rewriter, loc);
  Type ptrTy = aligned.getType();
  Value dataPtr =
      LLVM::GEPOp::create(rewriter, loc, ptrTy, llvmElemTy, aligned,
                          ValueRange{offset}, LLVM::GEPNoWrapFlags::inbounds)
          .getResult();

  if (cast<LLVM::LLVMPointerType>(dataPtr.getType()).getAddressSpace() != 0)
    dataPtr = LLVM::AddrSpaceCastOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 0),
        dataPtr);
  return dataPtr;
}

// Returns the aligned pointer for an optional memref operand, or a null
// pointer if the operand is absent.
//
// Same identity-layout precondition as extractContiguousMemRefPtr.
inline Value extractOptionalMemRefPtr(Value memrefDesc,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc) {
  Value result;
  if (memrefDesc) {
    result = extractContiguousMemRefPtr(memrefDesc, rewriter, loc);
  } else {
    result = LLVM::ZeroOp::create(
        rewriter, loc, LLVM::LLVMPointerType::get(rewriter.getContext(), 0));
  }
  return result;
}

// Returns the full LLVM memref descriptor wrapper for \p memrefDesc, exposing
// allocatedPtr / alignedPtr / offset / sizes / strides via MemRefDescriptor's
// accessors.  Use this when a runtime call needs to honor the slice (offset,
// per-dim strides) instead of treating the operand as contiguous.
//
// Reserved for future per-op zero-copy lowerings (hot ops where the upstream
// promote-then-copy materialization in --hip-promote-strided-operands is
// measurably expensive and the underlying library natively accepts strides).
// No in-tree callers today.
inline MemRefDescriptor
extractMemRefDescriptor(Value memrefDesc, ConversionPatternRewriter &rewriter,
                        Location loc) {
  (void)rewriter;
  (void)loc;
  return MemRefDescriptor(memrefDesc);
}

// Read the compiler-assigned op-state slot (`hip.op_state_slot`, set by
// --assign-op-state-slots) as an i32 constant Value, passed as the second
// argument (right after RuntimeState*) of a stateful op's wrap_* runtime call.
// Returns -1 when the attribute is absent (op not stateful / pass not run),
// which the runtime treats as "no slot". See
// docs/design/op-state-slots-design.md.
inline Value getOpStateSlotValue(Operation *op,
                                 ConversionPatternRewriter &rewriter,
                                 Location loc) {
  int32_t slot = -1;
  if (auto attr = op->getAttrOfType<IntegerAttr>("hip.op_state_slot"))
    slot = static_cast<int32_t>(attr.getInt());
  return LLVM::ConstantOp::create(rewriter, loc, rewriter.getI32Type(),
                                  rewriter.getI32IntegerAttr(slot));
}

// Helper: get a single memref dimension as an i64 Value, using a compile-time
// constant for static dims and extracting from the descriptor for dynamic dims.
inline Value getMemRefDimSize(MemRefType type, unsigned dimIdx,
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
inline Value computeNumElements(MemRefType type, Value descriptor,
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

// Extract the 4D shape (N, C, H, W) of a memref as LLVM i64 values.
// miopenSetNdTensorDescriptorWithLayout requires exactly 4 dimensions, so
// ranks 1-3 are left-padded with 1:
//   rank 1: [W]       -> [1, 1, 1, W]
//   rank 2: [H, W]    -> [1, 1, H, W]
//   rank 3: [C, H, W] -> [1, C, H, W]
//   rank 4: [N, C, H, W] as-is
// This preserves ONNX broadcasting semantics: a dim of 1 tells MIOpen
// to broadcast that dimension against the corresponding dim of the other
// operand.
inline SmallVector<Value, 4> extractShape4D(MemRefType type, Value descriptor,
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

// Pattern population functions (one per operator file)
void populateMemoryLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateConvLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateConvTransposeLoweringPatterns(const LLVMTypeConverter &converter,
                                           RewritePatternSet &patterns);
void populateMatmulLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateElementwiseLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
void populatePowerLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateActivationLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);
void populateBiasGeluLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);
void populateFastGeluLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);
void populateNormLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateGatherLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateGatherElementsLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);
void populateTopKLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateScatterElementsLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);
void populateCompressLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);
void populateOneHotLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateRangeLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateCastLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
// Shared lowering for hip.reduce_sum / hip.reduce_max / hip.reduce_prod.
// All three use the same wrap_reduce_{sum,max,prod} signature, so we
// template a single ReduceOpLowering and register all variants from one
// populate function.
void populateReduceLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateTransposeLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns);
void populateRopeLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateGqaLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns);
void populateMultiHeadAttentionLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
void populateMatMulNBitsLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
void populateQMoELoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateGatherBlockQuantizedLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
void populateGraphLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateCausalConvWithStateLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
void populateGemmLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateWhereLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateLinearAttentionLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);
void populateEqualLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateOrLoweringPatterns(const LLVMTypeConverter &converter,
                                RewritePatternSet &patterns);
void populateAndLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns);
void populateDivLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns);
void populateUnaryElementwiseLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
void populateCumSumLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populatePadLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns);
void populateTileLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateExpandLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateLessLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateGatherNDLoweringPatterns(const LLVMTypeConverter &converter,
                                      RewritePatternSet &patterns);
void populateModLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns);
void populateSliceLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateScatterNDLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns);
void populateNonZeroLoweringPatterns(const LLVMTypeConverter &converter,
                                     RewritePatternSet &patterns);
// Generic data-dependent-shape readback primitive (hip.readback_dim). Not tied
// to any single operator; any op that computes a runtime extent into a device
// i32 scalar can be sliced/sized via this.
void populateReadbackDimLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
// hip.readback_scalar: generic synchronized host-readback of a single device
// scalar of arbitrary element type (the i64/f32/f16 sibling of readback_dim).
void populateReadbackScalarLoweringPatterns(const LLVMTypeConverter &converter,
                                            RewritePatternSet &patterns);
void populateSizeLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateLoopLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateIfLoweringPatterns(const LLVMTypeConverter &converter,
                                RewritePatternSet &patterns);
void populatePoolLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateResizeLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateGlobalPoolLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIPTOLLVM_UTILS_H
