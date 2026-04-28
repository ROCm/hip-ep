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

inline constexpr const char *kHipMalloc = "hip_device_malloc";
inline constexpr const char *kHipFree = "hip_device_free";
inline constexpr const char *kHipGetPoolBase = "hipdnn_ep_get_pool_base";

inline constexpr const char *kMiopenConvolutionForward =
    "wrap_miopenConvolutionForward";
inline constexpr const char *kWrapHipblasltMatmul = "wrap_hipblasLtMatmul";
inline constexpr const char *kWrapMiopenT5LayerNormForward =
    "wrap_miopenT5LayerNormForward";
inline constexpr const char *kWrapSkipSimplifiedLayerNorm =
    "wrap_skip_simplified_layer_norm";
inline constexpr const char *kMiopenAdd = "hip_miopen_add";
inline constexpr const char *kMiopenMul = "hip_miopen_mul";
inline constexpr const char *kMiopenSoftmax = "hip_miopen_softmax";
inline constexpr const char *kHipTranspose = "hip_transpose";
inline constexpr const char *kWrapGather = "wrap_gather";
inline constexpr const char *kHipSilu = "hip_silu";
inline constexpr const char *kWrapMiopenActivationForward =
    "wrap_miopenActivationForward"; // hip.sigmoid
inline constexpr const char *kWrapElementwiseSub = "wrap_elementwise_sub";
inline constexpr const char *kWrapRotaryEmbedding = "wrap_rotary_embedding";
inline constexpr const char *kWrapMiopenOpTensor =
    "wrap_miopenOpTensor"; // hip.mul, hip.add (with 4D shape for broadcasting)
inline constexpr const char *kWrapCast = "wrap_cast";
inline constexpr const char *kWrapPower = "wrap_power";
inline constexpr const char *kWrapReduceSum = "wrap_reduce_sum";
inline constexpr const char *kWrapGQA = "wrap_group_query_attention";
inline constexpr const char *kWrapMatMulNBits = "wrap_matmul_nbits";
inline constexpr const char *kWrapQMoE = "wrap_qmoe";
inline constexpr const char *kWrapGemm = "wrap_gemm";
inline constexpr const char *kHipGetConstant = "hipdnn_ep_constant_get";
inline constexpr const char *kHipDNNGraphExecute = "hipdnn_graph_execute";
inline constexpr const char *kWrapCausalConvWithState =
    "wrap_causal_conv_with_state";

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
inline Value extractMemRefPtr(Value memrefDesc,
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
inline Value extractOptionalMemRefPtr(Value memrefDesc,
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
void populateMatmulLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateElementwiseLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
void populatePowerLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateActivationLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);
void populateNormLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateGatherLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns);
void populateCastLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateReduceSumLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns);
void populateTransposeLoweringPatterns(const LLVMTypeConverter &converter,
                                       RewritePatternSet &patterns);
void populateRopeLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateGqaLoweringPatterns(const LLVMTypeConverter &converter,
                                 RewritePatternSet &patterns);
void populateMatMulNBitsLoweringPatterns(const LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);
void populateQMoELoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateGraphLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);
void populateCausalConvWithStateLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns);
void populateGemmLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateDumpTensorLoweringPatterns(const LLVMTypeConverter &converter,
                                        RewritePatternSet &patterns);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIPTOLLVM_UTILS_H
