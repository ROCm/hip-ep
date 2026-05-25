/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#ifndef HIP_CONVERSION_HIPTOLLVM_UTILS_H
#define HIP_CONVERSION_HIPTOLLVM_UTILS_H

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/IR/HipShapeInterface.h"
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

inline constexpr const char *kWrapHipMemcpyAsync = "wrap_hipMemcpyAsync";
inline constexpr const char *kWrapHipMemcpy2DAsync = "wrap_hipMemcpy2DAsync";

inline constexpr const char *kMiopenConvolutionForward =
    "wrap_miopenConvolutionForward";
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
inline constexpr const char *kHipSilu = "hip_silu";
inline constexpr const char *kWrapMiopenActivationForward =
    "wrap_miopenActivationForward";                   // hip.sigmoid
inline constexpr const char *kWrapGelu = "wrap_gelu"; // hip.gelu
inline constexpr const char *kWrapElementwiseSub = "wrap_elementwise_sub";
inline constexpr const char *kWrapRotaryEmbedding = "wrap_rotary_embedding";
inline constexpr const char *kWrapMiopenOpTensor =
    "wrap_miopenOpTensor"; // hip.mul, hip.add (with 4D shape for broadcasting)
inline constexpr const char *kWrapCast = "wrap_cast";
inline constexpr const char *kWrapPower = "wrap_power";
inline constexpr const char *kWrapRange = "wrap_range";
inline constexpr const char *kWrapReduceSum = "wrap_reduce_sum";
inline constexpr const char *kWrapReduceMax = "wrap_reduce_max";
inline constexpr const char *kWrapGQA = "wrap_group_query_attention";
inline constexpr const char *kWrapMultiHeadAttention =
    "wrap_multi_head_attention";
inline constexpr const char *kWrapMatMulNBits = "wrap_matmul_nbits";
inline constexpr const char *kWrapQMoE = "wrap_qmoe";
inline constexpr const char *kWrapGemm = "wrap_gemm";
inline constexpr const char *kWrapLinearAttention = "wrap_linear_attention";
inline constexpr const char *kHipGetConstant = "hipdnn_ep_constant_get";
inline constexpr const char *kHipDNNGraphExecute = "hipdnn_graph_execute";
inline constexpr const char *kWrapCausalConvWithState =
    "wrap_causal_conv_with_state";
inline constexpr const char *kWrapWhere = "wrap_where";
inline constexpr const char *kWrapEqual = "wrap_equal";
inline constexpr const char *kWrapAnd = "wrap_and";
inline constexpr const char *kWrapNeg = "wrap_neg";
inline constexpr const char *kWrapNot = "wrap_not";
inline constexpr const char *kWrapCos = "wrap_cos";
inline constexpr const char *kWrapSin = "wrap_sin";
inline constexpr const char *kWrapDiv = "wrap_div";
inline constexpr const char *kWrapCumSum = "wrap_cumsum";
inline constexpr const char *kWrapPad = "wrap_pad";
inline constexpr const char *kWrapTile = "wrap_tile";
inline constexpr const char *kWrapExpand = "wrap_expand";
inline constexpr const char *kWrapReduceProd = "wrap_reduce_prod";
inline constexpr const char *kWrapLess = "wrap_less";
inline constexpr const char *kWrapGatherND = "wrap_gather_nd";
inline constexpr const char *kWrapSign = "wrap_sign";
inline constexpr const char *kWrapMod = "wrap_mod";
inline constexpr const char *kWrapSlice = "wrap_slice";
inline constexpr const char *kWrapScatterND = "wrap_scatter_nd";
inline constexpr const char *kWrapNonZero = "wrap_nonzero";
inline constexpr const char *kWrapSize = "wrap_size";
inline constexpr const char *kWrapShape = "wrap_shape";
inline constexpr const char *kHipdnnEpStateReadDim = "hipdnn_ep_state_read_dim";

// Phase 2 runtime ABI for translucent-propagator slot publishing.
// `_publish_dim` writes the runtime extent for a slot; `_alloc_for_slot`
// combines `dyn_pool_alloc` + `publish_buffer` into a single call; the
// `_resize` variant is reserved for Phase 3 coalescing.
inline constexpr const char *kHipdnnEpStatePublishDim =
    "hipdnn_ep_state_publish_dim";
inline constexpr const char *kHipdnnEpStateDynPoolAllocForSlot =
    "hipdnn_ep_state_dyn_pool_alloc_for_slot";
inline constexpr const char *kHipdnnEpStatePublishBufferResize =
    "hipdnn_ep_state_publish_buffer_resize";

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
  if (elemType.isUnsignedInteger(8))
    return 7; // HIPDNN_EP_DATATYPE_UINT8
  if (elemType.isSignedInteger(8) || elemType.isSignlessInteger(8))
    return 5; // HIPDNN_EP_DATATYPE_INT8
  if (elemType.isF64())
    return 6; // HIPDNN_EP_DATATYPE_DOUBLE
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

// Returns the slot id annotated by the `hip-annotate-input-dim-slots` pass
// for `(operand_idx, dim_idx)` on `op`, or -1 if no annotation exists. The
// attribute encoding is
//   hipdnn.input_dim_slots = [
//     <per-operand 0> = [[d, s], ...] | empty,
//     <per-operand 1> = [[d, s], ...] | empty,
//     ...
//   ]
// where each [d, s] is a DenseI32ArrayAttr of length 2: d = dim index, s =
// slot id. Lowerings call this on every dynamic-dim read and substitute a
// runtime `hipdnn_ep_state_read_dim` call when a slot match is found,
// so the kernel sees the runtime-published size rather than the
// pool-allocator's upper bound. Cheap: tiny attribute walk, no hot-path
// allocation.
inline int32_t lookupInputDimSlot(Operation *op, unsigned operandIdx,
                                  unsigned dimIdx) {
  auto outer = op->getAttrOfType<ArrayAttr>("hipdnn.input_dim_slots");
  if (!outer || operandIdx >= outer.size())
    return -1;
  auto perOperand = llvm::dyn_cast<ArrayAttr>(outer[operandIdx]);
  if (!perOperand)
    return -1;
  for (Attribute pair : perOperand) {
    auto arr = llvm::dyn_cast<DenseI32ArrayAttr>(pair);
    if (!arr || arr.size() != 2)
      continue;
    if (static_cast<unsigned>(arr[0]) == dimIdx)
      return arr[1];
  }
  return -1;
}

inline constexpr const char *kHipdnnEpStatePeekBuffer =
    "hipdnn_ep_state_peek_buffer";

// Emit `call @hipdnn_ep_state_peek_buffer(state, slot_id) : (ptr, i32) -> ptr`.
// Returns the runtime-published GPU pointer for `slotId`, or null when
// the producer published a null buffer (N=0 case — Category-C wrappers
// skip the allocation when there is nothing to fill, but the slot
// dim is still published as 0). Used by consumer lowerings to redirect
// their input pointer from the upper-bound DPS init buffer (which the
// Category-C producer never writes into) to the runtime-allocated
// exact-size buffer.
//
// We use the *peek* (non-aborting) flavor specifically to keep N=0
// scenarios alive: the kernel launches receive a null pointer + 0
// element count and short-circuit at the dispatcher level. The
// aborting `read_buffer` flavor is reserved for the EP-side resolver
// where a missing buffer indicates a real bug.
inline Value emitReadBufferCall(Value state, int32_t slotId,
                                ConversionPatternRewriter &rewriter,
                                Location loc) {
  ModuleOp module = state.getDefiningOp()
                        ? state.getDefiningOp()->getParentOfType<ModuleOp>()
                        : nullptr;
  if (!module) {
    Block *block = rewriter.getInsertionBlock();
    if (block) {
      Operation *parent = block->getParentOp();
      if (parent)
        module = parent->getParentOfType<ModuleOp>();
    }
  }
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  SmallVector<Type, 2> paramTypes = {ptrType, i32Type};
  FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
      rewriter, module, kHipdnnEpStatePeekBuffer, paramTypes, ptrType);
  Value slotConst = LLVM::ConstantOp::create(
      rewriter, loc, i32Type, rewriter.getI32IntegerAttr(slotId));
  return LLVM::CallOp::create(rewriter, loc, *funcOp,
                              ValueRange{state, slotConst})
      .getResult();
}

// Slot-aware variant of `extractContiguousMemRefPtr`. When the
// `hipdnn.input_slot_buffers` attribute (set by
// `hip-annotate-input-dim-slots`) records a non-negative slot id for
// `operandIdx`, returns the runtime-published exact-size buffer via
// `hipdnn_ep_state_read_buffer`. Otherwise falls back to the
// descriptor's `alignedPtr`.
//
// Triggers ONLY when the operand's IMMEDIATE producer is a slot
// publisher (e.g. `hip.nonzero`) — those publishers allocate a
// separate exact-size buffer at runtime and ignore their upper-bound
// DPS init, so the consumer needs to read the published pointer
// instead of the descriptor. Translucent propagators (e.g.
// `hip.transpose` consumed by `hip.scatter_nd`) DO write into their
// upper-bound DPS init, so the descriptor pointer is the correct one
// for the propagator's consumers — the annotation pass deliberately
// does NOT set `hipdnn.input_slot_buffers` for that case (only
// `hipdnn.input_dim_slots` for the shape).
inline Value extractContiguousMemRefPtrWithSlot(
    Operation *op, unsigned operandIdx, Value descriptor, Value state,
    ConversionPatternRewriter &rewriter, Location loc) {
  if (state) {
    if (auto bufs =
            op->getAttrOfType<DenseI32ArrayAttr>("hipdnn.input_slot_buffers")) {
      if (operandIdx < bufs.size()) {
        int32_t slotId = bufs[operandIdx];
        if (slotId >= 0)
          return emitReadBufferCall(state, slotId, rewriter, loc);
      }
    }
  }
  return extractContiguousMemRefPtr(descriptor, rewriter, loc);
}

// Emit `call @hipdnn_ep_state_read_dim(state, slot_id) : (ptr, i32) -> i64`.
// Looks up (or declares) the runtime symbol on demand. Returns the i64 SSA
// value of the published dim. The caller is responsible for ensuring
// `state` is the runtime state pointer (typically the converted
// `!hip.context` operand of the consumer op).
inline Value emitReadDimCall(Value state, int32_t slotId,
                             ConversionPatternRewriter &rewriter,
                             Location loc) {
  ModuleOp module = state.getDefiningOp()
                        ? state.getDefiningOp()->getParentOfType<ModuleOp>()
                        : nullptr;
  if (!module) {
    Block *block = rewriter.getInsertionBlock();
    if (block) {
      Operation *parent = block->getParentOp();
      if (parent)
        module = parent->getParentOfType<ModuleOp>();
    }
  }
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  SmallVector<Type, 2> paramTypes = {ptrType, i32Type};
  FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
      rewriter, module, kHipdnnEpStateReadDim, paramTypes, i64Type);
  Value slotConst = LLVM::ConstantOp::create(
      rewriter, loc, i32Type, rewriter.getI32IntegerAttr(slotId));
  return LLVM::CallOp::create(rewriter, loc, *funcOp,
                              ValueRange{state, slotConst})
      .getResult();
}

// Emit LLVM IR that evaluates a DimSpec tree to its i64 dim value at
// runtime. Walks the tree recursively starting from the root node.
//
// Supported leaves:
//   * Static(N)        -> llvm.mlir.constant N : i64
//   * RuntimeSlot(N)   -> call @hipdnn_ep_state_read_dim(state, N)
//
// Supported arithmetic (Phase 2.5): Add, Sub, Mul, FloorDiv, CeilDiv,
// Min, Max -- each lowered to the corresponding LLVM IR primitive
// over the recursive sub-expression results.
//
// Forbidden in-DLL: InputDim, InputValueI64 -- those are EP-side
// Cat-B leaves resolved by `MlirCustomOp::resolveValueFromI64Tensor`.
// If the tree contains one, returns a null Value so the caller can
// emit a verifier failure / fall back to the legacy descriptor read.
//
// Returns a null Value on any unsupported node so callers must
// always null-check the result.
inline Value emitDimSpecEvaluator(const DimSpec &ds, Value state,
                                  ConversionPatternRewriter &rewriter,
                                  Location loc) {
  if (ds.nodes().empty())
    return Value();
  Type i64Type = rewriter.getI64Type();

  std::function<Value(int32_t)> eval = [&](int32_t idx) -> Value {
    if (idx < 0 || idx >= (int32_t)ds.nodes().size())
      return Value();
    const auto &n = ds.nodes()[idx];
    switch (n.kind) {
    case DimSpecKind::Static:
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(n.value));
    case DimSpecKind::RuntimeSlot:
      if (!state)
        return Value();
      return emitReadDimCall(state, n.slot_id, rewriter, loc);
    case DimSpecKind::InputDim:
    case DimSpecKind::InputValueI64:
      // EP-only leaves; the consumer is responsible for picking the
      // legacy descriptor read instead of calling the evaluator.
      return Value();
    case DimSpecKind::Add:
    case DimSpecKind::Sub:
    case DimSpecKind::Mul:
    case DimSpecKind::FloorDiv:
    case DimSpecKind::CeilDiv:
    case DimSpecKind::Min:
    case DimSpecKind::Max: {
      Value lhs = eval(n.lhs);
      Value rhs = eval(n.rhs);
      if (!lhs || !rhs)
        return Value();
      switch (n.kind) {
      case DimSpecKind::Add:
        return LLVM::AddOp::create(rewriter, loc, lhs, rhs);
      case DimSpecKind::Sub:
        return LLVM::SubOp::create(rewriter, loc, lhs, rhs);
      case DimSpecKind::Mul:
        return LLVM::MulOp::create(rewriter, loc, lhs, rhs);
      case DimSpecKind::FloorDiv:
        // Operands are non-negative dim sizes -> sdiv is equivalent to
        // floordiv for our use case (Phase 2.5 corner notes); using
        // sdiv keeps the IR small.
        return LLVM::SDivOp::create(rewriter, loc, lhs, rhs);
      case DimSpecKind::CeilDiv: {
        // ceildiv(a, b) = (a + b - 1) / b for positive b.
        Value one = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(1));
        Value rhsMinusOne = LLVM::SubOp::create(rewriter, loc, rhs, one);
        Value sum = LLVM::AddOp::create(rewriter, loc, lhs, rhsMinusOne);
        return LLVM::SDivOp::create(rewriter, loc, sum, rhs);
      }
      case DimSpecKind::Min: {
        Value cmp = LLVM::ICmpOp::create(rewriter, loc,
                                         LLVM::ICmpPredicate::sle, lhs, rhs);
        return LLVM::SelectOp::create(rewriter, loc, cmp, lhs, rhs);
      }
      case DimSpecKind::Max: {
        Value cmp = LLVM::ICmpOp::create(rewriter, loc,
                                         LLVM::ICmpPredicate::sge, lhs, rhs);
        return LLVM::SelectOp::create(rewriter, loc, cmp, lhs, rhs);
      }
      default:
        return Value();
      }
    }
    }
    return Value();
  };
  return eval(0);
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

// Slot-aware variant of `getMemRefDimSize`. When `op` has been annotated
// by `hip-annotate-input-dim-slots` with a slot for `(operandIdx, dimIdx)`,
// returns the runtime-published dim via `hipdnn_ep_state_read_dim`
// instead of reading the descriptor (which encodes the upper-bound
// pool allocation). `state` is the runtime state pointer required for
// the read_dim call. `descriptor` is the converted memref descriptor
// for that operand. `operandIdx` is the ORIGINAL (pre-conversion)
// operand index, matching the indexing used by the annotation pass.
//
// Use this in lowerings that emit shape/count parameters for ops whose
// inputs may be Category-C upper-bound buffers (e.g. transpose of a
// NonZero output). For ops whose inputs cannot consume Category-C
// outputs (constant-time-shape inputs, weights, etc.), the plain
// `getMemRefDimSize` is sufficient.
inline Value getMemRefDimSizeWithSlot(Operation *op, unsigned operandIdx,
                                      MemRefType type, unsigned dimIdx,
                                      Value descriptor, Value state,
                                      ConversionPatternRewriter &rewriter,
                                      Location loc) {
  if (type.isDynamicDim(dimIdx)) {
    int32_t slotId = lookupInputDimSlot(op, operandIdx, dimIdx);
    if (slotId >= 0 && state)
      return emitReadDimCall(state, slotId, rewriter, loc);
  }
  return getMemRefDimSize(type, dimIdx, descriptor, rewriter, loc);
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

// Emit `call @hipdnn_ep_state_publish_dim(state, slot_id, count) :
//                                          (ptr, i32, i64) -> void`.
inline void emitPublishDimCall(Value state, int32_t slotId, Value count,
                               ConversionPatternRewriter &rewriter,
                               Location loc) {
  ModuleOp module = state.getDefiningOp()
                        ? state.getDefiningOp()->getParentOfType<ModuleOp>()
                        : nullptr;
  if (!module) {
    Block *block = rewriter.getInsertionBlock();
    if (block)
      if (Operation *p = block->getParentOp())
        module = p->getParentOfType<ModuleOp>();
  }
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  SmallVector<Type, 3> paramTypes = {ptrType, i32Type, i64Type};
  FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
      rewriter, module, kHipdnnEpStatePublishDim, paramTypes,
      LLVM::LLVMVoidType::get(module.getContext()));
  Value slotConst = LLVM::ConstantOp::create(
      rewriter, loc, i32Type, rewriter.getI32IntegerAttr(slotId));
  LLVM::CallOp::create(rewriter, loc, *funcOp,
                       ValueRange{state, slotConst, count});
}

// Emit `call @hipdnn_ep_state_publish_buffer_resize(state, slot_id, bytes) :
//                                                   (ptr, i32, i64) -> ptr`.
//
// This is the publisher-side primitive used by translucent-propagator
// wrapper lowerings to obtain the GPU buffer the kernel writes its
// result into. Uses the resize-or-alloc helper rather than the plain
// `dyn_pool_alloc_for_slot` so Phase 3 coalescing (where the same slot
// id may publish twice in one Compute) is transparent to the lowering.
inline Value emitPublishBufferResizeCall(Value state, int32_t slotId,
                                         Value bytes,
                                         ConversionPatternRewriter &rewriter,
                                         Location loc) {
  ModuleOp module = state.getDefiningOp()
                        ? state.getDefiningOp()->getParentOfType<ModuleOp>()
                        : nullptr;
  if (!module) {
    Block *block = rewriter.getInsertionBlock();
    if (block)
      if (Operation *p = block->getParentOp())
        module = p->getParentOfType<ModuleOp>();
  }
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  SmallVector<Type, 3> paramTypes = {ptrType, i32Type, i64Type};
  FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
      rewriter, module, kHipdnnEpStatePublishBufferResize, paramTypes, ptrType);
  Value slotConst = LLVM::ConstantOp::create(
      rewriter, loc, i32Type, rewriter.getI32IntegerAttr(slotId));
  return LLVM::CallOp::create(rewriter, loc, *funcOp,
                              ValueRange{state, slotConst, bytes})
      .getResult();
}

// Helper used by translucent-propagator lowerings (transpose, gather,
// tile, expand, ...) when the op has been annotated with
// `hipdnn.output_slot_ids` by the Phase 2.3 ReservePropagatorSlots
// pass. Emits a `publish_dim` call for every (resultIdx, dimIdx) pair
// whose slot id is non-negative. `dimSizeProvider` is a callback that
// returns the i64 SSA value of the runtime extent for a given
// (resultIdx, dimIdx). Returns nothing -- pure side-effect.
//
// Lowerings call this AFTER they've computed the output shape and
// BEFORE they emit the wrap_* kernel call, so the kernel + the
// publish are part of the same generated-code block. Use only when the
// lowering already has the dim values in hand; ops that need operand-
// tensor reads to know their output shape (Range, Slice, Tile, Expand,
// Pad, ...) should use the wrapper-side path via
// `emitOutputSlotIdsAlloca` instead and let the wrapper publish.
template <typename DimSizeFn>
inline void emitPropagatorSlotPublishes(Operation *op, Value state,
                                        DimSizeFn dimSizeProvider,
                                        ConversionPatternRewriter &rewriter,
                                        Location loc) {
  auto grid = op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids");
  if (!grid || !state)
    return;
  for (unsigned r = 0; r < grid.size(); ++r) {
    auto perResult = llvm::dyn_cast<DenseI32ArrayAttr>(grid[r]);
    if (!perResult)
      continue;
    for (int64_t d = 0; d < perResult.size(); ++d) {
      int32_t slot = perResult.asArrayRef()[d];
      if (slot < 0)
        continue;
      Value size = dimSizeProvider(r, (unsigned)d);
      if (!size)
        continue;
      emitPublishDimCall(state, slot, size, rewriter, loc);
    }
  }
}

// Companion to `emitPropagatorSlotPublishes`: materialise the
// `hipdnn.output_slot_ids` attribute as a flat int32_t[total] stack
// array suitable for being passed as a trailing `const int32_t *
// output_slot_ids` parameter to a runtime wrapper that does the
// publish itself.
//
// Layout: row-major, length = sum_r rank_r. Each result's row is
// concatenated in result-index order. Entries are the slot id (>=0) or
// -1 for "no slot".
//
// If the op carries no `hipdnn.output_slot_ids` attribute, returns a
// null `LLVM::ZeroOp` pointer + sets `outTotalLen` to 0 so the
// wrapper can fast-skip. Wrappers MUST therefore tolerate a null /
// zero-length slot-ids pointer.
inline Value emitOutputSlotIdsAlloca(Operation *op,
                                     ConversionPatternRewriter &rewriter,
                                     Location loc, int64_t &outTotalLen) {
  outTotalLen = 0;
  Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
  Type i32Type = rewriter.getI32Type();
  Type i64Type = rewriter.getI64Type();
  auto grid = op->getAttrOfType<ArrayAttr>("hipdnn.output_slot_ids");
  if (!grid)
    return LLVM::ZeroOp::create(rewriter, loc, ptrType);

  SmallVector<int32_t, 8> flat;
  for (unsigned r = 0; r < grid.size(); ++r) {
    auto perResult = llvm::dyn_cast<DenseI32ArrayAttr>(grid[r]);
    if (!perResult)
      continue;
    for (int64_t d = 0; d < perResult.size(); ++d)
      flat.push_back(perResult.asArrayRef()[d]);
  }
  outTotalLen = (int64_t)flat.size();
  if (outTotalLen == 0)
    return LLVM::ZeroOp::create(rewriter, loc, ptrType);

  auto arrayTy = LLVM::LLVMArrayType::get(i32Type, outTotalLen);
  Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                          rewriter.getI64IntegerAttr(1));
  Value alloca = LLVM::AllocaOp::create(rewriter, loc, ptrType, arrayTy, oneI64,
                                        /*alignment=*/4);
  for (int64_t i = 0; i < outTotalLen; ++i) {
    Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                         rewriter.getI32IntegerAttr(i));
    Value val = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                         rewriter.getI32IntegerAttr(flat[i]));
    Value gep =
        LLVM::GEPOp::create(rewriter, loc, ptrType, i32Type, alloca, idx);
    LLVM::StoreOp::create(rewriter, loc, val, gep);
  }
  return alloca;
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
void populateSizeLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);
void populateLoopLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns);

void populateConstantOfShapeLoweringPatterns(const LLVMTypeConverter &converter,
                                             RewritePatternSet &patterns);
void populateShapeLoweringPatterns(const LLVMTypeConverter &converter,
                                   RewritePatternSet &patterns);

} // namespace hip
} // namespace mlir

#endif // HIP_CONVERSION_HIPTOLLVM_UTILS_H
