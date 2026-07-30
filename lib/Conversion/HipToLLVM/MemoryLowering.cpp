/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

/// If module metadata is present, verify callback rank matches the ONNX
/// output shape recorded at compile time (hipdnn.output_shapes). Metadata
/// describes @main_graph only — auxiliary funcs in the same module are skipped.
static LogicalResult verifyCallbackRankAgainstMetadata(ModuleOp module,
                                                       func::FuncOp func,
                                                       int64_t outIdx,
                                                       int64_t callbackRank,
                                                       Location loc) {
  if (!func || func.getName() != "main_graph")
    return success();
  auto outputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");
  if (!outputShapes || outIdx < 0 ||
      outIdx >= static_cast<int64_t>(outputShapes.size()))
    return success();
  auto shapeAttr = dyn_cast<DenseI64ArrayAttr>(outputShapes[outIdx]);
  if (!shapeAttr)
    return success();
  if (static_cast<int64_t>(shapeAttr.size()) != callbackRank)
    return emitError(loc) << "hip.alloc_output callback rank " << callbackRank
                          << " does not match hipdnn.output_shapes[" << outIdx
                          << "] rank " << shapeAttr.size();
  return success();
}

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

// --- GetPoolOp:
//     hip.get_pool(%ctx, %pool_size) {domain_id = N} : memref<?xi8>
//       -> llvm.call @hipdnn_ep_get_pool_base(state, N, size) + memref desc.
//
// The domain_id attribute (default 0) is materialized as an i32 constant
// argument so the runtime can select the right per-domain pool slot.
// Single-domain models emit `domain_id = 0` and round-trip identically to
// the pre-multi-domain IR (printer elides the default).
struct GetPoolOpLowering : public ConvertOpToLLVMPattern<GetPoolOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    MemRefType memRefType = cast<MemRefType>(op.getPool().getType());

    // Runtime ABI: hipdnn_ep_get_pool_base(state: ptr, domain_id: i32,
    // needed_size: i64) -> ptr.
    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kHipGetPoolBase,
                               {ptrType, i32Type, i64Type}, ptrType);
    if (failed(funcOp))
      return failure();

    Value poolSize = adaptor.getPoolSize();
    Value domainIdVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Type,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getDomainId())));
    Value rawPtr = LLVM::CallOp::create(
                       rewriter, loc, *funcOp,
                       ValueRange{adaptor.getCtx(), domainIdVal, poolSize})
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

// --- GetHostScratchOp: hip.get_host_scratch(%ctx, %scratch_size) :memref<?xi8>
//     -> llvm.call @hipdnn_ep_get_host_scratch_base(state, size) + descriptor.
// The runtime returns hipHostMalloc(hipHostMallocMapped) memory, which is
// accessible from both host and device. The result memref uses the default
// address space (AS 0) — so host stores from materialized scalars work, and
// downstream GPU ops that consume it via hip-promote-strided-hip-operands /
// memref.view see the same pointer.
struct GetHostScratchOpLowering
    : public ConvertOpToLLVMPattern<GetHostScratchOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetHostScratchOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    MemRefType memRefType = cast<MemRefType>(op.getScratch().getType());

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGetHostScratch, {ptrType, i64Type}, ptrType);
    if (failed(funcOp))
      return failure();

    Value scratchSize = adaptor.getScratchSize();
    Value rawPtr =
        LLVM::CallOp::create(rewriter, loc, *funcOp,
                             ValueRange{adaptor.getCtx(), scratchSize})
            .getResult();

    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();

    Value hostPtr = rawPtr;
    if (cast<LLVM::LLVMPointerType>(rawPtr.getType()).getAddressSpace() !=
        *addrSpace)
      hostPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          rawPtr);

    Value stride1 = LLVM::ConstantOp::create(
        rewriter, loc, rewriter.getI64Type(), rewriter.getI64IntegerAttr(1));

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, hostPtr, hostPtr, {scratchSize}, {stride1}, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

// --- AllocOutputOp: hip.alloc_output(%ctx, %dyn...) {out_idx} : memref<...>
//     -> llvm.call @hipdnn_ep_alloc_output(state, out_idx, shape, rank,
//        elem_size) + a memref descriptor over the returned device pointer.
//
// The op obtains a graph-output buffer from the EP output allocator at the
// point where the output shape is known. The runtime callback shape uses the
// func.return / ONNX ABI rank when it differs from the internal memref (e.g. a
// rank-3 compute buffer returned through collapse_shape as rank-2). That rank
// mismatch is recorded by hip-use-output-allocator as the discardable attrs
// `hipdnn.abi_shape` / `hipdnn.abi_groups` (see stampAbiReshapeAttrs) -- read
// here to re-derive each external dim from the internal alloc sizes (which
// dominate this alloc site). The memref descriptor handed to downstream compute
// ops still uses the op's own (internal) type.
//
// Before (internal rank 3, ONNX return rank 2; abi_groups=[2,1]):
//   %out = hip.alloc_output(%ctx, %d0, %d1)
//            {out_idx = 0,
//             hipdnn.abi_shape  = array<i64: kDynamic, 2560>,
//             hipdnn.abi_groups = array<i64: 2, 1>} : memref<?x?x2560xf16>
//
// After (callback shape rank 2, ext dim0 = %d0 * %d1):
//   %shape = llvm.alloca ... x !llvm.array<2 x i64>
//   %p0    = llvm.mul %d0, %d1
//   llvm.store %p0,    %shape[0]
//   llvm.store %c2560, %shape[1]
//   %p = llvm.call @hipdnn_ep_alloc_output(..., %shape, %c2_rank, ...)
struct AllocOutputOpLowering : public ConvertOpToLLVMPattern<AllocOutputOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(AllocOutputOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MemRefType memRefType = cast<MemRefType>(op.getMemref().getType());

    if (!isConvertibleAndHasIdentityMaps(memRefType))
      return rewriter.notifyMatchFailure(op, "incompatible memref type");

    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();

    Type elemType = memRefType.getElementType();
    if (!elemType.isIntOrFloat())
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    // Round bits up to whole bytes so sub-byte types report >= 1 byte: a
    // bool (`i1`) output must be 1 byte, not 1/8 == 0.
    int64_t elemSizeBytes = (elemType.getIntOrFloatBitWidth() + 7) / 8;
    if (elemSizeBytes <= 0)
      return rewriter.notifyMatchFailure(op, "unsupported element bit width");

    int64_t rank = memRefType.getRank();

    // internalSizes[] interleave static dims (type constants) with the
    // dynamic-size operands (in type order); strides[] are row-major. They
    // build the memref descriptor the downstream compute ops consume, so they
    // always describe the op's own (internal) type. All are i64 (index lowers
    // to i64), materialized here, so they dominate the callback arithmetic
    // below.
    SmallVector<Value, 4> internalSizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, internalSizes, strides, sizeBytes, true);

    // Runtime callback shape: the ONNX / func.return output shape when it
    // differs from the internal memref rank. hip-use-output-allocator stamps
    // `hipdnn.abi_shape` (external shape) + `hipdnn.abi_groups`:
    //   * collapse (internal rank > external): groups[e] = # internal dims
    //     folded into external dim e; dynamic external dims = product of those
    //     internal sizes.
    //   * expand (internal rank < external): groups[i] = # external dims
    //     expanded from internal dim i; each dynamic external dim in that
    //     group takes the corresponding internal size (static dims from attr).
    // Absent / malformed attrs -> internal rank/sizes verbatim.
    auto abiShapeAttr = op->getAttrOfType<DenseI64ArrayAttr>(kAbiShapeAttrName);
    auto abiGroupsAttr =
        op->getAttrOfType<DenseI64ArrayAttr>(kAbiGroupsAttrName);

    SmallVector<Value, 4> callbackSizes;
    int64_t callbackRank = rank;
    if (abiShapeAttr && abiGroupsAttr) {
      ArrayRef<int64_t> abiShape = abiShapeAttr.asArrayRef();
      ArrayRef<int64_t> abiGroups = abiGroupsAttr.asArrayRef();
      int64_t extRank = static_cast<int64_t>(abiShape.size());
      int64_t groupsRank = static_cast<int64_t>(abiGroups.size());
      bool ok = false;

      // Collapse: one group entry per external dim.
      if (groupsRank == extRank && extRank < rank) {
        int64_t internalDim = 0;
        ok = true;
        for (int64_t e : llvm::seq<int64_t>(0, extRank)) {
          int64_t cnt = abiGroups[e];
          if (cnt <= 0 || internalDim + cnt > rank) {
            ok = false;
            break;
          }
          Value dim;
          if (!ShapedType::isDynamic(abiShape[e])) {
            dim = LLVM::ConstantOp::create(
                rewriter, loc, i64Type,
                rewriter.getI64IntegerAttr(abiShape[e]));
          } else {
            dim = internalSizes[internalDim];
            for (int64_t j : llvm::seq<int64_t>(1, cnt))
              dim = LLVM::MulOp::create(rewriter, loc, dim,
                                        internalSizes[internalDim + j]);
          }
          callbackSizes.push_back(dim);
          internalDim += cnt;
        }
        if (ok && internalDim != rank)
          ok = false;
        if (ok)
          callbackRank = extRank;
      }

      // Expand: one group entry per internal dim.
      if (!ok && groupsRank == rank && extRank > rank) {
        int64_t externalDim = 0;
        ok = true;
        for (int64_t i : llvm::seq<int64_t>(0, rank)) {
          int64_t cnt = abiGroups[i];
          if (cnt <= 0 || externalDim + cnt > extRank) {
            ok = false;
            break;
          }
          bool usedInternal = false;
          for (int64_t j : llvm::seq<int64_t>(0, cnt)) {
            int64_t e = externalDim + j;
            if (!ShapedType::isDynamic(abiShape[e])) {
              callbackSizes.push_back(LLVM::ConstantOp::create(
                  rewriter, loc, i64Type,
                  rewriter.getI64IntegerAttr(abiShape[e])));
            } else {
              if (usedInternal) {
                ok = false;
                break;
              }
              callbackSizes.push_back(internalSizes[i]);
              usedInternal = true;
            }
          }
          if (!ok)
            break;
          externalDim += cnt;
        }
        if (ok && externalDim != extRank)
          ok = false;
        if (ok)
          callbackRank = extRank;
      }

      if (!ok)
        callbackSizes.clear();
    }
    if (callbackSizes.empty()) {
      callbackSizes.assign(internalSizes.begin(), internalSizes.end());
      callbackRank = rank;
    }

    auto parentFunc = op->getParentOfType<func::FuncOp>();
    if (failed(verifyCallbackRankAgainstMetadata(
            module, parentFunc, op.getOutIdx(), callbackRank, loc)))
      return failure();

    // Stack-allocate the i64[callbackRank] shape array for the runtime ABI.
    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                            rewriter.getI64IntegerAttr(1));
    auto shapeArrayType =
        LLVM::LLVMArrayType::get(i64Type, callbackRank > 0 ? callbackRank : 1);
    Value shapeAlloca = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, shapeArrayType, oneI64, /*alignment=*/8);
    for (int64_t i : llvm::seq<int64_t>(0, callbackRank)) {
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                           rewriter.getI32IntegerAttr(i));
      Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                      shapeAlloca, idx);
      LLVM::StoreOp::create(rewriter, loc, callbackSizes[i], gep);
    }

    Value outIdxVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getOutIdx()));
    Value rankVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(callbackRank));
    Value elemSizeVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elemSizeBytes));

    // void* hipdnn_ep_alloc_output(void* state, int64_t out_idx,
    //                              const int64_t* shape, int64_t rank,
    //                              int64_t elem_size)
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipAllocOutput,
        {ptrType, i64Type, ptrType, i64Type, i64Type}, ptrType);
    if (failed(funcOp))
      return failure();

    Value rawPtr =
        LLVM::CallOp::create(rewriter, loc, *funcOp,
                             ValueRange{adaptor.getCtx(), outIdxVal,
                                        shapeAlloca, rankVal, elemSizeVal})
            .getResult();

    // The runtime returns a generic (AS 0) pointer; cast to the memref's
    // address space if it differs (mirrors GetConstant / AllocOp).
    FailureOr<unsigned> addrSpace =
        getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();

    Value dataPtr = rawPtr;
    if (cast<LLVM::LLVMPointerType>(rawPtr.getType()).getAddressSpace() !=
        *addrSpace)
      dataPtr = LLVM::AddrSpaceCastOp::create(
          rewriter, loc,
          LLVM::LLVMPointerType::get(rewriter.getContext(), *addrSpace),
          rawPtr);

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, dataPtr, dataPtr, internalSizes, strides, rewriter);
    rewriter.replaceOp(op, {desc});
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

/// Static strides in elements for identity or strided memref types (no dynamic
/// dims/strides).
static FailureOr<SmallVector<int64_t>> tryStaticStridesElems(MemRefType ty) {
  ArrayRef<int64_t> shape = ty.getShape();
  unsigned rank = ty.getRank();

  MemRefLayoutAttrInterface layout = ty.getLayout();
  if (layout.isIdentity()) {
    SmallVector<int64_t> out(rank);
    int64_t running = 1;
    for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
      out[static_cast<unsigned>(i)] = running;
      if (ShapedType::isDynamic(shape[i]))
        return failure();
      running *= shape[i];
    }
    return out;
  }

  if (auto sl = dyn_cast<StridedLayoutAttr>(layout)) {
    auto strides = llvm::to_vector(sl.getStrides());
    if (strides.size() != rank)
      return failure();
    for (int64_t s : strides)
      if (ShapedType::isDynamic(s))
        return failure();
    return strides;
  }

  return failure();
}

static FailureOr<SmallVector<int64_t>>
computeRowMajorStridesElems(ArrayRef<int64_t> shape) {
  unsigned rank = shape.size();
  SmallVector<int64_t> out(rank);
  int64_t running = 1;
  for (int i = static_cast<int>(rank) - 1; i >= 0; --i) {
    out[static_cast<unsigned>(i)] = running;
    if (ShapedType::isDynamic(shape[static_cast<unsigned>(i)]))
      return failure();
    running *= shape[static_cast<unsigned>(i)];
  }
  return out;
}

static FailureOr<int64_t> tryElemSizeBytes(Type elemTy) {
  if (!elemTy.isIntOrFloat())
    return failure();
  unsigned bits = elemTy.getIntOrFloatBitWidth();
  if (bits % 8 != 0)
    return failure();
  return static_cast<int64_t>(bits / 8);
}

static bool strideVectorsEqual(ArrayRef<int64_t> a, ArrayRef<int64_t> b) {
  if (a.size() != b.size())
    return false;
  for (size_t i = 0, e = a.size(); i != e; ++i) {
    if (a[i] != b[i])
      return false;
  }
  return true;
}

/// Lowers memref.copy on GPU memrefs to wrap_hipMemcpyAsync /
/// wrap_hipMemcpy2DAsync when strides are statically known. Otherwise leaves
/// conversion to the default MemRef→LLVM copy lowering (benefit 1).
struct MemRefCopyOpLowering : public ConvertOpToLLVMPattern<memref::CopyOp> {
  MemRefCopyOpLowering(const LLVMTypeConverter &converter)
      : ConvertOpToLLVMPattern(converter, PatternBenefit(10)) {}

  LogicalResult
  matchAndRewrite(memref::CopyOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();

    auto llvmFn = op->getParentOfType<LLVM::LLVMFuncOp>();
    if (!llvmFn)
      return rewriter.notifyMatchFailure(op, "expected parent llvm.func");

    auto srcTy = dyn_cast<MemRefType>(op.getSource().getType());
    auto dstTy = dyn_cast<MemRefType>(op.getTarget().getType());
    if (!srcTy || !dstTy)
      return rewriter.notifyMatchFailure(op, "expected ranked memrefs");

    if (srcTy.getShape() != dstTy.getShape())
      return rewriter.notifyMatchFailure(op, "copy shape mismatch");

    FailureOr<int64_t> elemBytesOr = tryElemSizeBytes(srcTy.getElementType());
    if (failed(elemBytesOr))
      return rewriter.notifyMatchFailure(op, "unsupported element type");
    int64_t elemBytes = *elemBytesOr;

    FailureOr<SmallVector<int64_t>> srcStridesOr = tryStaticStridesElems(srcTy);
    FailureOr<SmallVector<int64_t>> dstStridesOr = tryStaticStridesElems(dstTy);
    if (failed(srcStridesOr) || failed(dstStridesOr))
      return rewriter.notifyMatchFailure(op, "need static strides/layout");

    ArrayRef<int64_t> srcStrides = *srcStridesOr;
    ArrayRef<int64_t> dstStrides = *dstStridesOr;

    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = LLVM::LLVMPointerType::get(rewriter.getContext(), 0);
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();

    Value statePtr = llvmFn.getArgument(0);
    Value srcPtr = extractMemRefDataPtr(adaptor.getSource(), srcTy,
                                        getTypeConverter(), rewriter, loc);
    Value dstPtr = extractMemRefDataPtr(adaptor.getTarget(), dstTy,
                                        getTypeConverter(), rewriter, loc);
    if (!srcPtr || !dstPtr)
      return failure();

    int64_t rank = srcTy.getRank();
    ArrayRef<int64_t> shape = srcTy.getShape();

    auto i64Const = [&](int64_t v) -> Value {
      return LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                      rewriter.getI64IntegerAttr(v));
    };

    FailureOr<SmallVector<int64_t>> denseStridesOr =
        computeRowMajorStridesElems(shape);
    if (failed(denseStridesOr))
      return rewriter.notifyMatchFailure(op, "need static shape");

    // Single D2D memcpy only for dense row-major storage (no holes).
    if (strideVectorsEqual(srcStrides, dstStrides) &&
        strideVectorsEqual(srcStrides, ArrayRef<int64_t>(*denseStridesOr))) {
      int64_t numElems = 1;
      for (int64_t d : shape)
        numElems *= d;
      int64_t totalBytes = numElems * elemBytes;

      FailureOr<LLVM::LLVMFuncOp> memcpyFn =
          LLVM::lookupOrCreateFn(rewriter, module, kWrapHipMemcpyAsync,
                                 {ptrType, ptrType, ptrType, i64Type}, i32Type);
      if (failed(memcpyFn))
        return failure();

      LLVM::CallOp::create(
          rewriter, loc, *memcpyFn,
          ValueRange{statePtr, dstPtr, srcPtr, i64Const(totalBytes)});
      rewriter.eraseOp(op);
      return success();
    }

    // Pitched 2D copy: last dim contiguous (stride 1) on BOTH sides, and a
    // common contiguous suffix exists (computed below as splitDim). Either
    // side may be a strided subview into a parent allocation -- the canonical
    // case is Concat / insert_slice, which bufferizes to a memref.subview on
    // the destination side: the slice's outer strides inherit the parent's
    // (larger) pitch, while the inner suffix [axis..rank-1] is still
    // contiguous. The splitDim finder + outer-collapse loop below cover both
    // src-strided and dst-strided shapes, so we do NOT require either side to
    // be dense row-major here.
    //
    // Before (a Concat axis=1 of two halves bufferizes to a strided-dst copy):
    //   %parent = memref.alloc() : memref<1x256x64x64xf16>
    //   %slice  = memref.subview %parent[0,0,0,0] [1,128,64,64] [1,1,1,1]
    //       : memref<1x256x64x64xf16>
    //         to memref<1x128x64x64xf16, strided<[1048576,4096,64,1]>>
    //   memref.copy %in, %slice
    //       : memref<1x128x64x64xf16>
    //         to memref<1x128x64x64xf16, strided<[1048576,4096,64,1]>>
    //   // dst is NOT dense row-major (outer stride 1048576 keeps the parent
    //   // pitch, dense would be 524288) -> old code rejected this copy.
    //
    // After (splitDim=1: row = contiguous suffix [1..3] = 128*64*64 elems,
    // height = dim 0 = 1, dst pitch = dstStrides[0] = 1048576 elems):
    //   llvm.call @wrap_hipMemcpy2DAsync(%state, %dstPtr, %dstPitchBytes,
    //       %srcPtr, %srcPitchBytes, %widthBytes, %height) : ...
    if (rank < 2)
      return rewriter.notifyMatchFailure(
          op, "rank-1 non-uniform stride needs different lowering");

    if (srcStrides[static_cast<unsigned>(rank - 1)] != 1 ||
        dstStrides[static_cast<unsigned>(rank - 1)] != 1)
      return rewriter.notifyMatchFailure(op,
                                         "last dimension must be contiguous");

    // To express this copy as a single hipMemcpy2DAsync we need to split the
    // dims into two groups:
    //   * a "width" group: a contiguous suffix [splitDim..rank-1] whose total
    //     size equals one row, requires
    //         stride[i] == stride[i+1] * shape[i+1]   for i in
    //         [splitDim..rank-2)
    //     (and stride[rank-1] == 1, already verified above);
    //   * a "height" group: the prefix [0..splitDim) whose dim sizes multiply
    //     into `height`. Because hipMemcpy2DAsync only takes ONE srcPitch /
    //     dstPitch, these prefix dims must themselves be collapsible, i.e.
    //         stride[i] == stride[i+1] * shape[i+1]   for i in [0..splitDim-1).
    // If neither condition holds we cannot lower with a single 2D pitched
    // copy and must bail out (a future ND/loop lowering can pick this up).
    for (int64_t i = 0; i < rank; ++i) {
      if (ShapedType::isDynamic(shape[static_cast<unsigned>(i)]))
        return rewriter.notifyMatchFailure(op, "need static shape");
    }

    // Find the longest contiguous suffix that holds on BOTH src and dst.
    int64_t splitDim = rank - 1;
    for (int64_t i = rank - 2; i >= 0; --i) {
      int64_t expectSrc = srcStrides[static_cast<unsigned>(i + 1)] *
                          shape[static_cast<unsigned>(i + 1)];
      int64_t expectDst = dstStrides[static_cast<unsigned>(i + 1)] *
                          shape[static_cast<unsigned>(i + 1)];
      if (srcStrides[static_cast<unsigned>(i)] != expectSrc ||
          dstStrides[static_cast<unsigned>(i)] != expectDst)
        break;
      splitDim = i;
    }

    // splitDim == 0 means the whole thing is dense; that case is already
    // handled by the d2d memcpy path above, so we should never reach here.
    if (splitDim == 0)
      return rewriter.notifyMatchFailure(
          op, "fully dense copy should use plain memcpy");

    // Outer dims [0..splitDim) must also collapse into a single height pitch.
    for (int64_t i = 0; i + 1 < splitDim; ++i) {
      int64_t expectSrc = srcStrides[static_cast<unsigned>(i + 1)] *
                          shape[static_cast<unsigned>(i + 1)];
      int64_t expectDst = dstStrides[static_cast<unsigned>(i + 1)] *
                          shape[static_cast<unsigned>(i + 1)];
      if (srcStrides[static_cast<unsigned>(i)] != expectSrc ||
          dstStrides[static_cast<unsigned>(i)] != expectDst)
        return rewriter.notifyMatchFailure(
            op, "non-collapsible outer strides; needs ND copy lowering");
    }

    int64_t widthElems = 1;
    for (int64_t i = splitDim; i < rank; ++i)
      widthElems *= shape[static_cast<unsigned>(i)];
    int64_t widthBytes = widthElems * elemBytes;

    int64_t height = 1;
    for (int64_t i = 0; i < splitDim; ++i)
      height *= shape[static_cast<unsigned>(i)];

    int64_t srcPitchElems = srcStrides[static_cast<unsigned>(splitDim - 1)];
    int64_t dstPitchElems = dstStrides[static_cast<unsigned>(splitDim - 1)];
    int64_t srcPitchBytes = srcPitchElems * elemBytes;
    int64_t dstPitchBytes = dstPitchElems * elemBytes;

    // Degenerate pitched-2D: very thin rows (few bytes) over many rows make
    // hipMemcpy2DAsync pathological -- the copy engine processes each row as a
    // separate tiny transfer, so the copy serializes into `height`
    // micro-transfers. The canonical trigger is an interleave (e.g. sinusoidal
    // position embedding's Concat(unsqueeze(sin), unsqueeze(cos), axis=last)),
    // which bufferizes to a strided-dst copy with widthElems=1 and a very large
    // height. A single parallel strided-copy kernel launch (one thread per
    // element) is far faster. Wide rows keep the DMA path, where
    // hipMemcpy2DAsync issues efficient contiguous bursts.
    //
    // Before:
    //   memref.copy %sin, %dst_even
    //       : memref<...x64x1xf16>
    //         to memref<...x64x1xf16, strided<[...,2,1]>>
    //   // splitDim picks widthElems=1, height=40000, dstPitch=2 elems
    //   //   -> wrap_hipMemcpy2DAsync(width=2B, height=40000)  (slow)
    // After:
    //   wrap_strided_copy(%state, %dst, %src, elem=2, height=40000,
    //                     srcPitch=1, dstPitch=2, row=1)       (one kernel)
    constexpr int64_t kThinRowBytesMax = 256;
    constexpr int64_t kManyRowsMin = 256;
    if (widthBytes <= kThinRowBytesMax && height >= kManyRowsMin) {
      FailureOr<LLVM::LLVMFuncOp> stridedFn =
          LLVM::lookupOrCreateFn(rewriter, module, kWrapStridedCopy,
                                 {ptrType, ptrType, ptrType, i64Type, i64Type,
                                  i64Type, i64Type, i64Type},
                                 i32Type);
      if (failed(stridedFn))
        return failure();

      LLVM::CallOp::create(
          rewriter, loc, *stridedFn,
          ValueRange{statePtr, dstPtr, srcPtr, i64Const(elemBytes),
                     i64Const(height), i64Const(srcPitchElems),
                     i64Const(dstPitchElems), i64Const(widthElems)});
      rewriter.eraseOp(op);
      return success();
    }

    FailureOr<LLVM::LLVMFuncOp> memcpy2dFn = LLVM::lookupOrCreateFn(
        rewriter, module, kWrapHipMemcpy2DAsync,
        {ptrType, ptrType, i64Type, ptrType, i64Type, i64Type, i64Type},
        i32Type);
    if (failed(memcpy2dFn))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *memcpy2dFn,
                         ValueRange{statePtr, dstPtr, i64Const(dstPitchBytes),
                                    srcPtr, i64Const(srcPitchBytes),
                                    i64Const(widthBytes), i64Const(height)});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateMemoryLoweringPatterns(const LLVMTypeConverter &converter,
                                    RewritePatternSet &patterns) {
  patterns.add<AllocOpLowering, FreeOpLowering, GetPoolOpLowering,
               GetHostScratchOpLowering, AllocOutputOpLowering,
               GetConstantOpLowering, MemRefAllocOpLowering,
               MemRefDeallocOpLowering>(converter);
  patterns.add<MemRefCopyOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
