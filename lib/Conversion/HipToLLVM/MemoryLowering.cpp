/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"

namespace mlir {
namespace hip {
namespace {

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
// point where the output shape is known. The shape is handed to the runtime as
// a stack-allocated i64[rank] array (static dims become constants, dynamic dims
// come from the op's operands, in type order); the runtime returns the device
// pointer, which the lowering wraps in a standard memref descriptor with
// row-major strides. Unlike AllocOp this issues no hipMalloc/hipFree -- the
// buffer is EP-owned (a graph output), matching AllocOutputOp::getEffects
// (no Allocate effect). Works uniformly for static, dynamic, and mixed shapes
// because getMemRefDescriptorSizes interleaves type constants with operands.
//
// Before:
//   %out = hip.alloc_output(%ctx, %M, %N) {out_idx = 0 : i64} : memref<?x?xf16>
//
// After:
//   %shape = llvm.alloca %c1 x !llvm.array<2 x i64>
//   llvm.store %M, %shape[0]            // gep elem i64, index 0
//   llvm.store %N, %shape[1]            // gep elem i64, index 1
//   %p = llvm.call @hipdnn_ep_alloc_output(%state, %c0_outidx, %shape,
//                                          %c2_rank, %c2_elem)
//          : (!llvm.ptr, i64, !llvm.ptr, i64, i64) -> !llvm.ptr
//   // descriptor { alloc=%p, aligned=%p, offset=0, sizes=[%M,%N],
//   //              strides=[%N,1] }
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
    int64_t elemSizeBytes = elemType.getIntOrFloatBitWidth() / 8;
    if (elemSizeBytes <= 0)
      return rewriter.notifyMatchFailure(op, "unsupported element bit width");

    int64_t rank = memRefType.getRank();

    // sizes[] interleaves static dims (type constants) with the dynamic-size
    // operands (in type order); strides[] are row-major. Same helper AllocOp
    // uses, so static / dynamic / mixed shapes all flow through one path. The
    // returned sizeBytes is unused here (the runtime computes bytes itself).
    SmallVector<Value, 4> sizes;
    SmallVector<Value, 4> strides;
    Value sizeBytes;
    getMemRefDescriptorSizes(loc, memRefType, adaptor.getDynamicSizes(),
                             rewriter, sizes, strides, sizeBytes, true);

    // Stack-allocate the i64[rank] shape array and populate it from sizes[].
    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                            rewriter.getI64IntegerAttr(1));
    auto shapeArrayType =
        LLVM::LLVMArrayType::get(i64Type, rank > 0 ? rank : 1);
    Value shapeAlloca = LLVM::AllocaOp::create(
        rewriter, loc, ptrType, shapeArrayType, oneI64, /*alignment=*/8);
    for (int64_t i : llvm::seq<int64_t>(0, rank)) {
      Value idx = LLVM::ConstantOp::create(rewriter, loc, i32Type,
                                           rewriter.getI32IntegerAttr(i));
      Value gep = LLVM::GEPOp::create(rewriter, loc, ptrType, i64Type,
                                      shapeAlloca, idx);
      LLVM::StoreOp::create(rewriter, loc, sizes[i], gep);
    }

    Value outIdxVal = LLVM::ConstantOp::create(
        rewriter, loc, i64Type, rewriter.getI64IntegerAttr(op.getOutIdx()));
    Value rankVal = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(rank));
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
        loc, memRefType, dataPtr, dataPtr, sizes, strides, rewriter);
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
