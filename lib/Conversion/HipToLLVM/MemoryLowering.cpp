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

// --- HostSyncOp: hip.host_sync(%ctx) -> llvm.call
// @hipdnn_ep_stream_sync(state). Issued by `hip-materialize-host-scalars` ahead
// of a host `memref.load` whose source memref aliases the runtime host-scratch
// and whose preceding writer was a `hip.*` GPU op.  Without the sync the host
// load races the GPU write on `hipHostMallocMapped` memory.
struct HostSyncOpLowering : public ConvertOpToLLVMPattern<HostSyncOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(HostSyncOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i32Type = rewriter.getI32Type();

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipdnnEpStreamSync, {ptrType}, i32Type);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp, ValueRange{adaptor.getCtx()});
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

/// Returns true when the type's layout guarantees the last dimension has
/// stride 1 (contiguous last axis).  Both identity layout and strided layout
/// with an explicit innermost stride of 1 qualify.
static bool isLastDimStrideOne(MemRefType ty) {
  if (ty.getRank() == 0)
    return true;
  MemRefLayoutAttrInterface layout = ty.getLayout();
  if (layout.isIdentity())
    return true;
  if (auto sl = dyn_cast<StridedLayoutAttr>(layout)) {
    if (sl.getStrides().empty())
      return false;
    return sl.getStrides().back() == 1;
  }
  return false;
}

/// Returns true when the destination is dense row-major identity layout — the
/// canonical shape produced by `memref.alloc` (and by --hip-promote-strided-
/// operands' temporary alloc) for a freshly materialised buffer.  Identity
/// layout (no explicit layout attr) trivially qualifies; an explicit
/// strided<[..., 1]> layout qualifies only when all strides are static AND
/// match the row-major prefix-product pattern.
static bool isIdentityRowMajor(MemRefType ty) {
  MemRefLayoutAttrInterface layout = ty.getLayout();
  if (layout.isIdentity())
    return true;
  auto sl = dyn_cast<StridedLayoutAttr>(layout);
  if (!sl)
    return false;
  ArrayRef<int64_t> strides = sl.getStrides();
  ArrayRef<int64_t> shape = ty.getShape();
  if (strides.size() != shape.size())
    return false;
  if (sl.getOffset() != 0)
    return false;
  int64_t running = 1;
  for (int i = static_cast<int>(shape.size()) - 1; i >= 0; --i) {
    if (strides[static_cast<unsigned>(i)] != running)
      return false;
    if (ShapedType::isDynamic(shape[static_cast<unsigned>(i)]))
      return false;
    running *= shape[static_cast<unsigned>(i)];
  }
  return true;
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
/// wrap_hipMemcpy2DAsync.
///
/// Three regimes, in order:
///   1. Static-stride dense row-major (src ≡ dst, no holes): single
///      wrap_hipMemcpyAsync of total bytes.
///   2. Static-stride pitched (src has holes, dst dense row-major):
///      wrap_hipMemcpy2DAsync after collapsing fully-static dims into a
///      width / height pair.
///   3. Dynamic-stride / dynamic-shape (typical promote-strided output:
///      src has dynamic outer strides from a subview, dst is identity
///      row-major dyn-shape alloc): wrap_hipMemcpyAsync (rank ≤ 1) or
///      wrap_hipMemcpy2DAsync with runtime-computed sizes / src-pitch.
///
/// Regimes 2 and 3 require the source's outer dims to collapse uniformly into
/// a single height pitch (stride[i] == stride[i+1] * shape[i+1]).  This holds
/// for a subview of a contiguous parent that keeps all parent dims, but NOT
/// for a subview that drops an outer dim (e.g. selecting one component of an
/// unbind / Split on a packed [N, C, H, W] tensor): there stride[0] stays at
/// the parent row stride while stride[1]*shape[1] is the smaller post-collapse
/// extent.  Both regimes therefore verify the collapse where it is statically
/// provable and bail (notifyMatchFailure) when it provably fails.
///
/// Cases that bail here -- including the dropped-outer-dim copy above and
/// arbitrary non-row-major / strided destinations -- fall through to the
/// standard MemRefToLLVM `CopyOpLowering` (registered at lower benefit by
/// populateFinalizeMemRefToLLVMConversionPatterns), which emits the runtime
/// `memrefCopy` libcall.  That helper (lib/Runtime/real/hip.cpp) walks the
/// outer index space one contiguous row at a time and honours arbitrary
/// rank-N strides, so it is the correct general fallback.  (The 2D fast paths
/// here exist only to avoid a per-row launch when the copy IS a single pitched
/// blit.)
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

    // ---------------------------------------------------------------------
    // Dynamic-stride / dynamic-shape path.
    //
    // Triggered when either operand's layout has dynamic strides at type
    // level (e.g. `memref<?x?x16x256xf16, strided<[?, ?, 256, 1], offset: ?>>`
    // produced by --hip-promote-strided-operands wrapping a memref.subview of
    // a contiguous parent into a fresh alloc).
    //
    // Strategy: emit ONE runtime call (`wrap_hipMemcpyAsync` for rank-1 or
    // fully-degenerate, `wrap_hipMemcpy2DAsync` otherwise) using sizes and
    // src-pitch read from the converted memref descriptors.  Dst is required
    // to be identity row-major (the canonical alloc shape) so dst-pitch is
    // simply the product of inner sizes.
    //
    // To express the copy as a single 2D pitched memcpy we split the dims
    // into two groups (mirrors the static-path algorithm below):
    //   * a "width" group: contiguous suffix [splitDim..rank-1] whose dims
    //     are guaranteed to multiply tightly because
    //       stride[i] == stride[i+1] * shape[i+1]
    //     holds for all i in [splitDim..rank-2) at TYPE level (i.e. the
    //     equality is statically provable -- which requires all three
    //     terms to be static integers);
    //   * a "height" group: prefix [0..splitDim), collapsed at runtime into
    //     a single product.
    //
    // Dynamic strides act as a HARD BARRIER for the suffix check: we cannot
    // verify the equality at compile time, so we stop extending the suffix
    // there.  Hard-coding splitDim = rank-1 (the previous behaviour) silently
    // misreads the source whenever an inner dim is non-contiguously sized --
    // e.g. when subview-ing a tensor that already has a stride hole between
    // its inner dims (transformer QKV split into [B,S,H,D] from a packed
    // [B,S,3*H*D] is a typical producer).
    //
    // Runtime invariant assumed (NOT verified at compile time): the prefix
    // dims [0..splitDim) collapse uniformly into one height pitch, i.e.
    //   src.stride[i] == src.stride[i+1] * src.size[i+1]  for i in
    //   [0..splitDim-1)
    // This holds for any subview of a contiguous parent, which is the only
    // producer of these copies in the current pipeline.
    //
    // Before:
    //   memref.copy %sv, %tmp
    //     : memref<?x?x16x128xf16, strided<[?, 8192, 128, 1], offset: ?>>
    //       to memref<?x?x16x128xf16>
    //   ; sv comes from a subview of a packed [B,S,8192] (= QKV concatenated
    //   ; into 16*128 = 2048 q + 2*16*128 = 4096 kv + ... layout).  Between
    //   ; each (b,s) row the source has a 8192-2048 = 6144-element hole.
    // After (with the fix, splitDim = 2):
    //   %height = mul %size0, %size1               ; B * S
    //   %width  = const(16 * 128 * 2)              ; H * D * elem_bytes
    //   %src_pitch = mul %src.stride[1], elem_bytes ; runtime = 8192 * 2
    //   call @wrap_hipMemcpy2DAsync(state, dst, %width, src, %src_pitch,
    //                                %width, %height)
    // Without the fix (splitDim = rank-1 = 3) the inner row would be
    // `width = const(128 * 2)` and `src_pitch = src.stride[2] * 2 = 128 * 2`,
    // i.e. equal -- the kernel would silently copy the whole [B*S*16*128]
    // contiguously starting at `src` and step right through the QKV hole.
    // ---------------------------------------------------------------------
    if (failed(srcStridesOr) || failed(dstStridesOr)) {
      if (!isLastDimStrideOne(srcTy) || !isLastDimStrideOne(dstTy))
        return rewriter.notifyMatchFailure(
            op, "dynamic-stride copy requires last dim contiguous on both");
      if (!isIdentityRowMajor(dstTy))
        return rewriter.notifyMatchFailure(
            op, "dynamic-stride copy requires identity row-major destination");

      int64_t rank = srcTy.getRank();
      ArrayRef<int64_t> shape = srcTy.getShape();
      Value elemBytesVal = LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(elemBytes));

      MemRefDescriptor srcDesc(adaptor.getSource());
      // Destination is identity row-major; its runtime sizes/strides are
      // recoverable from the static shape + computed pitch, so we never need
      // to query its descriptor here.

      auto runtimeDim = [&](MemRefDescriptor &desc, unsigned i) -> Value {
        if (ShapedType::isDynamic(shape[i]))
          return desc.size(rewriter, loc, i);
        return LLVM::ConstantOp::create(
            rewriter, loc, i64Type,
            rewriter.getI64IntegerAttr(shape[static_cast<unsigned>(i)]));
      };

      // Rank-0 / rank-1: degenerate to a single 1D D2D memcpy. All elements
      // are contiguous on src (last stride 1) and on dst (identity).
      if (rank <= 1) {
        Value nElems = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                                rewriter.getI64IntegerAttr(1));
        if (rank == 1)
          nElems = runtimeDim(srcDesc, 0);
        Value totalBytes =
            LLVM::MulOp::create(rewriter, loc, nElems, elemBytesVal);

        FailureOr<LLVM::LLVMFuncOp> memcpyFn = LLVM::lookupOrCreateFn(
            rewriter, module, kWrapHipMemcpyAsync,
            {ptrType, ptrType, ptrType, i64Type}, i32Type);
        if (failed(memcpyFn))
          return failure();
        LLVM::CallOp::create(rewriter, loc, *memcpyFn,
                             ValueRange{statePtr, dstPtr, srcPtr, totalBytes});
        rewriter.eraseOp(op);
        return success();
      }

      // Pull type-level strides (statically known values + dynamic sentinels).
      SmallVector<int64_t> typeStrides;
      int64_t typeOffset = 0;
      if (failed(srcTy.getStridesAndOffset(typeStrides, typeOffset)))
        return rewriter.notifyMatchFailure(
            op, "source has no stride layout (cannot derive pitch)");

      // Find the longest contiguous inner suffix [splitDim..rank-1] using
      // type-level strides.  A dim is included iff
      //   stride[i] == stride[i+1] * shape[i+1]
      // is statically provable.  Dynamic strides / dynamic shapes BREAK the
      // walk -- we cannot verify the equality, so we conservatively bail at
      // that boundary.  The walk starts from the last dim (already verified
      // by isLastDimStrideOne to have stride 1).
      int64_t splitDim = rank - 1;
      for (int64_t i = rank - 2; i >= 0; --i) {
        if (ShapedType::isDynamic(typeStrides[i]) ||
            ShapedType::isDynamic(typeStrides[i + 1]) ||
            ShapedType::isDynamic(shape[i + 1]))
          break;
        if (typeStrides[i] != typeStrides[i + 1] * shape[i + 1])
          break;
        splitDim = i;
      }

      // hipMemcpy2DAsync exposes only ONE source pitch, so the height-group
      // prefix dims [0..splitDim) must collapse UNIFORMLY into that single
      // pitch, i.e. stride[i] == stride[i+1] * shape[i+1] for every
      // i in [0..splitDim-1).  For a subview of a contiguous parent this holds
      // (sometimes only at runtime, when prefix shapes are dynamic -- the QKV
      // [B,S,H,D]-from-packed-[B,S,3HD] producer relies on that runtime
      // equality, which we cannot disprove statically and therefore allow).
      // But a subview that DROPS an outer dim -- e.g. selecting one component
      // of an unbind / Split on a packed [N, C, H, W] tensor -- leaves
      // stride[0] = parent_row_stride while stride[1]*shape[1] is the smaller
      // post-collapse inner extent; the two differ by the dropped dim's factor.
      // A single 2D copy with pitch = stride[splitDim-1] would then walk every
      // height row at that one (too-small) pitch and silently read the wrong
      // source rows.  When we can STATICALLY PROVE the prefix does not collapse,
      // bail so the generic strided @memrefCopy libcall (registered by the
      // standard memref-to-llvm lowering at lower benefit) handles it correctly
      // by walking the outer index space one row at a time.
      //
      // Before:  memref.copy %sv, %tmp
      //   : memref<?x16x36xf16, strided<[3456, 72, 1], offset: ?>>
      //     to memref<?x16x36xf16>
      //   ; sv = Squeeze(Split(view_16=[N,3,16,72], axis=1)[0])[.,:,36:72].
      //   ; stride[0]=3456 (view_16 row stride) but stride[1]*shape[1]=72*16
      //   ; =1152 -> the prefix dim 0 does NOT collapse onto dim 1.
      // After:   (this pattern bails on the provable mismatch) the standard
      //          lowering emits @memrefCopy, which copies each 36-elem row at
      //          the correct outer stride 3456.
      for (int64_t i = 0; i + 1 < splitDim; ++i) {
        if (ShapedType::isDynamic(typeStrides[i]) ||
            ShapedType::isDynamic(typeStrides[i + 1]) ||
            ShapedType::isDynamic(shape[i + 1]))
          continue; // cannot disprove collapse; keep the runtime-correct 2D path
        if (typeStrides[i] != typeStrides[i + 1] * shape[i + 1])
          return rewriter.notifyMatchFailure(
              op, "non-collapsible outer strides; defer to generic memrefCopy");
      }

      // Width = product of sizes [splitDim..rank-1] * elem_bytes.  Each size
      // is either static (-> emitted as a constant) or dynamic (-> read from
      // the descriptor); `runtimeDim` handles both.
      Value widthElems = LLVM::ConstantOp::create(
          rewriter, loc, i64Type, rewriter.getI64IntegerAttr(1));
      for (int64_t i = splitDim; i < rank; ++i)
        widthElems =
            LLVM::MulOp::create(rewriter, loc, widthElems,
                                runtimeDim(srcDesc, static_cast<unsigned>(i)));
      Value widthBytes =
          LLVM::MulOp::create(rewriter, loc, widthElems, elemBytesVal);

      // splitDim == 0 means the entire memref is contiguous from `srcPtr`;
      // collapse to a flat memcpy.  src.stride[splitDim-1] is undefined here.
      if (splitDim == 0) {
        FailureOr<LLVM::LLVMFuncOp> memcpyFn = LLVM::lookupOrCreateFn(
            rewriter, module, kWrapHipMemcpyAsync,
            {ptrType, ptrType, ptrType, i64Type}, i32Type);
        if (failed(memcpyFn))
          return failure();
        LLVM::CallOp::create(rewriter, loc, *memcpyFn,
                             ValueRange{statePtr, dstPtr, srcPtr, widthBytes});
        rewriter.eraseOp(op);
        return success();
      }

      // Height = product of runtime sizes [0..splitDim).
      Value height = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                              rewriter.getI64IntegerAttr(1));
      for (int64_t i = 0; i < splitDim; ++i)
        height =
            LLVM::MulOp::create(rewriter, loc, height,
                                runtimeDim(srcDesc, static_cast<unsigned>(i)));

      // Src pitch = src.stride[splitDim-1] * elem_bytes (runtime -- the
      // descriptor holds the actual stride even when the type-level value
      // is dynamic).
      Value srcStrideOuter =
          srcDesc.stride(rewriter, loc, static_cast<unsigned>(splitDim - 1));
      Value srcPitchBytes =
          LLVM::MulOp::create(rewriter, loc, srcStrideOuter, elemBytesVal);

      // Dst pitch = inner row bytes (identity row-major dst, so adjacent
      // rows are tightly packed).
      Value dstPitchBytes = widthBytes;

      FailureOr<LLVM::LLVMFuncOp> memcpy2dFn = LLVM::lookupOrCreateFn(
          rewriter, module, kWrapHipMemcpy2DAsync,
          {ptrType, ptrType, i64Type, ptrType, i64Type, i64Type, i64Type},
          i32Type);
      if (failed(memcpy2dFn))
        return failure();

      LLVM::CallOp::create(rewriter, loc, *memcpy2dFn,
                           ValueRange{statePtr, dstPtr, dstPitchBytes, srcPtr,
                                      srcPitchBytes, widthBytes, height});
      rewriter.eraseOp(op);
      return success();
    }

    ArrayRef<int64_t> srcStrides = *srcStridesOr;
    ArrayRef<int64_t> dstStrides = *dstStridesOr;

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

    // Pitched 2D copy: last dim contiguous (stride 1), collapse leading dims.
    // Destination must be dense row-major (typical output buffer); source may
    // be a strided view into a parent allocation.
    if (rank < 2)
      return rewriter.notifyMatchFailure(
          op, "rank-1 non-uniform stride needs different lowering");

    if (!strideVectorsEqual(dstStrides, ArrayRef<int64_t>(*denseStridesOr)))
      return rewriter.notifyMatchFailure(
          op, "expect dense row-major destination for pitched copy");

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
  patterns
      .add<AllocOpLowering, FreeOpLowering, GetPoolOpLowering,
           GetHostScratchOpLowering, HostSyncOpLowering, GetConstantOpLowering,
           MemRefAllocOpLowering, MemRefDeallocOpLowering>(converter);
  patterns.add<MemRefCopyOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
