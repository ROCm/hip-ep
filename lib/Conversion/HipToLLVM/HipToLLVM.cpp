/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/IR/HipDialect.h"
#include "hip/Dialect/Transforms/Passes.h"
#include "mlir/Conversion/LLVMCommon/ConversionTarget.h"
#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTHIPTOLLVMPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

static constexpr const char *kHipCreateHandle = "hipCreateHandle";
static constexpr const char *kHipDestroyHandle = "hipDestroyHandle";
static constexpr const char *kHipMalloc = "hip_device_malloc";
static constexpr const char *kHipFree = "hip_device_free";

static constexpr const char *kHipblasltMatmul = "hip_hipblaslt_matmul";
static constexpr const char *kMiopenRmsNorm = "hip_miopen_rms_norm";
static constexpr const char *kMiopenSkipRmsNorm = "hip_miopen_skip_rms_norm";
static constexpr const char *kMiopenRope = "hip_miopen_rope";
static constexpr const char *kMiopenAdd = "hip_miopen_add";
static constexpr const char *kMiopenMul = "hip_miopen_mul";
static constexpr const char *kMiopenSoftmax = "hip_miopen_softmax";
static constexpr const char *kHipTranspose = "hip_transpose";
static constexpr const char *kHipGather = "hip_gather";
static constexpr const char *kHipSilu = "hip_silu";
static constexpr const char *kHipGqa = "hip_gqa";

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

// --- CreateHandleOp: hip.create_handle() -> llvm.call @hipCreateHandle()
struct CreateHandleOpLowering : public ConvertOpToLLVMPattern<CreateHandleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(CreateHandleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipCreateHandle, /*paramTypes=*/{}, ptrType);
    if (failed(funcOp))
      return failure();

    auto callOp = LLVM::CallOp::create(rewriter, loc, *funcOp, ValueRange());
    rewriter.replaceOp(op, callOp.getResult());
    return success();
  }
};

// --- DestroyHandleOp: hip.destroy_handle(%h) -> llvm.call
// @hipDestroyHandle(%h)
struct DestroyHandleOpLowering
    : public ConvertOpToLLVMPattern<DestroyHandleOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(DestroyHandleOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipDestroyHandle, ptrType, voidType);
    if (failed(funcOp))
      return failure();

    LLVM::CallOp::create(rewriter, loc, *funcOp, adaptor.getHandle());
    rewriter.eraseOp(op);
    return success();
  }
};

// --- AllocOp: hip.alloc(%handle, %dyn...) -> hipMalloc(bytes) + memref
// descriptor
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

    // Compute sizes and sizeBytes (dynamic sizes are after the handle).
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

// --- FreeOp: hip.free(%handle, %memref) -> llvm.call @hipFree(allocated_ptr)
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

    SmallVector<Value> args = {adaptor.getHandle(),
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
struct MiopenRmsNormOpLowering
    : public ConvertOpToLLVMPattern<MiopenRmsNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MiopenRmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    SmallVector<Type> paramTypes = {ptrType, ptrType,   ptrType,
                                    ptrType, indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenRmsNorm, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    int rank = cast<MemRefType>(op.getInput().getType()).getRank();
    MemRefDescriptor inputDesc(adaptor.getInput());

    // D = last dim; N = product of all other dims
    Value D = inputDesc.size(rewriter, loc, rank - 1);
    Value N = inputDesc.size(rewriter, loc, 0);
    for (int i = 1; i < rank - 1; i++)
      N = LLVM::MulOp::create(rewriter, loc, N,
                              inputDesc.size(rewriter, loc, i));

    SmallVector<Value> args = {
        adaptor.getHandle(),
        extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getWeight(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        N,
        D};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.skip_rms_norm(handle, x, skip, weight, output, residual)
struct MiopenSkipRmsNormOpLowering
    : public ConvertOpToLLVMPattern<MiopenSkipRmsNormOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MiopenSkipRmsNormOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    SmallVector<Type> paramTypes(6, ptrType);
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenSkipRmsNorm, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getHandle(),
        extractMemRefPtr(adaptor.getX(), rewriter, loc),
        extractMemRefPtr(adaptor.getSkip(), rewriter, loc),
        extractMemRefPtr(adaptor.getWeight(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc),
        extractMemRefPtr(adaptor.getResidual(), rewriter, loc)};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.rope(handle, q, k, cos_cache, sin_cache, start_pos)
struct MiopenRopeOpLowering : public ConvertOpToLLVMPattern<MiopenRopeOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(MiopenRopeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    SmallVector<Type> paramTypes = {ptrType, ptrType, ptrType,
                                    ptrType, ptrType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kMiopenRope, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    SmallVector<Value> args = {
        adaptor.getHandle(),
        extractMemRefPtr(adaptor.getQ(), rewriter, loc),
        extractMemRefPtr(adaptor.getK(), rewriter, loc),
        extractMemRefPtr(adaptor.getCosCache(), rewriter, loc),
        extractMemRefPtr(adaptor.getSinCache(), rewriter, loc),
        adaptor.getStartPos()};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// hip.miopen.add / hip.miopen.mul  (element-wise binary ops)
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
    for (int i = 0; i < rank; i++)
      num = LLVM::MulOp::create(
          rewriter, loc, num,
          MemRefDescriptor(descriptor).size(rewriter, loc, i));
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

    SmallVector<Value> args = {adaptor.getHandle(),
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
        adaptor.getHandle(),
        extractMemRefPtr(adaptor.getInput(), rewriter, loc),
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
    MemRefDescriptor inputDesc(adaptor.getInput());
    Value rankVal = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                             rewriter.getIndexAttr(rank));
    Value one = LLVM::ConstantOp::create(rewriter, loc, indexType,
                                         rewriter.getIndexAttr(1));

    SmallVector<Value, 3> shape;
    for (int i = 0; i < 3; i++)
      shape.push_back(i < rank ? inputDesc.size(rewriter, loc, i) : one);

    SmallVector<Value> args = {
        adaptor.getHandle(),
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
        adaptor.getHandle(),
        extractMemRefPtr(adaptor.getIndices(), rewriter, loc),
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
        adaptor.getHandle(),
        extractMemRefPtr(adaptor.getInput(), rewriter, loc),
        extractMemRefPtr(adaptor.getOutput(), rewriter, loc)};

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
        adaptor.getHandle(),
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

  // !hip.handle -> !llvm.ptr (opaque pointer to runtime context)
  typeConverter.addConversion([ctx](HandleType type) -> Type {
    return LLVM::LLVMPointerType::get(ctx, 0);
  });

  RewritePatternSet patterns(ctx);
  patterns.add<CreateHandleOpLowering, DestroyHandleOpLowering, AllocOpLowering,
               FreeOpLowering, MiopenGraphOpLowering, HipblasltGraphOpLowering,
               HipblasltMatmulOpLowering, MiopenRmsNormOpLowering,
               MiopenSkipRmsNormOpLowering, MiopenRopeOpLowering,
               MiopenSoftmaxOpLowering, TransposeOpLowering, GatherOpLowering,
               SiluOpLowering, GqaOpLowering>(typeConverter);
  patterns.insert<MiopenBinaryOpLowering<MiopenAddOp>>(typeConverter,
                                                       kMiopenAdd);
  patterns.insert<MiopenBinaryOpLowering<MiopenMulOp>>(typeConverter,
                                                       kMiopenMul);
  patterns.add<MemRefAllocOpLowering, MemRefDeallocOpLowering>(typeConverter);

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
