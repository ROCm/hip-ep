//===- HipToLLVM.cpp - HIP to LLVM dialect conversion ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "HipDialect.h"
#include "HipPasses.h"
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

namespace {

static constexpr const char *kHipCreateHandle = "hipCreateHandle";
static constexpr const char *kHipDestroyHandle = "hipDestroyHandle";
static constexpr const char *kHipMalloc = "hipMalloc";
static constexpr const char *kHipFree = "hipFree";
static constexpr const char *kHipGemmF32 = "hip_gemm_f32";

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

// --- DestroyHandleOp: hip.destroy_handle(%h) -> llvm.call @hipDestroyHandle(%h)
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

// --- AllocOp: hip.alloc(%handle, %dyn...) -> hipMalloc(bytes) + memref descriptor
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
      return rewriter.notifyMatchFailure(op, "could not compute element ptr type");
    FailureOr<unsigned> addrSpace = getTypeConverter()->getMemRefAddressSpace(memRefType);
    if (failed(addrSpace))
      return failure();
    if (cast<LLVM::LLVMPointerType>(allocatedPtr.getType()).getAddressSpace() != *addrSpace)
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
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();

    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipFree, ptrType, voidType);
    if (failed(funcOp))
      return failure();

    Value memrefDesc = adaptor.getMemref();
    Value allocatedPtr =
        MemRefDescriptor(memrefDesc).allocatedPtr(rewriter, loc);
    // hipFree expects void*; if memref is in non-default address space, cast
    auto ptrTy = allocatedPtr.getType();
    if (cast<LLVM::LLVMPointerType>(ptrTy).getAddressSpace() != 0)
      allocatedPtr = rewriter.create<LLVM::AddrSpaceCastOp>(loc, ptrType, allocatedPtr);

    LLVM::CallOp::create(rewriter, loc, *funcOp, allocatedPtr);
    rewriter.eraseOp(op);
    return success();
  }
};

// --- GemmOp: hip.gemm(%handle, %A, %B, %C, %M, %K, %N)
//            -> llvm.call @hip_gemm_f32(%A, %B, %C, %M, %K, %N)
struct GemmOpLowering : public ConvertOpToLLVMPattern<GemmOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GemmOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type voidType = getVoidType();
    Type ptrType = getPtrType();
    Type indexType = getIndexType();

    // Declare hip_gemm_f32(A: ptr, B: ptr, C: ptr, M: i64, K: i64, N: i64)
    SmallVector<Type, 6> paramTypes = {ptrType, ptrType, ptrType,
                                       indexType, indexType, indexType};
    FailureOr<LLVM::LLVMFuncOp> funcOp = LLVM::lookupOrCreateFn(
        rewriter, module, kHipGemmF32, paramTypes, voidType);
    if (failed(funcOp))
      return failure();

    // The pointer args (A, B, C) are already !llvm.ptr after type conversion.
    // The dimension args (M, K, N) are index -> i64 after type conversion.
    SmallVector<Value, 6> args = {adaptor.getA(), adaptor.getB(),
                                  adaptor.getC(), adaptor.getM(),
                                  adaptor.getK(), adaptor.getN()};

    LLVM::CallOp::create(rewriter, loc, *funcOp, args);
    rewriter.eraseOp(op);
    return success();
  }
};

// --- Pass
struct ConvertHipToLLVMPass
    : public PassWrapper<ConvertHipToLLVMPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertHipToLLVMPass)

  StringRef getArgument() const final { return "convert-hip-to-llvm"; }
  StringRef getDescription() const final {
    return "Convert HIP dialect to LLVM dialect";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<LLVM::LLVMDialect>();
    registry.insert<memref::MemRefDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    MLIRContext *ctx = module.getContext();

    LowerToLLVMOptions options(ctx);
    LLVMTypeConverter typeConverter(ctx, options);

    // Convert !hip.handle to !llvm.ptr
    typeConverter.addConversion(
        [ctx](HandleType type) -> Type {
          return LLVM::LLVMPointerType::get(ctx, 0);
        });

    RewritePatternSet patterns(ctx);
    patterns.add<CreateHandleOpLowering, DestroyHandleOpLowering,
                 AllocOpLowering, FreeOpLowering,
                 GemmOpLowering>(typeConverter);

    LLVMConversionTarget target(*ctx);
    target.addLegalDialect<LLVM::LLVMDialect>();
    target.addIllegalDialect<HipDialect>();
    target.addLegalOp<ModuleOp>();

    if (failed(applyPartialConversion(module, target, std::move(patterns))))
      signalPassFailure();
  }
};

} // namespace

std::unique_ptr<Pass> createConvertHipToLLVMPass() {
  return std::make_unique<ConvertHipToLLVMPass>();
}

void registerHipPasses() {
  PassRegistration<ConvertHipToLLVMPass>();
}

} // namespace hip
} // namespace mlir
