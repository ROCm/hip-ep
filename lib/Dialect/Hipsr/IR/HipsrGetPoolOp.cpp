/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Conversion/HipsrToLLVM/HipsrToLLVM.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/Conversion/LLVMCommon/Pattern.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/LLVMIR/FunctionCallUtils.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;
using namespace mlir::hipsr;

void GetPoolOp::getEffects(
    SmallVectorImpl<SideEffects::EffectInstance<MemoryEffects::Effect>>
        &effects) {
  effects.emplace_back(MemoryEffects::Allocate::get(),
                       getOperation()->getResult(0),
                       SideEffects::DefaultResource::get());
}

namespace {

constexpr const char *kGetPoolBase = "hipdnn_ep_get_pool_base";

struct GetPoolLowering : public ConvertOpToLLVMPattern<GetPoolOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(GetPoolOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type hostPtrType = LLVM::LLVMPointerType::get(ctx, 0);
    Type devicePtrType = LLVM::LLVMPointerType::get(ctx, 1);
    Type i32Type = rewriter.getI32Type();
    Type i64Type = rewriter.getI64Type();
    MemRefType memRefType = cast<MemRefType>(op.getPool().getType());

    FailureOr<LLVM::LLVMFuncOp> funcOp =
        LLVM::lookupOrCreateFn(rewriter, module, kGetPoolBase,
                               {hostPtrType, i32Type, i64Type}, devicePtrType);
    if (failed(funcOp)) {
      return failure();
    }

    Value poolSize = adaptor.getPoolSize();
    Value domainIdVal = LLVM::ConstantOp::create(
        rewriter, loc, i32Type,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(op.getDomainId())));
    Value gpuPtr = LLVM::CallOp::create(
                       rewriter, loc, *funcOp,
                       ValueRange{adaptor.getCtx(), domainIdVal, poolSize})
                       .getResult();

    Value stride1 = LLVM::ConstantOp::create(rewriter, loc, i64Type,
                                             rewriter.getI64IntegerAttr(1));

    MemRefDescriptor desc = createMemRefDescriptor(
        loc, memRefType, gpuPtr, gpuPtr, {poolSize}, {stride1}, rewriter);
    rewriter.replaceOp(op, {desc});
    return success();
  }
};

} // namespace

void mlir::hipsr::populateHipsrGetPoolLoweringPatterns(
    const LLVMTypeConverter &converter, RewritePatternSet &patterns) {
  patterns.add<GetPoolLowering>(converter);
}
