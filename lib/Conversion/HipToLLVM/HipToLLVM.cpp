/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"

namespace mlir {
namespace hip {

#define GEN_PASS_DEF_CONVERTHIPTOLLVMPASS
#include "hip/Dialect/Transforms/Passes.h.inc"

namespace {

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
  populateMemoryLoweringPatterns(typeConverter, patterns);
  populateConvLoweringPatterns(typeConverter, patterns);
  populateMatmulLoweringPatterns(typeConverter, patterns);
  populateElementwiseLoweringPatterns(typeConverter, patterns);
  populatePowerLoweringPatterns(typeConverter, patterns);
  populateActivationLoweringPatterns(typeConverter, patterns);
  populateNormLoweringPatterns(typeConverter, patterns);
  populateGatherLoweringPatterns(typeConverter, patterns);
  populateRangeLoweringPatterns(typeConverter, patterns);
  populateCastLoweringPatterns(typeConverter, patterns);
  populateReduceLoweringPatterns(typeConverter, patterns);
  populateTransposeLoweringPatterns(typeConverter, patterns);
  populateRopeLoweringPatterns(typeConverter, patterns);
  populateGqaLoweringPatterns(typeConverter, patterns);
  populateMultiHeadAttentionLoweringPatterns(typeConverter, patterns);
  populateMatMulNBitsLoweringPatterns(typeConverter, patterns);
  populateQMoELoweringPatterns(typeConverter, patterns);
  populateGemmLoweringPatterns(typeConverter, patterns);
  populateLinearAttentionLoweringPatterns(typeConverter, patterns);
  populateGraphLoweringPatterns(typeConverter, patterns);
  populateLoopLoweringPatterns(typeConverter, patterns);
  populateCausalConvWithStateLoweringPatterns(typeConverter, patterns);
  populateWhereLoweringPatterns(typeConverter, patterns);
  populateEqualLoweringPatterns(typeConverter, patterns);
  populateAndLoweringPatterns(typeConverter, patterns);
  populateDivLoweringPatterns(typeConverter, patterns);
  populateUnaryElementwiseLoweringPatterns(typeConverter, patterns);
  populateCumSumLoweringPatterns(typeConverter, patterns);
  populatePadLoweringPatterns(typeConverter, patterns);
  populateTileLoweringPatterns(typeConverter, patterns);
  populateExpandLoweringPatterns(typeConverter, patterns);
  populateLessLoweringPatterns(typeConverter, patterns);
  populateGatherNDLoweringPatterns(typeConverter, patterns);
  populateModLoweringPatterns(typeConverter, patterns);
  populateSliceLoweringPatterns(typeConverter, patterns);
  populateScatterNDLoweringPatterns(typeConverter, patterns);
  populateNonZeroLoweringPatterns(typeConverter, patterns);
  populateSizeLoweringPatterns(typeConverter, patterns);
  populateResizeLoweringPatterns(typeConverter, patterns);

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
