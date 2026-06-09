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

  // Rewrite @main_graph from the convert-hip-to-llvm internal ABI (context +
  // unpacked memref params) into the runtime calling convention. Allocator vs
  // classic mode is read from the `hipdnn.output_allocator` module attribute
  // (set by hip-set-output-allocator-attr); the expected param count is then
  // used only to verify @main_graph matches that mode. Each memref unpacks to
  // 3 + 2*rank params; a returned memref lowers to a by-value struct result (no
  // extra param). The classic main_graph has input AND output params; the
  // allocator main_graph has input params only and returns the
  // in-graph-allocated output memref.
  //
  // Allocator mode (rank-1 input + in-graph-allocated output):
  //   Before (outputs are NOT params; main_graph returns the memref
  //   descriptor):
  //     llvm.func @main_graph(%ctx, %inAlloc, %inAligned, %inOff, %inSz, %inSt)
  //         -> !llvm.struct<(ptr,ptr,i64,array<1xi64>,array<1xi64>)> { ... }
  //   After (wrapper drops the outputs ptr; ignores the returned descriptor --
  //   the output reaches the EP via the hipdnn_ep_alloc_output callback):
  //     llvm.func @main_graph_internal(...same...) -> !llvm.struct<...>
  //     llvm.func @main_graph(%ctx: !llvm.ptr, %inputs: !llvm.ptr) -> i32 {
  //       %m = llvm.load <inputs[0]> ; <unpack %m into internal args>
  //       %_ = llvm.call @main_graph_internal(%ctx, <unpacked>) -> struct
  //       llvm.return %c0_i32
  //     }
  // Classic mode is the same except the wrapper takes (%ctx, %inputs,
  // %outputs), also unpacks each output out-param, and forwards the internal's
  // i32/void result.
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

    // Each memref unpacks to allocatedPtr + alignedPtr + offset + sizes[rank]
    // + strides[rank] = 3 + 2*rank LLVM params. The allocator main_graph has
    // the INPUT params only (outputs are allocated in-graph and returned as a
    // by-value struct, which adds no param); the classic main_graph also has
    // the output out-params.
    constexpr unsigned kMemRefPtrs = 2;   // allocatedPtr + alignedPtr
    constexpr unsigned kMemRefOffset = 1; // offset scalar
    unsigned expectedAllocator = 1;       // context
    for (auto shapeAttr : inputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedAllocator += kMemRefPtrs + kMemRefOffset + rank + rank;
    }
    unsigned expectedClassic = expectedAllocator;
    for (auto shapeAttr : outputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedClassic += kMemRefPtrs + kMemRefOffset + rank + rank;
    }

    unsigned actualParams = mainFunc.getFunctionType().getNumParams();
    // Mode is decided by the `hipdnn.output_allocator` module attribute (set by
    // hip-set-output-allocator-attr), NOT by param count -- this disambiguates
    // a zero-output graph, where expectedClassic == expectedAllocator. The
    // count is then only used to verify @main_graph matches the chosen mode.
    bool allocatorMode = module->hasAttr("hipdnn.output_allocator");
    unsigned expected = allocatorMode ? expectedAllocator : expectedClassic;
    if (actualParams != expected) {
      return module.emitError()
             << "[HipToLLVM] @main_graph parameter count mismatch: expected "
             << expected
             << (allocatorMode ? " (allocator), got " : " (classic), got ")
             << actualParams;
    }

    OpBuilder builder(module.getContext());
    Location loc = mainFunc.getLoc();

    // Rename the original main_graph (with unpacked memref params) so we can
    // create a new wrapper that takes the runtime calling convention --
    // (ctx, inputs, outputs) in classic mode, or (ctx, inputs) in allocator
    // mode (outputs are allocated in-graph) -- and unpacks the input (and, in
    // classic mode, output) memref structs before forwarding the call.
    mainFunc.setName("main_graph_internal");
    mainFunc.setLinkage(LLVM::Linkage::Private);

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    Type i32Type = builder.getI32Type();
    // Allocator wrapper drops the trailing outputs pointer arg.
    SmallVector<Type> newParamTypes =
        allocatorMode ? SmallVector<Type>{ptrType, ptrType}
                      : SmallVector<Type>{ptrType, ptrType, ptrType};
    auto newFuncType = LLVM::LLVMFunctionType::get(i32Type, newParamTypes);

    // Create the new main_graph wrapper with the simplified runtime signature
    // ((ctx, inputs, outputs) classic / (ctx, inputs) allocator) computed
    // above.
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

    // Classic mode forwards each output out-param (unpacked memref struct) to
    // the internal call. Allocator mode has no output args -- the internal
    // main_graph allocates and returns the output via hip.alloc_output, so the
    // outputs pointer is absent from the wrapper signature.
    if (!allocatorMode) {
      Value outputsArg = entryBlock->getArgument(2);
      for (int64_t i = 0; i < outputCount; i++) {
        int64_t rank = cast<DenseI64ArrayAttr>(outputShapesAttr[i]).size();
        Value outputIdxVal = builder.create<LLVM::ConstantOp>(
            loc, i32Type, builder.getI32IntegerAttr(i));
        Value outputSlotPtr = builder.create<LLVM::GEPOp>(
            loc, ptrType, ptrType, outputsArg, ValueRange{outputIdxVal});
        Value outputStructPtr =
            builder.create<LLVM::LoadOp>(loc, ptrType, outputSlotPtr);
        Type memrefStructType = getMemRefStructType(builder, rank, 1);
        Value outputMemref = builder.create<LLVM::LoadOp>(loc, memrefStructType,
                                                          outputStructPtr);
        unpackMemRefStructWithAddrCast(builder, loc, outputMemref, rank,
                                       mainInternalArgs);
      }
    }

    // Forward to the internal main_graph and return the i32 status:
    //   - allocator mode: the internal returns a memref descriptor (by-value
    //     struct) for the in-graph-allocated output. Call it, DISCARD the
    //     struct result (the output reaches the EP through the
    //     hipdnn_ep_alloc_output callback, not the return value), and return 0.
    //   - classic void internal (real lowering): call, return 0.
    //   - classic i32 internal (hand-written LIT shape): call, forward result.
    auto internalRetTy = mainFunc.getFunctionType().getReturnType();
    if (allocatorMode || isa<LLVM::LLVMVoidType>(internalRetTy)) {
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
                       << actualParams << " params -> " << newParamTypes.size()
                       << " params ("
                       << (allocatorMode ? "allocator" : "classic") << ")\n");
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
