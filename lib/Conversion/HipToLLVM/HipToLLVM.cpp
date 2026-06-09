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

  // Rewrite @main_graph from the convert-hip-to-llvm internal ABI into the
  // runtime calling convention by splitting it into two functions:
  //
  //   main_graph_internal  the original graph body. Keeps the MLIR memref
  //     calling convention: every memref arg is passed as a FLAT list of
  //     scalars, not as one aggregate.
  //   main_graph  a thin wrapper with the runtime's narrow ABI -- the EP only
  //     knows how to call (ctx, inputs[, outputs]) where inputs/outputs are C
  //     arrays of pointers to memref descriptor structs. It rebuilds the flat
  //     argument list and calls main_graph_internal.
  //
  // The core idea is the "unpack" that bridges those two ABIs.
  //
  // A ranked memref lowers (MLIR's standard memref-to-LLVM convention) to a
  // descriptor STRUCT with 3 + 2*rank fields:
  //   { allocatedPtr, alignedPtr, offset, sizes[rank], strides[rank] }
  //      idx 0          idx 1       idx 2   idx 3        idx 4
  // But MLIR does NOT pass that struct by value as one parameter: a memref
  // parameter is EXPLODED into its 3 + 2*rank fields, i.e. that many separate
  // scalar parameters (2 ptrs, the i64 offset, then rank i64 sizes and rank
  // i64 strides). So for one rank-R input main_graph_internal's signature is
  // literally (%alloc:ptr, %aligned:ptr, %off:i64, <rank sizes:i64>,
  // <rank strides:i64>), NOT (%desc: struct<...>). That flat form is what
  // "unpacked memref params" means throughout this file.
  //
  // The runtime can't produce that flat list -- it only hands the EP a single
  // `inputs` pointer. So the wrapper, for each input i, does:
  //   1. GEP inputs[i]   -> ptr to the i-th descriptor pointer
  //   2. load it         -> ptr to the descriptor struct
  //   3. load that       -> the {ptr,ptr,i64,[..],[..]} struct value
  //   4. extractvalue field-by-field -> the 3 + 2*rank flat scalars
  // Step 4 is the "unpack" (see unpackMemRefStructWithAddrCast, which also
  // addrspace-casts the two ptrs back to addrspace(0)). Push those scalars in
  // order into the internal call's arg list; do it for the context + every
  // input (+ every output in classic mode) and the flat list exactly matches
  // main_graph_internal's exploded signature.
  //
  // Allocator vs classic mode is read from the `hipdnn.use_output_allocator`
  // module attribute (set by hip-use-output-allocator); the expected param count
  // is then used only to verify @main_graph matches that mode. The classic
  // main_graph has input AND output params; the allocator main_graph has input
  // params only and returns the in-graph-allocated output as a by-value memref
  // struct result (which adds no parameter).
  //
  // Allocator mode, rank-1 input + in-graph-allocated output.
  // Before (outputs are NOT params; the input is already exploded into
  // 5 = 3 + 2*1 scalars; main_graph returns the output descriptor):
  //   llvm.func @main_graph(%ctx, %inAlloc, %inAligned, %inOff, %inSz, %inSt)
  //       -> !llvm.struct<(ptr,ptr,i64,array<1xi64>,array<1xi64>)> { ... }
  // After (wrapper drops the outputs ptr and ignores the returned descriptor --
  // the output reaches the EP via the hipdnn_ep_alloc_output callback):
  //   llvm.func @main_graph_internal(<same 5 scalars>) -> !llvm.struct<...>
  //   llvm.func @main_graph(%ctx: ptr, %inputs: ptr) -> i32 {
  //     %p  = llvm.getelementptr %inputs[0]   ; &inputs[0]
  //     %sp = llvm.load %p                     ; ptr to the descriptor
  //     %m  = llvm.load %sp                     ; the descriptor struct value
  //     ; ---- unpack %m into 5 scalars ----
  //     %a  = llvm.extractvalue %m[0]          ; allocatedPtr
  //     %al = llvm.extractvalue %m[1]          ; alignedPtr
  //     %o  = llvm.extractvalue %m[2]          ; offset
  //     %s0 = llvm.extractvalue %m[3, 0]       ; sizes[0]
  //     %t0 = llvm.extractvalue %m[4, 0]       ; strides[0]
  //     ; -----------------------------------
  //     %_  = llvm.call @main_graph_internal(%ctx, %a, %al, %o, %s0, %t0)
  //     llvm.return %c0_i32
  //   }
  // Classic mode is the same except the wrapper takes (%ctx, %inputs,
  // %outputs), unpacks each output out-param the same way, and forwards the
  // internal's i32/void result.
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

    // Mode is decided by the `hipdnn.use_output_allocator` module attribute (set
    // by hip-use-output-allocator), NOT by param count. The attr is a typed
    // bool: read its VALUE (a module may carry it set to false), so absence and
    // `= false` both mean classic mode.
    auto allocatorAttr =
        module->getAttrOfType<BoolAttr>("hipdnn.use_output_allocator");
    bool allocatorMode = allocatorAttr && allocatorAttr.getValue();

    // Each memref unpacks to allocatedPtr + alignedPtr + offset + sizes[rank]
    // + strides[rank] = 3 + 2*rank LLVM params. Both modes have the context +
    // input params; only classic mode adds the output out-params (in allocator
    // mode outputs are allocated in-graph and returned as a by-value struct,
    // which adds no param) -- so the attr gates whether the output scope is
    // counted.
    constexpr unsigned kMemRefPtrs = 2;   // allocatedPtr + alignedPtr
    constexpr unsigned kMemRefOffset = 1; // offset scalar
    unsigned expectedParams = 1;          // context
    for (auto shapeAttr : inputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedParams += kMemRefPtrs + kMemRefOffset + rank + rank;
    }
    if (!allocatorMode) {
      for (auto shapeAttr : outputShapesAttr) {
        int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
        expectedParams += kMemRefPtrs + kMemRefOffset + rank + rank;
      }
    }

    unsigned actualParams = mainFunc.getFunctionType().getNumParams();
    if (actualParams != expectedParams) {
      return module.emitError()
             << "[HipToLLVM] @main_graph parameter count mismatch: expected "
             << expectedParams
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
