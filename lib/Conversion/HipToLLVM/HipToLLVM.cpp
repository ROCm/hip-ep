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
  //   main_graph_internal  the original graph body, keeping the MLIR memref
  //     calling convention: a ranked-memref arg is NOT one struct parameter --
  //     it is EXPLODED into its 3 + 2*rank descriptor fields
  //     {allocatedPtr, alignedPtr, offset, sizes[rank], strides[rank]} as that
  //     many separate scalar params. This flat form is what "unpacked memref
  //     params" means throughout this file.
  //   main_graph  a thin wrapper with the runtime's narrow ABI: the EP only
  //     hands it (ctx, inputs[, outputs]) where inputs/outputs are C arrays of
  //     pointers to memref descriptor structs. For each memref the wrapper
  //     loads the descriptor and re-explodes it into scalars
  //     (unpackMemRefStructWithAddrCast, which also addrspace-casts the two
  //     ptrs back to addrspace(0)), then calls main_graph_internal with the
  //     concatenated flat list. That re-explode is the "unpack" bridging the
  //     two ABIs.
  //
  // Mode is read from the `hipdnn.use_output_allocator` module attribute (set
  // by hip-use-output-allocator); the expected param count is then used only to
  // verify @main_graph matches it. Classic main_graph has input AND output
  // params; allocator main_graph has input params only and returns the
  // in-graph-allocated output as a by-value memref result (which adds no
  // param).
  //
  // Allocator mode, rank-1 input + in-graph-allocated output:
  // Before (no output params; input already exploded into 5 = 3 + 2*1 scalars;
  // main_graph returns the output descriptor):
  //   llvm.func @main_graph(%ctx, %inAlloc, %inAligned, %inOff, %inSz, %inSt)
  //       -> !llvm.struct<(ptr,ptr,i64,array<1xi64>,array<1xi64>)> { ... }
  // After (wrapper drops the outputs ptr and ignores the returned descriptor --
  // the output reaches the EP via the hipdnn_ep_alloc_output callback):
  //   llvm.func @main_graph_internal(<same 5 scalars>) -> !llvm.struct<...>
  //   llvm.func @main_graph(%ctx: ptr, %inputs: ptr) -> i32 {
  //     %sp = llvm.load (gep %inputs[0])       ; ptr to the descriptor
  //     %m  = llvm.load %sp                     ; the descriptor struct value
  //     %a  = llvm.extractvalue %m[0]          ; allocatedPtr  \  unpack into
  //     %al = llvm.extractvalue %m[1]          ; alignedPtr     > the 5 scalars
  //     %o  = llvm.extractvalue %m[2]          ; offset        /  the internal
  //     %s0 = llvm.extractvalue %m[3, 0]       ; sizes[0]      \  signature
  //     %t0 = llvm.extractvalue %m[4, 0]       ; strides[0]    /  expects
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

    // Mode comes from the attribute's VALUE, not its presence (see header):
    // absent or `= false` => classic.
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

    for (int64_t i : llvm::seq<int64_t>(0, inputCount)) {
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
      for (int64_t i : llvm::seq<int64_t>(0, outputCount)) {
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
    // The `allocatorMode` term is required: in allocator mode internalRetTy is
    // the memref descriptor struct (neither void nor i32), so without it this
    // would fall into the else branch and return that struct from the i32
    // wrapper -- a type mismatch. The attr routes allocator mode to discard +
    // return 0 instead.
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

  // Map each HIP memory space (#hip.mem<device|host|pinned|managed>) to a
  // DISTINCT LLVM address space, numbered by the MemorySpaceKind enum:
  //   device = 0, host = 1, pinned = 2, managed = 3.
  // Device deliberately stays AS 0 so the dominant GPU data path (model
  // params, activations, the device pool, all kernel/GEMM operands) and the
  // flat bare-pointer runtime ABI are byte-unchanged; only the rarer
  // host/pinned/managed boundary buffers move to a non-default space. Keeping
  // the space distinct in the lowered LLVM type makes D2H/H2D boundaries
  // self-describing and lets address-space-based alias reasoning treat device
  // vs host pointers as non-aliasing.
  //
  //   memref<8xi64, #hip.mem<host>>  ->  !llvm.ptr<1> in the descriptor
  //   memref<8xi64, #hip.mem<device>> (or no space)  ->  !llvm.ptr<0>
  //
  // Every per-op lowering resolves the space via
  // getTypeConverter()->getMemRefAddressSpace() and addrspace-casts the
  // runtime's generic (AS 0) pointer up to it / back down at the call boundary
  // (see MemoryLowering.cpp and unpackMemRefStructWithAddrCast), so runtime
  // declarations stay AS-0 `!llvm.ptr`. The host target collapses all these
  // spaces to one flat space for codegen, so CPU loads/stores on the
  // host/pinned/managed buffers (e.g. tensor.extract after a D2H transfer)
  // remain valid. Without this hook the LLVM type converter rejects any memref
  // carrying a non-integer memory space and the whole conversion fails.
  typeConverter.addTypeAttributeConversion(
      [ctx](BaseMemRefType, MemorySpaceAttr attr)
          -> TypeConverter::AttributeConversionResult {
        return IntegerAttr::get(IntegerType::get(ctx, 64),
                                static_cast<int64_t>(attr.getKind()));
      });

  RewritePatternSet patterns(ctx);

  // HIP dialect-specific lowerings
  populateMemoryLoweringPatterns(typeConverter, patterns);
  populateConvLoweringPatterns(typeConverter, patterns);
  populateConvTransposeLoweringPatterns(typeConverter, patterns);
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
  populateGatherBlockQuantizedLoweringPatterns(typeConverter, patterns);
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
  populateReadbackDimLoweringPatterns(typeConverter, patterns);
  populateReadbackScalarLoweringPatterns(typeConverter, patterns);
  populateTransferLoweringPatterns(typeConverter, patterns);
  populateSizeLoweringPatterns(typeConverter, patterns);
  populatePoolLoweringPatterns(typeConverter, patterns);
  populateResizeLoweringPatterns(typeConverter, patterns);
  populateGlobalPoolLoweringPatterns(typeConverter, patterns);

  // Standard dialect lowerings
  // Bundle func/memref/arith/cf lowering with HIP lowering to minimize
  // unrealized casts at the memref/LLVM boundary. Running them as separate
  // stages would require a reconcile-unrealized-casts cleanup pass.
  populateFuncToLLVMConversionPatterns(typeConverter, patterns);
  populateFinalizeMemRefToLLVMConversionPatterns(typeConverter, patterns);
  // ArithToLLVM has no direct lowering for arith.{ceil,floor}div{s,u}i; these
  // must first be expanded into primitive arith ops (cmp/sub/add/div/select),
  // so the expand patterns and the ToLLVM patterns form a pair that must be
  // populated together. Upstream's -convert-arith-to-llvm pass does exactly
  // this; because we hand-roll the pattern set, we replicate the pairing.
  // Omitting the expand patterns lets a stray ceildivsi (e.g. from
  // dynamic-shape index arithmetic) survive applyPartialConversion and abort
  // MLIR->LLVM translation with "missing LLVMTranslationDialectInterface".
  arith::populateCeilFloorDivExpandOpsPatterns(patterns);
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
