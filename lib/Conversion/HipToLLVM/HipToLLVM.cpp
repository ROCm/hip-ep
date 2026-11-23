/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "HipToLLVMUtils.h"
#include "hip/Conversion/HipToLLVM/Passes.h"
#include "hip/Dialect/IR/RuntimeAccessOpInterface.h"

#include "mlir/Conversion/ConvertToLLVM/ToLLVMInterface.h"
#include "mlir/Dialect/Bufferization/Transforms/BufferViewFlowAnalysis.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Dialect.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

#include <iterator>

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
  using Base::Base;
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
      return LLVM::AddrSpaceCastOp::create(builder, loc, as0PtrType, ptr);
    };

    Value allocPtr = LLVM::ExtractValueOp::create(
        builder, loc, memrefStruct, ArrayRef<int64_t>{kAllocPtrIdx});
    args.push_back(castToAs0(allocPtr));

    Value alignedPtr = LLVM::ExtractValueOp::create(
        builder, loc, memrefStruct, ArrayRef<int64_t>{kAlignedPtrIdx});
    args.push_back(castToAs0(alignedPtr));

    args.push_back(LLVM::ExtractValueOp::create(builder, loc, memrefStruct,
                                                ArrayRef<int64_t>{kOffsetIdx}));

    for (int64_t dim : llvm::seq<int64_t>(0, rank))
      args.push_back(LLVM::ExtractValueOp::create(
          builder, loc, memrefStruct, ArrayRef<int64_t>{kSizesIdx, dim}));
    for (int64_t dim : llvm::seq<int64_t>(0, rank))
      args.push_back(LLVM::ExtractValueOp::create(
          builder, loc, memrefStruct, ArrayRef<int64_t>{kStridesIdx, dim}));
  }

  // Split @main_graph into two functions so the runtime's simple (ctx, inputs)
  // ABI can reach the graph body:
  //
  //   main_graph_internal  the graph body, unchanged. It keeps MLIR's memref
  //     ABI: each memref arg is passed as its 3 + 2*rank scalar fields
  //     {allocatedPtr, alignedPtr, offset, sizes[rank], strides[rank]}, not as
  //     one struct. ("unpacked memref params" means this flat scalar form.)
  //   main_graph  a thin wrapper (called by inference_compute, not the EP
  //     directly) taking (ctx, inputs). Here `inputs` is a C array of pointers
  //     to memref descriptor structs that inference_compute built from the
  //     input tensors -- NOT the span_t the EP passes to inference_compute.
  //     For each input it loads the descriptor and unpacks it back into the
  //     scalar fields
  //     (unpackMemRefStructWithAddrCast, which also casts the two ptrs to
  //     addrspace(0)), then calls main_graph_internal. Outputs are NOT passed
  //     in: the graph allocates them via hip.alloc_output (the EP callback) and
  //     returns them by value, which the wrapper ignores.
  //
  // Example: 1 rank-1 input, 1 in-graph output.
  // Before (input already flattened to 5 = 3 + 2*1 scalars; returns the output
  // descriptor):
  //   llvm.func @main_graph(%ctx, %inAlloc, %inAligned, %inOff, %inSz, %inSt)
  //       -> !llvm.struct<(ptr,ptr,i64,array<1xi64>,array<1xi64>)> { ... }
  // After (wrapper takes only (ctx, inputs); the returned descriptor is
  // ignored, the output reaches the EP via the callback). A marked fallible
  // runtime call returns immediately from the internal graph:
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
  LogicalResult transformMainFunction(ModuleOp module) {
    WalkResult markerValidation = module.walk([&](LLVM::CallOp call) {
      Attribute marker = call->getAttr(kFallibleRuntimeCallAttrName);
      if (!marker || isa<UnitAttr>(marker))
        return WalkResult::advance();
      call.emitError() << "requires '" << kFallibleRuntimeCallAttrName
                       << "' to be a unit attribute";
      return WalkResult::interrupt();
    });
    if (markerValidation.wasInterrupted())
      return failure();

    auto mainFunc = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");

    auto inputCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.input_count");
    auto outputCountAttr =
        module->getAttrOfType<IntegerAttr>("hipdnn.output_count");
    auto inputShapesAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.input_shapes");
    auto outputShapesAttr =
        module->getAttrOfType<ArrayAttr>("hipdnn.output_shapes");

    const bool hasCompleteMainAbi = mainFunc && inputCountAttr &&
                                    outputCountAttr && inputShapesAttr &&
                                    outputShapesAttr;
    if (!hasCompleteMainAbi) {
      LLVM::CallOp unpropagatedCall;
      module.walk([&](LLVM::CallOp call) {
        if (isFallibleRuntimeCall(call)) {
          unpropagatedCall = call;
          return WalkResult::interrupt();
        }
        return WalkResult::advance();
      });
      if (unpropagatedCall) {
        return unpropagatedCall.emitError(
            "fallible runtime call requires @main_graph and complete "
            "hipdnn input/output metadata for status propagation");
      }
      if (!mainFunc)
        return success();
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

    // Each memref is 3 + 2*rank scalar params (allocatedPtr, alignedPtr,
    // offset, sizes[rank], strides[rank]). @main_graph has context + inputs
    // only; outputs are returned by value and add no param.
    constexpr unsigned kMemRefPtrs = 2;   // allocatedPtr + alignedPtr
    constexpr unsigned kMemRefOffset = 1; // offset scalar
    unsigned expectedParams = 1;          // context
    for (auto shapeAttr : inputShapesAttr) {
      int64_t rank = cast<DenseI64ArrayAttr>(shapeAttr).size();
      expectedParams += kMemRefPtrs + kMemRefOffset + rank + rank;
    }

    unsigned actualParams = mainFunc.getFunctionType().getNumParams();
    if (actualParams != expectedParams) {
      return module.emitError()
             << "[HipToLLVM] @main_graph parameter count mismatch: expected "
             << expectedParams << ", got " << actualParams;
    }

    OpBuilder builder(module.getContext());
    Location loc = mainFunc.getLoc();
    Type i32Type = builder.getI32Type();

    // The public graph ABI can propagate a runtime status only from
    // @main_graph. Reject helper-local marked calls instead of silently
    // dropping their result until outlined functions have an explicit error
    // convention.
    WalkResult helperCall = module.walk([&](LLVM::CallOp call) {
      if (!isFallibleRuntimeCall(call) ||
          call->getParentOfType<LLVM::LLVMFuncOp>() == mainFunc) {
        return WalkResult::advance();
      }
      call.emitError(
          "fallible runtime call is unsupported outside @main_graph");
      return WalkResult::interrupt();
    });
    if (helperCall.wasInterrupted())
      return failure();

    // Short-circuit @main_graph immediately after the first failed marked
    // call. This prevents later kernels from consuming an invalid result and
    // propagates the exact nonzero status through inference_compute.
    SmallVector<LLVM::CallOp> fallibleCalls;
    mainFunc.walk([&](LLVM::CallOp call) {
      if (isFallibleRuntimeCall(call))
        fallibleCalls.push_back(call);
    });
    const bool propagatesRuntimeStatus = !fallibleCalls.empty();
    if (propagatesRuntimeStatus) {
      for (LLVM::CallOp call : fallibleCalls) {
        if (call.getNumResults() != 1 ||
            !call.getResult().getType().isInteger(32)) {
          return call.emitError(
              "fallible runtime wrapper must return one i32 status");
        }
      }

      Block &entry = mainFunc.getBody().front();
      builder.setInsertionPointToStart(&entry);
      Value zero = LLVM::ConstantOp::create(builder, loc, i32Type,
                                            builder.getI32IntegerAttr(0));

      SmallVector<LLVM::ReturnOp> returns;
      mainFunc.walk(
          [&](LLVM::ReturnOp returnOp) { returns.push_back(returnOp); });
      for (LLVM::ReturnOp returnOp : returns) {
        builder.setInsertionPoint(returnOp);
        LLVM::ReturnOp::create(builder, returnOp.getLoc(), ValueRange{zero});
        returnOp.erase();
      }

      LLVM::LLVMFunctionType oldType = mainFunc.getFunctionType();
      mainFunc.setFunctionType(LLVM::LLVMFunctionType::get(
          i32Type, oldType.getParams(), oldType.isVarArg()));
      mainFunc.removeResAttrsAttr();

      Block *errorBlock = mainFunc.addBlock();
      Value errorStatus = errorBlock->addArgument(i32Type, loc);
      builder.setInsertionPointToStart(errorBlock);
      LLVM::ReturnOp::create(builder, loc, ValueRange{errorStatus});

      // Split from the last call back: a split moves every operation after
      // the call into the new block, so going forward would move the rest of
      // the graph once per call. Backward, each operation moves once.
      for (LLVM::CallOp call : llvm::reverse(fallibleCalls)) {
        Block *callBlock = call->getBlock();
        Block *continueBlock =
            callBlock->splitBlock(std::next(call->getIterator()));
        builder.setInsertionPointToEnd(callBlock);
        Value failed = LLVM::ICmpOp::create(builder, call.getLoc(),
                                            LLVM::ICmpPredicate::ne,
                                            call.getResult(), zero);
        LLVM::CondBrOp::create(builder, call.getLoc(), failed, errorBlock,
                               ValueRange{call.getResult()}, continueBlock,
                               ValueRange{});
      }
    }

    // Rename the original main_graph so we can add a (ctx, inputs) wrapper that
    // unpacks the input memref structs and forwards the call. No outputs arg.
    mainFunc.setName("main_graph_internal");
    mainFunc.setLinkage(LLVM::Linkage::Private);

    Type ptrType = LLVM::LLVMPointerType::get(builder.getContext(), 0);
    SmallVector<Type> newParamTypes{ptrType, ptrType}; // (ctx, inputs)
    auto newFuncType = LLVM::LLVMFunctionType::get(i32Type, newParamTypes);

    // Create the (ctx, inputs) wrapper.
    builder.setInsertionPoint(mainFunc);
    auto newMainFunc =
        LLVM::LLVMFuncOp::create(builder, loc, "main_graph", newFuncType);
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
      Value inputIdxVal = LLVM::ConstantOp::create(
          builder, loc, i32Type, builder.getI32IntegerAttr(i));
      Value inputSlotPtr = LLVM::GEPOp::create(
          builder, loc, ptrType, ptrType, inputsArg, ValueRange{inputIdxVal});
      Value inputStructPtr =
          LLVM::LoadOp::create(builder, loc, ptrType, inputSlotPtr);
      Type memrefStructType = getMemRefStructType(builder, rank, 1);
      Value inputMemref =
          LLVM::LoadOp::create(builder, loc, memrefStructType, inputStructPtr);
      unpackMemRefStructWithAddrCast(builder, loc, inputMemref, rank,
                                     mainInternalArgs);
    }

    // Outputs are not out-params: the graph allocates them through callbacks,
    // so its original returned descriptors are intentionally discarded. When
    // fallible runtime calls are present, the rewritten internal graph instead
    // returns the first failed status or zero; otherwise preserve the
    // historical zero status.
    auto internalCall =
        LLVM::CallOp::create(builder, loc, mainFunc, mainInternalArgs);
    Value status = propagatesRuntimeStatus
                       ? internalCall.getResult()
                       : LLVM::ConstantOp::create(builder, loc, i32Type,
                                                  builder.getI32IntegerAttr(0));
    LLVM::ReturnOp::create(builder, loc, status);

    COMPILER_DEBUG_LOG("[HipToLLVM] Transformed @main_graph signature: "
                       << actualParams << " params -> " << newParamTypes.size()
                       << " params\n");
    return success();
  }
};

static void
populateLoweringPatterns(LLVMTypeConverter &typeConverter,
                         RewritePatternSet &patterns,
                         const DenseMap<Operation *, Value> &hostScalars) {
  // HIP dialect-specific lowerings
  populateMemoryLoweringPatterns(typeConverter, patterns);
  populateConvLoweringPatterns(typeConverter, patterns);
  populateConvTransposeLoweringPatterns(typeConverter, patterns);
  populateMatmulLoweringPatterns(typeConverter, patterns);
  populateRocMlirLoweringPatterns(typeConverter, patterns);
  populateElementwiseLoweringPatterns(typeConverter, patterns);
  populatePowerLoweringPatterns(typeConverter, patterns);
  populateActivationLoweringPatterns(typeConverter, patterns);
  populateBiasGeluLoweringPatterns(typeConverter, patterns);
  populateFastGeluLoweringPatterns(typeConverter, patterns);
  populateTriluLoweringPatterns(typeConverter, patterns);
  populateNormLoweringPatterns(typeConverter, patterns);
  populateGatherLoweringPatterns(typeConverter, patterns);
  populateGatherElementsLoweringPatterns(typeConverter, patterns);
  populateTopKLoweringPatterns(typeConverter, patterns);
  populateRandomNormalLikeLoweringPatterns(typeConverter, patterns);
  populateArgMaxLoweringPatterns(typeConverter, patterns);
  populateScatterElementsLoweringPatterns(typeConverter, patterns);
  populateCompressLoweringPatterns(typeConverter, patterns);
  populateOneHotLoweringPatterns(typeConverter, patterns);
  populateRangeLoweringPatterns(typeConverter, patterns);
  populateCastLoweringPatterns(typeConverter, patterns);
  populateReduceLoweringPatterns(typeConverter, patterns);
  populateTransposeLoweringPatterns(typeConverter, patterns);
  populateRopeLoweringPatterns(typeConverter, patterns);
  populateGqaLoweringPatterns(typeConverter, patterns, hostScalars);
  populateMultiHeadAttentionLoweringPatterns(typeConverter, patterns);
  populateMatMulNBitsLoweringPatterns(typeConverter, patterns);
  populateDequantBnb4LoweringPatterns(typeConverter, patterns);
  populateQMoELoweringPatterns(typeConverter, patterns);
  populateQMoEAmdLoweringPatterns(typeConverter, patterns);
  populateGatherBlockQuantizedLoweringPatterns(typeConverter, patterns);
  populateQdqLoweringPatterns(typeConverter, patterns);
  populateGemmLoweringPatterns(typeConverter, patterns);
  populateLinearAttentionLoweringPatterns(typeConverter, patterns);
  populateGatedDeltaNetLoweringPatterns(typeConverter, patterns);
  populateKQuantLoweringPatterns(typeConverter, patterns);
  populateDynamicSparseAttentionLoweringPatterns(typeConverter, patterns);
  populateSparseAttentionIndexerLoweringPatterns(typeConverter, patterns);
  populateGraphLoweringPatterns(typeConverter, patterns);
  populateLoopLoweringPatterns(typeConverter, patterns);
  populateScanLoweringPatterns(typeConverter, patterns);
  populateIfLoweringPatterns(typeConverter, patterns);
  populateCausalConvWithStateLoweringPatterns(typeConverter, patterns);
  populateNGramHashMappingLoweringPatterns(typeConverter, patterns);
  populateWhereLoweringPatterns(typeConverter, patterns);
  populateEqualLoweringPatterns(typeConverter, patterns);
  populateOrLoweringPatterns(typeConverter, patterns);
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
  populateNonMaxSuppressionLoweringPatterns(typeConverter, patterns);
  populateReadbackDimLoweringPatterns(typeConverter, patterns, hostScalars);
  populateReadbackScalarLoweringPatterns(typeConverter, patterns, hostScalars);
  populateSizeLoweringPatterns(typeConverter, patterns);
  populatePoolLoweringPatterns(typeConverter, patterns);
  populateResizeLoweringPatterns(typeConverter, patterns);
  populateGridSampleLoweringPatterns(typeConverter, patterns);
  populateRoiAlignLoweringPatterns(typeConverter, patterns);
  populateGlobalPoolLoweringPatterns(typeConverter, patterns);
  populateQElementwiseLoweringPatterns(typeConverter, patterns);
  populateQMatMulLoweringPatterns(typeConverter, patterns);
  populateQGemmLoweringPatterns(typeConverter, patterns);
  populateQConvLoweringPatterns(typeConverter, patterns);
  populateQLinearConvLoweringPatterns(typeConverter, patterns);
  populateQLinearMatMulLoweringPatterns(typeConverter, patterns);
  populateQLpNormalizationLoweringPatterns(typeConverter, patterns);
  populateQSigmoidLoweringPatterns(typeConverter, patterns);

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
}

// A promised interface is not necessarily registered. The standard dialect
// patterns above handle those cases without querying an unresolved promise.
static void populateDialectLowerings(Dialect *dialect,
                                     LLVMTypeConverter &typeConverter,
                                     RewritePatternSet &patterns,
                                     LLVMConversionTarget &target) {
  if (!dialect || dialect->hasPromisedInterface(
                      dialect->getTypeID(),
                      ConvertToLLVMPatternInterface::getInterfaceID()))
    return;
  if (auto *iface = dyn_cast<ConvertToLLVMPatternInterface>(dialect))
    iface->populateConvertToLLVMConversionPatterns(target, typeConverter,
                                                   patterns);
}

void ConvertHipToLLVMPass::runOnOperation() {
  ModuleOp module = getOperation();
  MLIRContext *ctx = module.getContext();

  if (Attribute policy = module->getAttr(kRequireFallibleMemRefCopiesAttrName);
      policy && !isa<UnitAttr>(policy)) {
    module.emitError() << "'" << kRequireFallibleMemRefCopiesAttrName
                       << "' must be a unit attribute";
    signalPassFailure();
    return;
  }

  LLVMTypeConverter typeConverter(ctx);
  // Scratch placement and the runtime memref ABI both use 64-bit indices.
  if (typeConverter.getIndexTypeBitwidth() != 64) {
    module.emitError("HIP runtime requires 64-bit indices");
    return signalPassFailure();
  }
  // Establish native type/legality rules before plugins extend them.
  typeConverter.addConversion([ctx](ContextType) -> Type {
    return LLVM::LLVMPointerType::get(ctx, 0);
  });
  RewritePatternSet patterns(ctx), alternatives(ctx);
  LLVMConversionTarget target(*ctx);
  target.addLegalDialect<LLVM::LLVMDialect>();
  target.addIllegalDialect<HipDialect>();
  target.addIllegalOp<memref::AllocOp, memref::DeallocOp>();
  target.addLegalOp<ModuleOp>();
  DenseSet<Dialect *> seenDialects;
  SmallVector<Operation *> mappedInputOps;
  SmallVector<Value> scratches;
  module.walk([&](Operation *op) {
    Dialect *dialect = op->getDialect();
    if (seenDialects.insert(dialect).second)
      populateDialectLowerings(dialect, typeConverter, alternatives, target);
    if (auto scratch = dyn_cast<GetHostScratchOp>(op))
      scratches.push_back(scratch.getResult());
    if (auto access = dyn_cast<RuntimeAccessOpInterface>(op))
      if (access.supportsHostMappedInputs())
        mappedInputOps.push_back(op);
  });

  RuntimeAccessOverrides overrides(alternatives);
  llvm::erase_if(mappedInputOps, [&](Operation *op) {
    return !overrides.contains(op->getName());
  });
  // Placement may have relied on mapped-input support before a plugin was
  // loaded. Check consumers of scratch-derived views, not unrelated ops.
  if (!scratches.empty() && !mappedInputOps.empty()) {
    BufferViewFlowAnalysis aliases(module);
    DenseSet<Value> scratchAliases;
    for (Value scratch : scratches)
      for (Value alias : aliases.resolve(scratch))
        scratchAliases.insert(alias);
    for (Operation *op : mappedInputOps)
      if (llvm::any_of(op->getOperands(), [&](Value operand) {
            return scratchAliases.contains(operand);
          })) {
        op->emitError("alternative lowering has no host-mapped input contract");
        return signalPassFailure();
      }
  }

  const auto hostScalars = collectHostScalarValues(
      module, overrides, typeConverter.getIndexTypeBitwidth(),
      reportHostScalars);
  populateLoweringPatterns(typeConverter, patterns, hostScalars);
  // Preserve native-before-plugin order for equal-benefit patterns.
  for (auto &pattern : alternatives.getNativePatterns())
    patterns.add(std::move(pattern));
  patterns.add(std::move(alternatives.getPDLPatterns()));
  if (failed(applyPartialConversion(module, target, std::move(patterns))))
    return signalPassFailure();

  // A ui8 If condition is compared as signless i8. Lowering the readback
  // inserts the reverse cast, and that round trip is not executable LLVM.
  // Drop casts whose chain returns to the original type.
  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk(
      [&](UnrealizedConversionCastOp castOp) { casts.push_back(castOp); });
  reconcileUnrealizedCasts(casts);

  if (failed(transformMainFunction(module)))
    signalPassFailure();
}

} // namespace

} // namespace hip
} // namespace mlir
