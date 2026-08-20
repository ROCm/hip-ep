/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- LoopLowering.cpp - Lower hip.loop to runtime driver calls ---------===//
//
// Lowers `hip.loop` to a call into the HIPDNN runtime via the fast-path
// `hipdnn_ep_run_counted_loop` or the slow-path `hipdnn_ep_run_loop`
// symbol, with a per-loop codegen'd LLVM trampoline that bridges the
// runtime's fixed-arity callback contract to the body's variable-arity
// memref-descriptor signature.
//
// Trampoline ABI:
//   Receives  : (state*, frame*, iter_dev*, cond_dev*, current_descs*,
//                cap_descs*, next_descs*)
//   Builds    : rank-0 memref descriptors for iter and cond from the raw
//               device pointers; loads each loop-carried / capture
//               descriptor struct from its slot in the *_descs array.
//   Calls body: (state, iter, cond_in, v_current..., captures..., frame)
//               -> (i32 status, [cond_out,] v_next...).
//   Publishes : returned carrier descriptors into next_descs only after all
//               frame allocations succeed. Runtime swaps descriptor sets only
//               when the callback returns success.
//
//===----------------------------------------------------------------------===//

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace hip {
namespace {

// Runtime symbol names. See `lib/Runtime/hipdnn_ep_runtime.h`.
constexpr const char *kRunCountedLoop = "hipdnn_ep_run_counted_loop";
constexpr const char *kRunLoop = "hipdnn_ep_run_loop";

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

/// Append the expanded fields of a (rank-aware) memref descriptor struct
/// to `out`.  Field order matches the calling convention produced by
/// `populateFuncToLLVMConversionPatterns`: allocPtr, alignedPtr, offset,
/// then sizes[0..rank), then strides[0..rank).
static void expandMemRefStruct(OpBuilder &b, Location loc, Value descStruct,
                               int64_t rank, SmallVectorImpl<Value> &out) {
  Value allocPtr = LLVM::ExtractValueOp::create(
      b, loc, descStruct, ArrayRef<int64_t>{kAllocPtrIdx});
  out.push_back(allocPtr);
  Value alignedPtr = LLVM::ExtractValueOp::create(
      b, loc, descStruct, ArrayRef<int64_t>{kAlignedPtrIdx});
  out.push_back(alignedPtr);
  Value offset = LLVM::ExtractValueOp::create(b, loc, descStruct,
                                              ArrayRef<int64_t>{kOffsetIdx});
  out.push_back(offset);
  for (int64_t dim = 0; dim < rank; ++dim)
    out.push_back(LLVM::ExtractValueOp::create(
        b, loc, descStruct, ArrayRef<int64_t>{kSizesIdx, dim}));
  for (int64_t dim = 0; dim < rank; ++dim)
    out.push_back(LLVM::ExtractValueOp::create(
        b, loc, descStruct, ArrayRef<int64_t>{kStridesIdx, dim}));
}

/// Emit the expanded fields of a rank-0 memref descriptor (allocPtr,
/// alignedPtr, offset) directly into `out`, given just a raw pointer.
/// Used by the trampoline to wrap the runtime-supplied iter / cond
/// device pointers without round-tripping through a struct value.
static void emitRank0DescriptorFields(OpBuilder &b, Location loc, Value rawPtr,
                                      SmallVectorImpl<Value> &out) {
  Type i64Ty = b.getI64Type();
  Value zero = LLVM::ConstantOp::create(b, loc, i64Ty, b.getI64IntegerAttr(0));
  out.push_back(rawPtr); // allocPtr
  out.push_back(rawPtr); // alignedPtr
  out.push_back(zero);   // offset
}

/// Return the rank of the memref described by `descStruct`'s LLVM struct
/// type, by inspecting the sizes-array field's element count.  Returns 0
/// for rank-0 memrefs (3-element struct{ptr, ptr, i64}).
static int64_t memrefRankFromStructType(Type structTy) {
  auto st = dyn_cast<LLVM::LLVMStructType>(structTy);
  if (!st)
    return 0;
  // Rank-0: {ptr, ptr, i64}.  Rank-N: {ptr, ptr, i64, [N x i64], [N x i64]}.
  if (st.getBody().size() < 4)
    return 0;
  auto arrTy = dyn_cast<LLVM::LLVMArrayType>(st.getBody()[kSizesIdx]);
  return arrTy ? static_cast<int64_t>(arrTy.getNumElements()) : 0;
}

/// Build (or reuse) the trampoline LLVMFuncOp for this loop.
///
/// Inserts `<body>_trampoline` at module scope.  Its body:
///   * Receives (state, frame, iter_dev, cond_dev, current, captures, next).
///   * Builds rank-0 memref descriptors for iter and cond from the raw ptrs.
///   * Loads the loop-carried and captured memref descriptors from the
///     pointer arrays (each entry is a pointer to a descriptor struct
///     allocated by the calling main_graph function).
///   * Calls the outlined body using MLIR's standard descriptor-result ABI.
///   * Copies dynamic cond_out into the frame-local condition slot and writes
///     all next carrier descriptors transactionally.
///
/// Precondition: the body has not yet been converted to LLVMFuncOp.  The
/// LoopOpLowering caller enforces this by failing if `module.lookupSymbol
/// <LLVM::LLVMFuncOp>(bodyName)` resolves -- the partial-conversion driver
/// processes the body and the LoopOp in the same applyPartialConversion
/// call, but our index math here assumes the unexpanded struct-per-memref
/// form returned by `typeConverter.convertType(memref) -> struct`.
static LLVM::LLVMFuncOp
createOrGetTrampoline(OpBuilder &b, ModuleOp module, Location loc, LoopOp op,
                      func::FuncOp bodyFuncFn,
                      const LLVMTypeConverter &typeConverter, unsigned numLC,
                      unsigned numCap) {
  StringRef bodyName = op.getBodyFunc();
  std::string trampolineName = (bodyName + "_trampoline").str();
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(trampolineName))
    return existing;

  MLIRContext *ctx = b.getContext();
  Type ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
  Type i32Ty = b.getI32Type();

  // Fixed trampoline signature:
  //   (state, frame, iter, cond, current_lc[], cap[], next_lc[]) -> i32.
  auto trampType = LLVM::LLVMFunctionType::get(
      i32Ty, {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy, ptrTy});

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  auto tramp = LLVM::LLVMFuncOp::create(b, loc, trampolineName, trampType,
                                        LLVM::Linkage::Internal);

  Block *entry = tramp.addEntryBlock(b);
  b.setInsertionPointToStart(entry);

  Value stateArg = entry->getArgument(0);
  Value frameArg = entry->getArgument(1);
  Value iterPtr = entry->getArgument(2);
  Value condPtr = entry->getArgument(3);
  Value lcArr = entry->getArgument(4);
  Value capArr = entry->getArgument(5);
  Value nextArr = entry->getArgument(6);

  // Convert each body func.func arg type through the LLVMTypeConverter to
  // get the per-arg LLVM struct (one struct per memref).  This is the
  // unexpanded form -- the expanded form (allocPtr, alignedPtr, offset,
  // sizes..., strides...) is produced by populateFuncToLLVMConversion
  // Patterns when the body itself is converted, and we emit it from each
  // struct via `expandMemRefStruct` below at the call site.
  SmallVector<Type> bodyArgTypes;
  for (Type t : bodyFuncFn.getArgumentTypes()) {
    SmallVector<Type, 1> converted;
    if (failed(typeConverter.convertType(t, converted)) || converted.empty()) {
      tramp.emitError("could not convert body arg type ") << t << " to LLVM";
      return {};
    }
    bodyArgTypes.append(converted.begin(), converted.end());
  }

  // Argument layout (the frame is appended so established carrier indexes do
  // not move):
  //   arg 0           : !hip.context           (lowered to !llvm.ptr)
  //   arg 1           : memref<i64>            (iter)
  //   arg 2           : memref<i1>             (cond_in)
  //   arg 3..3+numLC  : loop-carried memrefs   (v_in_i)
  //   ..+numCap       : captures               (cap_j)
  //   final arg        : !hip.loop_frame
  bool condIsPassthrough = op.getCondIsPassthrough();
  unsigned expectedArgCount = 4 + numLC + numCap;
  if (bodyArgTypes.size() != expectedArgCount) {
    tramp.emitError("body arg count mismatch: expected ")
        << expectedArgCount << " (4 + " << numLC << " + " << numCap << "), got "
        << bodyArgTypes.size();
    return {};
  }

  // Load each loop-carried / capture descriptor struct from its slot.
  // (We DON'T need iter/cond -- those are passed as raw ptrs from the
  // runtime and we emit their expanded fields directly.)
  auto loadDesc = [&](Value array, unsigned index, Type structTy) -> Value {
    Value idx = LLVM::ConstantOp::create(b, loc, b.getI64Type(),
                                         b.getI64IntegerAttr(index));
    Value slotPtr =
        LLVM::GEPOp::create(b, loc, ptrTy, ptrTy, array, ValueRange{idx});
    Value descPtr = LLVM::LoadOp::create(b, loc, ptrTy, slotPtr);
    return LLVM::LoadOp::create(b, loc, structTy, descPtr);
  };

  SmallVector<Value> lcDescs;
  for (unsigned i = 0; i < numLC; ++i)
    lcDescs.push_back(loadDesc(lcArr, i, bodyArgTypes[3 + i]));

  FailureOr<LLVM::LLVMFuncOp> setCurrentFn =
      LLVM::lookupOrCreateFn(b, module, "hipdnn_ep_loop_frame_set_current",
                             {ptrTy, i32Ty, ptrTy}, i32Ty);
  if (failed(setCurrentFn))
    return {};
  Value zero = LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(0));
  Value setCurrentStatus = zero;
  for (unsigned i = 0; i < numLC; ++i) {
    Value index = LLVM::ConstantOp::create(
        b, loc, i32Ty, b.getI32IntegerAttr(static_cast<int32_t>(i)));
    Value currentData = LLVM::ExtractValueOp::create(
        b, loc, lcDescs[i], ArrayRef<int64_t>{kAlignedPtrIdx});
    Value status =
        LLVM::CallOp::create(b, loc, *setCurrentFn,
                             ValueRange{frameArg, index, currentData})
            .getResult();
    Value priorSucceeded = LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::eq,
                                                setCurrentStatus, zero);
    setCurrentStatus = LLVM::SelectOp::create(b, loc, priorSucceeded, status,
                                              setCurrentStatus);
  }

  // A malformed frame must not reach the body with partially installed
  // current pointers. Preserve the first setter failure and return it through
  // the trampoline's existing status result.
  Value setCurrentSucceeded = LLVM::ICmpOp::create(
      b, loc, LLVM::ICmpPredicate::eq, setCurrentStatus, zero);
  Block *setCurrentFailureBlock = tramp.addBlock();
  Block *prepareBodyBlock = tramp.addBlock();
  LLVM::CondBrOp::create(b, loc, setCurrentSucceeded, prepareBodyBlock,
                         setCurrentFailureBlock);
  b.setInsertionPointToStart(setCurrentFailureBlock);
  LLVM::ReturnOp::create(b, loc, setCurrentStatus);
  b.setInsertionPointToStart(prepareBodyBlock);

  SmallVector<Value> capDescs;
  for (unsigned j = 0; j < numCap; ++j)
    capDescs.push_back(loadDesc(capArr, j, bodyArgTypes[3 + numLC + j]));

  // Assemble call args for the body, EXPANDED form.
  // Order matches the outlined body: state, iter, cond, current carriers,
  // captures, frame.
  SmallVector<Value> bodyCallArgs;
  bodyCallArgs.push_back(stateArg);
  emitRank0DescriptorFields(b, loc, iterPtr, bodyCallArgs);
  emitRank0DescriptorFields(b, loc, condPtr, bodyCallArgs);
  for (Value d : lcDescs)
    expandMemRefStruct(b, loc, d, memrefRankFromStructType(d.getType()),
                       bodyCallArgs);
  for (Value d : capDescs)
    expandMemRefStruct(b, loc, d, memrefRankFromStructType(d.getType()),
                       bodyCallArgs);
  bodyCallArgs.push_back(frameArg);

  // Func-to-LLVM returns one memref descriptor directly, or a literal struct
  // containing one descriptor per source-level result.
  SmallVector<Type> bodyResultTypes;
  for (Type t : bodyFuncFn.getResultTypes()) {
    SmallVector<Type, 1> converted;
    if (failed(typeConverter.convertType(t, converted)) ||
        converted.size() != 1) {
      tramp.emitError("could not convert body result type ") << t;
      return {};
    }
    bodyResultTypes.push_back(converted.front());
  }
  unsigned carrierResultStart = condIsPassthrough ? 1u : 2u;
  if (bodyResultTypes.size() != carrierResultStart + numLC) {
    tramp.emitError("body result count mismatch for descriptor-return ABI");
    return {};
  }
  Type packedResultType =
      bodyResultTypes.size() == 1
          ? bodyResultTypes.front()
          : LLVM::LLVMStructType::getLiteral(ctx, bodyResultTypes);
  Value packed =
      LLVM::CallOp::create(b, loc, TypeRange{packedResultType},
                           FlatSymbolRefAttr::get(ctx, bodyName), bodyCallArgs)
          .getResult();
  auto getBodyResult = [&](unsigned index) -> Value {
    if (bodyResultTypes.size() == 1)
      return packed;
    return LLVM::ExtractValueOp::create(b, loc, packed,
                                        ArrayRef<int64_t>{index});
  };

  // Body status is result #0. Allocation failure is also recorded on the
  // frame by hip.loop_alloc. Neither path may publish a descriptor.
  Value bodyStatus = getBodyResult(0);
  FailureOr<LLVM::LLVMFuncOp> statusFn = LLVM::lookupOrCreateFn(
      b, module, "hipdnn_ep_loop_frame_status", {ptrTy}, i32Ty);
  if (failed(statusFn))
    return {};
  Value frameStatus =
      LLVM::CallOp::create(b, loc, *statusFn, ValueRange{frameArg}).getResult();
  Value bodySucceeded =
      LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::eq, bodyStatus, zero);
  Value frameSucceeded =
      LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::eq, frameStatus, zero);
  Value succeeded = LLVM::AndOp::create(b, loc, bodySucceeded, frameSucceeded);
  Value failureStatus =
      LLVM::SelectOp::create(b, loc, bodySucceeded, frameStatus, bodyStatus);
  Block *validatePublishBlock = tramp.addBlock();
  Block *failureBlock = tramp.addBlock();
  LLVM::CondBrOp::create(b, loc, succeeded, validatePublishBlock, failureBlock);

  b.setInsertionPointToStart(failureBlock);
  LLVM::ReturnOp::create(b, loc, failureStatus);

  b.setInsertionPointToStart(validatePublishBlock);
  FailureOr<LLVM::LLVMFuncOp> publishFn = LLVM::lookupOrCreateFn(
      b, module, "hipdnn_ep_loop_frame_publish", {ptrTy, i32Ty, ptrTy}, i32Ty);
  if (failed(publishFn))
    return {};
  Value publishCallStatus = zero;
  for (unsigned i = 0; i < numLC; ++i) {
    Value descriptor = getBodyResult(carrierResultStart + i);
    Value data = LLVM::ExtractValueOp::create(
        b, loc, descriptor, ArrayRef<int64_t>{kAlignedPtrIdx});
    Value index = LLVM::ConstantOp::create(
        b, loc, i32Ty, b.getI32IntegerAttr(static_cast<int32_t>(i)));
    Value status = LLVM::CallOp::create(b, loc, *publishFn,
                                        ValueRange{frameArg, index, data})
                       .getResult();
    Value priorSucceeded = LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::eq,
                                                publishCallStatus, zero);
    publishCallStatus = LLVM::SelectOp::create(b, loc, priorSucceeded, status,
                                               publishCallStatus);
  }
  Value framePublishStatus =
      LLVM::CallOp::create(b, loc, *statusFn, ValueRange{frameArg}).getResult();
  Value publishCallsSucceeded = LLVM::ICmpOp::create(
      b, loc, LLVM::ICmpPredicate::eq, publishCallStatus, zero);
  Value publishStatus = LLVM::SelectOp::create(
      b, loc, publishCallsSucceeded, framePublishStatus, publishCallStatus);
  Value publishSucceeded = LLVM::ICmpOp::create(b, loc, LLVM::ICmpPredicate::eq,
                                                publishStatus, zero);
  Block *publishBlock = tramp.addBlock();
  Block *publishFailureBlock = tramp.addBlock();
  LLVM::CondBrOp::create(b, loc, publishSucceeded, publishBlock,
                         publishFailureBlock);
  b.setInsertionPointToStart(publishFailureBlock);
  LLVM::ReturnOp::create(b, loc, publishStatus);

  b.setInsertionPointToStart(publishBlock);
  Value callbackStatus = zero;
  if (!condIsPassthrough) {
    Value condDesc = getBodyResult(1);
    Value condSrc = LLVM::ExtractValueOp::create(
        b, loc, condDesc, ArrayRef<int64_t>{kAlignedPtrIdx});
    Value oneByte = LLVM::ConstantOp::create(b, loc, b.getI64Type(),
                                             b.getI64IntegerAttr(1));
    FailureOr<LLVM::LLVMFuncOp> copyFn =
        LLVM::lookupOrCreateFn(b, module, kWrapHipMemcpyAsync,
                               {ptrTy, ptrTy, ptrTy, b.getI64Type()}, i32Ty);
    if (failed(copyFn))
      return {};
    callbackStatus =
        LLVM::CallOp::create(b, loc, *copyFn,
                             ValueRange{stateArg, condPtr, condSrc, oneByte})
            .getResult();
  }

  // Write the complete next descriptor set. Runtime swaps current/next only
  // when callbackStatus is zero, so a failed body cannot partially publish.
  for (unsigned i = 0; i < numLC; ++i) {
    Value index = LLVM::ConstantOp::create(b, loc, b.getI64Type(),
                                           b.getI64IntegerAttr(i));
    Value slotPtr =
        LLVM::GEPOp::create(b, loc, ptrTy, ptrTy, nextArr, ValueRange{index});
    Value descPtr = LLVM::LoadOp::create(b, loc, ptrTy, slotPtr);
    LLVM::StoreOp::create(b, loc, getBodyResult(carrierResultStart + i),
                          descPtr);
  }

  LLVM::ReturnOp::create(b, loc, callbackStatus);

  return tramp;
}

//===----------------------------------------------------------------------===//
// hip.loop -> hipdnn_ep_run_{counted_,}loop  +  trampoline
//===----------------------------------------------------------------------===//

struct LoopOpLowering : public ConvertOpToLLVMPattern<LoopOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(LoopOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    Type i1Ty = rewriter.getI1Type();

    // Trampoline construction needs the body's unconverted func::FuncOp
    // signature (we run the LLVMTypeConverter on the memref args to get the
    // per-arg descriptor struct types).  If the body has already been
    // converted to LLVMFuncOp by an earlier worklist visit in the same
    // partial-conversion call, our index math no longer holds -- the body
    // args have been expanded into per-field scalars/ptrs.  Fail
    // explicitly so it's diagnosable instead of producing a broken
    // trampoline; in practice this never trips because the conversion
    // driver visits the LoopOp before recursing into its body func.
    StringRef bodyName = op.getBodyFunc();
    auto bodyFuncFn = module.lookupSymbol<func::FuncOp>(bodyName);
    std::string trampolineName = (bodyName + "_trampoline").str();
    auto precreatedTrampoline =
        module.lookupSymbol<LLVM::LLVMFuncOp>(trampolineName);
    if (!bodyFuncFn && !precreatedTrampoline) {
      if (module.lookupSymbol<LLVM::LLVMFuncOp>(bodyName))
        return op.emitOpError("body func '")
               << bodyName
               << "' was converted to LLVMFuncOp before LoopOpLowering ran; "
                  "trampoline construction requires the unconverted "
                  "func::FuncOp signature";
      return op.emitOpError("body func '") << bodyName << "' not found";
    }

    unsigned numLC = adaptor.getVInit().size();
    unsigned numCap = adaptor.getCaptures().size();

    // Generate (or fetch) the trampoline.  Inserted at module scope.
    LLVM::LLVMFuncOp tramp;
    {
      if (precreatedTrampoline) {
        tramp = precreatedTrampoline;
      } else {
        OpBuilder modBuilder(module.getBody(), module.getBody()->end());
        tramp = createOrGetTrampoline(modBuilder, module, loc, op, bodyFuncFn,
                                      *getTypeConverter(), numLC, numCap);
      }
      if (!tramp)
        return failure();
    }

    // Reference the trampoline by address-of.
    Value trampPtr =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, tramp.getSymNameAttr());

    // Build two stack arrays of descriptor pointers. The initial array owns
    // copies of v_init descriptors; the scratch array provides one descriptor
    // slot per carrier for the callback's atomic next set. Runtime swaps these
    // arrays after successful iterations and returns the final array pointer.
    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                            rewriter.getI64IntegerAttr(1));
    auto buildLCArray = [&](bool initialize) -> Value {
      Type arrTy = LLVM::LLVMArrayType::get(ptrTy, numLC == 0 ? 1 : numLC);
      Value array = LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrTy, oneI64,
                                           /*alignment=*/8);
      for (unsigned i = 0; i < numLC; ++i) {
        Value desc = adaptor.getVInit()[i];
        Value descSlot = LLVM::AllocaOp::create(
            rewriter, loc, ptrTy, desc.getType(), oneI64, /*alignment=*/8);
        if (initialize)
          LLVM::StoreOp::create(rewriter, loc, desc, descSlot);
        Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                rewriter.getI64IntegerAttr(i));
        Value slotPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrTy, array,
                                            ValueRange{idxVal});
        LLVM::StoreOp::create(rewriter, loc, descSlot, slotPtr);
      }
      return array;
    };
    Value lcArrayPtr = buildLCArray(/*initialize=*/true);
    Value scratchArrayPtrA = buildLCArray(/*initialize=*/false);
    Value scratchArrayPtrB = buildLCArray(/*initialize=*/false);
    Value finalArrayOut = LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy,
                                                 oneI64, /*alignment=*/8);
    LLVM::StoreOp::create(rewriter, loc, lcArrayPtr, finalArrayOut);
    Value frameOut = LLVM::AllocaOp::create(rewriter, loc, ptrTy, ptrTy, oneI64,
                                            /*alignment=*/8);
    Value nullPtr = LLVM::ZeroOp::create(rewriter, loc, ptrTy);
    LLVM::StoreOp::create(rewriter, loc, nullPtr, frameOut);

    // Same for captures.  If numCap is 0, allocate a tiny placeholder so
    // the runtime sees a non-null pointer (and num_cap=0).
    Value capArrayPtr;
    {
      Type arrTy = LLVM::LLVMArrayType::get(ptrTy, numCap == 0 ? 1 : numCap);
      capArrayPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrTy, oneI64,
                                           /*alignment=*/8);
      for (unsigned j = 0; j < numCap; ++j) {
        Value desc = adaptor.getCaptures()[j];
        Value descSlot = LLVM::AllocaOp::create(
            rewriter, loc, ptrTy, desc.getType(), oneI64, /*alignment=*/8);
        LLVM::StoreOp::create(rewriter, loc, desc, descSlot);
        Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                rewriter.getI64IntegerAttr(j));
        Value slotPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrTy,
                                            capArrayPtr, ValueRange{idxVal});
        LLVM::StoreOp::create(rewriter, loc, descSlot, slotPtr);
      }
    }

    // Convert max_trip_count (which is `index`) to i64.
    Value maxTripCountI64 = adaptor.getMaxTripCount();
    if (!isa<IntegerType>(maxTripCountI64.getType()) ||
        cast<IntegerType>(maxTripCountI64.getType()).getWidth() != 64) {
      // Index type lowers to i64 via the LLVMTypeConverter; if it's
      // something else (shouldn't happen), promote.
      maxTripCountI64 =
          arith::IndexCastUIOp::create(rewriter, loc, i64Ty, maxTripCountI64);
    }

    // cond_init is always i1 (we made it so in the outlining pass).
    Value condInit = adaptor.getCondInit();
    if (!condInit)
      condInit = LLVM::ConstantOp::create(rewriter, loc, i1Ty,
                                          rewriter.getBoolAttr(true));

    Value numLCConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(numLC)));
    Value numCapConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(numCap)));

    // Pick fast vs slow runtime symbol.
    StringRef runtimeSymbol =
        op.getCondIsPassthrough() ? kRunCountedLoop : kRunLoop;

    // Runtime signature:
    //   i32 (*)(state*, body_fn*, i64 M, i1 cond_init, i32 num_lc,
    //           i32 num_cap, ptr* initial, ptr* scratch, ptr* captures,
    //           ptr** final_array)
    SmallVector<Type, 13> paramTypes = {ptrTy, ptrTy, i64Ty, i1Ty,  i32Ty,
                                        i32Ty, ptrTy, ptrTy, ptrTy, ptrTy,
                                        ptrTy, ptrTy, ptrTy};
    FailureOr<LLVM::LLVMFuncOp> runLoopFn = LLVM::lookupOrCreateFn(
        rewriter, module, runtimeSymbol, paramTypes, i32Ty);
    if (failed(runLoopFn))
      return failure();

    Value parentFrame =
        adaptor.getParentFrame() ? adaptor.getParentFrame() : nullPtr;
    SmallVector<Value, 13> args = {
        adaptor.getCtx(), trampPtr,    maxTripCountI64, condInit,
        numLCConst,       numCapConst, lcArrayPtr,      scratchArrayPtrA,
        scratchArrayPtrB, capArrayPtr, parentFrame,     finalArrayOut,
        frameOut};
    Value runStatus =
        LLVM::CallOp::create(rewriter, loc, *runLoopFn, args).getResult();
    Value zeroStatus = LLVM::ConstantOp::create(rewriter, loc, i32Ty,
                                                rewriter.getI32IntegerAttr(0));
    Value runSucceeded = LLVM::ICmpOp::create(
        rewriter, loc, LLVM::ICmpPredicate::eq, runStatus, zeroStatus);

    Value returnedArray =
        LLVM::LoadOp::create(rewriter, loc, ptrTy, finalArrayOut);
    Value finalArray = LLVM::SelectOp::create(rewriter, loc, runSucceeded,
                                              returnedArray, lcArrayPtr);
    SmallVector<Value> finalDescriptors;
    finalDescriptors.reserve(numLC);
    for (unsigned i = 0; i < numLC; ++i) {
      Value index = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                             rewriter.getI64IntegerAttr(i));
      Value slot = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrTy, finalArray,
                                       ValueRange{index});
      Value descPtr = LLVM::LoadOp::create(rewriter, loc, ptrTy, slot);
      finalDescriptors.push_back(LLVM::LoadOp::create(
          rewriter, loc, adaptor.getVInit()[i].getType(), descPtr));
    }
    Value returnedFrame = LLVM::LoadOp::create(rewriter, loc, ptrTy, frameOut);
    Value safeFrame = LLVM::SelectOp::create(rewriter, loc, runSucceeded,
                                             returnedFrame, nullPtr);
    finalDescriptors.push_back(safeFrame);
    rewriter.replaceOp(op, finalDescriptors);
    return success();
  }
};

} // namespace

LogicalResult precreateLoopTrampolines(ModuleOp module,
                                       const LLVMTypeConverter &converter) {
  LogicalResult result = success();
  module.walk([&](LoopOp loop) {
    if (failed(result))
      return;
    auto body = module.lookupSymbol<func::FuncOp>(loop.getBodyFuncAttr());
    if (!body) {
      loop.emitOpError("body func must exist before trampoline precreation");
      result = failure();
      return;
    }
    OpBuilder builder(module.getBody(), module.getBody()->end());
    LLVM::LLVMFuncOp trampoline = createOrGetTrampoline(
        builder, module, loop.getLoc(), loop, body, converter,
        loop.getNumLoopCarried(), loop.getCaptures().size());
    if (!trampoline)
      result = failure();
  });
  return result;
}

void populateLoopLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<LoopOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
