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
//   Receives  : (state*, iter_dev*, cond_dev*, lc_descs*, cap_descs*)
//   Builds    : rank-0 memref descriptors for iter and cond from the raw
//               device pointers; loads each loop-carried / capture
//               descriptor struct from its slot in the *_descs array.
//   Calls body: (state, iter, cond_in, v_in..., captures...,
//                [cond_out if !cond_is_passthrough], v_out...).
// Aliasing invariant: each v_in_i shares its buffer with v_out_i, and
// cond_in shares with cond_out when not passthrough. Safe under v1
// single-pass-per-kernel body semantics; the runtime driver enforces
// non-nesting so the shared per-state iter/cond buffers cannot race.
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
///   * Receives (state, iter_dev, cond_dev, lc_descs, cap_descs).
///   * Builds rank-0 memref descriptors for iter and cond from the raw ptrs.
///   * Loads the loop-carried and captured memref descriptors from the
///     pointer arrays (each entry is a pointer to a descriptor struct
///     allocated by the calling main_graph function).
///   * Calls the outlined body with the body's positional signature
///     (state, iter, cond_in, v_in..., captures..., cond_out, v_out...).
///   * Returns 0.
///
/// Aliasing invariant: cond_in and cond_out share the same buffer, and
/// each v_in_i shares its buffer with the corresponding v_out_i.  This
/// matches the runtime driver's per-iter buffer management and the body's
/// single-pass-per-kernel safety (see file header).
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

  // Fixed trampoline signature: (state, iter, cond, lc[], cap[]) -> i32.
  auto trampType =
      LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, ptrTy, ptrTy, ptrTy});

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  auto tramp = LLVM::LLVMFuncOp::create(b, loc, trampolineName, trampType,
                                        LLVM::Linkage::Internal);

  Block *entry = tramp.addEntryBlock(b);
  b.setInsertionPointToStart(entry);

  Value stateArg = entry->getArgument(0);
  Value iterPtr = entry->getArgument(1);
  Value condPtr = entry->getArgument(2);
  Value lcArr = entry->getArgument(3);
  Value capArr = entry->getArgument(4);

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

  // Layout (matches outlining pass output):
  //   arg 0           : !hip.context           (lowered to !llvm.ptr)
  //   arg 1           : memref<i64>            (iter)
  //   arg 2           : memref<i1>             (cond_in)
  //   arg 3..3+numLC  : loop-carried memrefs   (v_in_i)
  //   ..+numCap       : captures               (cap_j)
  //   [if !cond_is_passthrough]
  //     ..            : memref<i1>             (cond_out, bufferize.result)
  //   ..              : loop-carried memrefs   (v_out_i, bufferize.result)
  bool condIsPassthrough = op.getCondIsPassthrough();
  unsigned expectedArgCount =
      3 + 2 * numLC + numCap + (condIsPassthrough ? 0 : 1);
  if (bodyArgTypes.size() != expectedArgCount) {
    tramp.emitError("body arg count mismatch: expected ")
        << expectedArgCount << " (3 + 2*" << numLC << " + " << numCap << " + "
        << (condIsPassthrough ? 0 : 1) << "), got " << bodyArgTypes.size();
    return {};
  }
  // After populateFuncToLLVMConversionPatterns, each memref arg in
  // bodyArgTypes is presented as a single struct type (the per-rank
  // descriptor type).  At the call site we must emit one ARG PER FIELD
  // because the func-to-llvm lowering rewrites the body's signature to
  // the expanded form (allocPtr, alignedPtr, offset, sizes..., strides...)
  // -- the call has to match.
  //
  // Locate body arg positions:
  //   arg 0           : !hip.context           (lowered to !llvm.ptr)
  //   arg 1           : memref<i64>            (iter)
  //   arg 2           : memref<i1>             (cond_in)
  //   arg 3..3+numLC  : loop-carried memrefs   (v_in_i)
  //   ..+numCap       : captures               (cap_j)
  //   [if !cond_is_passthrough]
  //     ..            : memref<i1>             (cond_out, bufferize.result)
  //   ..              : loop-carried memrefs   (v_out_i, bufferize.result)

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

  SmallVector<Value> capDescs;
  for (unsigned j = 0; j < numCap; ++j)
    capDescs.push_back(loadDesc(capArr, j, bodyArgTypes[3 + numLC + j]));

  // Assemble call args for the body, EXPANDED form.
  // Order must match the outlining-pass output post-BufferResultsToOutParams:
  //   state, iter, cond_in, v_in_0..numLC-1, cap_0..numCap-1,
  //   [cond_out (only when !cond_is_passthrough)],
  //   v_out_0..numLC-1
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
  if (!condIsPassthrough) {
    // cond_out aliased with cond_in (same buffer, single-pass kernel safety).
    emitRank0DescriptorFields(b, loc, condPtr, bodyCallArgs);
  }
  // v_out_i == v_in_i (same buffer, single-pass kernel safety).
  for (Value d : lcDescs)
    expandMemRefStruct(b, loc, d, memrefRankFromStructType(d.getType()),
                       bodyCallArgs);

  // Call the body by symbol name.  The verifier checks this matches once
  // the body has been converted to LLVMFuncOp (which it will be by the
  // end of the same partial-conversion pass, via
  // populateFuncToLLVMConversionPatterns).
  LLVM::CallOp::create(b, loc, /*results=*/TypeRange{},
                       FlatSymbolRefAttr::get(ctx, bodyName), bodyCallArgs);

  Value zero = LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(b, loc, ValueRange{zero});

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
    if (!bodyFuncFn) {
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
      OpBuilder modBuilder(module.getBody(), module.getBody()->end());
      tramp = createOrGetTrampoline(modBuilder, module, loc, op, bodyFuncFn,
                                    *getTypeConverter(), numLC, numCap);
      if (!tramp)
        return failure();
    }

    // Reference the trampoline by address-of.
    Value trampPtr =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, tramp.getSymNameAttr());

    // Build a stack array of descriptor pointers for loop-carried operands.
    //   ptr[numLC]: each entry is alloca'd to hold the corresponding
    //   memref descriptor struct, with the descriptor's value stored
    //   into it.
    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                            rewriter.getI64IntegerAttr(1));
    Value lcArrayPtr;
    {
      Type arrTy = LLVM::LLVMArrayType::get(ptrTy, numLC == 0 ? 1 : numLC);
      lcArrayPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrTy, oneI64,
                                          /*alignment=*/8);
      for (unsigned i = 0; i < numLC; ++i) {
        Value desc = adaptor.getVInit()[i];
        Value descSlot = LLVM::AllocaOp::create(
            rewriter, loc, ptrTy, desc.getType(), oneI64, /*alignment=*/8);
        LLVM::StoreOp::create(rewriter, loc, desc, descSlot);
        Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                rewriter.getI64IntegerAttr(i));
        Value slotPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrTy,
                                            lcArrayPtr, ValueRange{idxVal});
        LLVM::StoreOp::create(rewriter, loc, descSlot, slotPtr);
      }
    }

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
    //   i32 (*)(state*, body_fn*, i64 M, i1 cond_init,
    //           i32 num_lc, i32 num_cap, ptr* lc_descs, ptr* cap_descs)
    SmallVector<Type, 8> paramTypes = {ptrTy, ptrTy, i64Ty, i1Ty,
                                       i32Ty, i32Ty, ptrTy, ptrTy};
    FailureOr<LLVM::LLVMFuncOp> runLoopFn = LLVM::lookupOrCreateFn(
        rewriter, module, runtimeSymbol, paramTypes, i32Ty);
    if (failed(runLoopFn))
      return failure();

    SmallVector<Value, 8> args = {adaptor.getCtx(), trampPtr,   maxTripCountI64,
                                  condInit,         numLCConst, numCapConst,
                                  lcArrayPtr,       capArrayPtr};
    LLVM::CallOp::create(rewriter, loc, *runLoopFn, args);

    // hip.loop has no LLVM-level results post-bufferization (loop-carried
    // results were converted to out-params).  Replace any remaining uses
    // (should be zero) and erase.
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateLoopLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<LoopOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
