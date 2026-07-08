/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- IfLowering.cpp - Lower hip.if to runtime driver calls --------------===//
//
// Lowers `hip.if` to a call into `hipdnn_ep_run_if` with per-branch LLVM
// trampolines that bridge the runtime's fixed-arity callback contract to
// each outlined body's variable-arity memref-descriptor signature.
//
//===----------------------------------------------------------------------===//

#include "HipToLLVMUtils.h"

#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace hip {
namespace {

constexpr const char *kRunIf = "hipdnn_ep_run_if";

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

static int64_t memrefRankFromStructType(Type structTy) {
  auto st = dyn_cast<LLVM::LLVMStructType>(structTy);
  if (!st)
    return 0;
  if (st.getBody().size() < 4)
    return 0;
  auto arrTy = dyn_cast<LLVM::LLVMArrayType>(st.getBody()[kSizesIdx]);
  return arrTy ? static_cast<int64_t>(arrTy.getNumElements()) : 0;
}

static LLVM::LLVMFuncOp
createOrGetIfTrampoline(OpBuilder &b, ModuleOp module, Location loc,
                        StringRef bodyName, func::FuncOp bodyFuncFn,
                        const LLVMTypeConverter &typeConverter,
                        unsigned numCap, unsigned numOut) {
  std::string trampolineName = (bodyName + "_trampoline").str();
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(trampolineName))
    return existing;

  MLIRContext *ctx = b.getContext();
  Type ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
  Type i32Ty = b.getI32Type();

  auto trampType =
      LLVM::LLVMFunctionType::get(i32Ty, {ptrTy, ptrTy, ptrTy});

  OpBuilder::InsertionGuard guard(b);
  b.setInsertionPointToEnd(module.getBody());
  auto tramp = LLVM::LLVMFuncOp::create(b, loc, trampolineName, trampType,
                                        LLVM::Linkage::Internal);

  Block *entry = tramp.addEntryBlock(b);
  b.setInsertionPointToStart(entry);

  Value stateArg = entry->getArgument(0);
  Value outArr = entry->getArgument(1);
  Value capArr = entry->getArgument(2);

  SmallVector<Type> bodyArgTypes;
  for (Type t : bodyFuncFn.getArgumentTypes()) {
    SmallVector<Type, 1> converted;
    if (failed(typeConverter.convertType(t, converted)) || converted.empty()) {
      tramp.emitError("could not convert body arg type ") << t << " to LLVM";
      return {};
    }
    bodyArgTypes.append(converted.begin(), converted.end());
  }

  unsigned expectedArgCount = 1 + numCap + numOut;
  if (bodyArgTypes.size() != expectedArgCount) {
    tramp.emitError("body arg count mismatch: expected ")
        << expectedArgCount << " (1 + " << numCap << " + " << numOut
        << "), got " << bodyArgTypes.size();
    return {};
  }

  auto loadDesc = [&](Value array, unsigned index, Type structTy) -> Value {
    Value idx = LLVM::ConstantOp::create(b, loc, b.getI64Type(),
                                         b.getI64IntegerAttr(index));
    Value slotPtr =
        LLVM::GEPOp::create(b, loc, ptrTy, ptrTy, array, ValueRange{idx});
    Value descPtr = LLVM::LoadOp::create(b, loc, ptrTy, slotPtr);
    return LLVM::LoadOp::create(b, loc, structTy, descPtr);
  };

  SmallVector<Value> capDescs;
  for (unsigned j = 0; j < numCap; ++j)
    capDescs.push_back(loadDesc(capArr, j, bodyArgTypes[1 + j]));

  SmallVector<Value> outDescs;
  for (unsigned i = 0; i < numOut; ++i)
    outDescs.push_back(loadDesc(outArr, i, bodyArgTypes[1 + numCap + i]));

  SmallVector<Value> bodyCallArgs;
  bodyCallArgs.push_back(stateArg);
  for (Value d : capDescs)
    expandMemRefStruct(b, loc, d, memrefRankFromStructType(d.getType()),
                       bodyCallArgs);
  for (Value d : outDescs)
    expandMemRefStruct(b, loc, d, memrefRankFromStructType(d.getType()),
                       bodyCallArgs);

  LLVM::CallOp::create(b, loc, TypeRange{},
                       FlatSymbolRefAttr::get(ctx, bodyName), bodyCallArgs);

  Value zero = LLVM::ConstantOp::create(b, loc, i32Ty, b.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(b, loc, ValueRange{zero});
  return tramp;
}

struct IfOpLowering : public ConvertOpToLLVMPattern<IfOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    MLIRContext *ctx = rewriter.getContext();
    Type ptrTy = LLVM::LLVMPointerType::get(ctx, 0);
    Type i32Ty = rewriter.getI32Type();
    Type i64Ty = rewriter.getI64Type();
    Type i1Ty = rewriter.getI1Type();

    StringRef thenName = op.getThenFunc();
    StringRef elseName = op.getElseFunc();
    auto thenFuncFn = module.lookupSymbol<func::FuncOp>(thenName);
    auto elseFuncFn = module.lookupSymbol<func::FuncOp>(elseName);
    if (!thenFuncFn || !elseFuncFn)
      return op.emitOpError("outlined then/else func not found");

    unsigned numOut = adaptor.getOInit().size();
    unsigned numCap = adaptor.getCaptures().size();

    LLVM::LLVMFuncOp thenTramp;
    LLVM::LLVMFuncOp elseTramp;
    {
      OpBuilder modBuilder(module.getBody(), module.getBody()->end());
      thenTramp = createOrGetIfTrampoline(modBuilder, module, loc, thenName,
                                          thenFuncFn, *getTypeConverter(),
                                          numCap, numOut);
      elseTramp = createOrGetIfTrampoline(modBuilder, module, loc, elseName,
                                          elseFuncFn, *getTypeConverter(),
                                          numCap, numOut);
      if (!thenTramp || !elseTramp)
        return failure();
    }

    Value thenTrampPtr =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, thenTramp.getSymNameAttr());
    Value elseTrampPtr =
        LLVM::AddressOfOp::create(rewriter, loc, ptrTy, elseTramp.getSymNameAttr());

    Value oneI64 = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                            rewriter.getI64IntegerAttr(1));

    auto buildDescArray = [&](ValueRange descs) -> Value {
      unsigned n = descs.size();
      Type arrTy = LLVM::LLVMArrayType::get(ptrTy, n == 0 ? 1 : n);
      Value arrPtr = LLVM::AllocaOp::create(rewriter, loc, ptrTy, arrTy, oneI64,
                                            /*alignment=*/8);
      for (unsigned i = 0; i < n; ++i) {
        Value desc = descs[i];
        Value descSlot = LLVM::AllocaOp::create(rewriter, loc, ptrTy, desc.getType(),
                                              oneI64, /*alignment=*/8);
        LLVM::StoreOp::create(rewriter, loc, desc, descSlot);
        Value idxVal = LLVM::ConstantOp::create(rewriter, loc, i64Ty,
                                                rewriter.getI64IntegerAttr(i));
        Value slotPtr = LLVM::GEPOp::create(rewriter, loc, ptrTy, ptrTy, arrPtr,
                                            ValueRange{idxVal});
        LLVM::StoreOp::create(rewriter, loc, descSlot, slotPtr);
      }
      return arrPtr;
    };

    Value outArrayPtr = buildDescArray(adaptor.getOInit());
    Value capArrayPtr = buildDescArray(adaptor.getCaptures());

    Value cond = adaptor.getCond();
    Value numOutConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(numOut)));
    Value numCapConst = LLVM::ConstantOp::create(
        rewriter, loc, i32Ty,
        rewriter.getI32IntegerAttr(static_cast<int32_t>(numCap)));

    SmallVector<Type, 8> paramTypes = {ptrTy, i1Ty, ptrTy, ptrTy,
                                       i32Ty, i32Ty, ptrTy, ptrTy};
    FailureOr<LLVM::LLVMFuncOp> runIfFn = LLVM::lookupOrCreateFn(
        rewriter, module, kRunIf, paramTypes, i32Ty);
    if (failed(runIfFn))
      return failure();

    SmallVector<Value, 8> args = {
        adaptor.getCtx(),      cond,          thenTrampPtr, elseTrampPtr,
        numOutConst,           numCapConst,   outArrayPtr,  capArrayPtr};
    LLVM::CallOp::create(rewriter, loc, *runIfFn, args);
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateIfLoweringPatterns(const LLVMTypeConverter &converter,
                                RewritePatternSet &patterns) {
  patterns.add<IfOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
