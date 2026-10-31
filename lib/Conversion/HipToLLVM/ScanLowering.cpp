/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- ScanLowering.cpp - Lower hip.scan to a runtime-driven loop --------===//

#include "HipToLLVMUtils.h"

#include "mlir/Conversion/LLVMCommon/MemRefBuilder.h"
#include "mlir/IR/SymbolTable.h"

namespace mlir {
namespace hip {
namespace {

constexpr const char *kRunScan = "hipdnn_ep_run_scan";

static int64_t memrefRankFromStructType(Type structTy) {
  auto st = dyn_cast<LLVM::LLVMStructType>(structTy);
  if (!st || st.getBody().size() < 4)
    return 0;
  auto array = dyn_cast<LLVM::LLVMArrayType>(st.getBody()[kSizesIdx]);
  return array ? static_cast<int64_t>(array.getNumElements()) : 0;
}

static void expandMemRefStruct(OpBuilder &builder, Location loc, Value desc,
                               int64_t rank,
                               SmallVectorImpl<Value> &arguments) {
  arguments.push_back(LLVM::ExtractValueOp::create(
      builder, loc, desc, ArrayRef<int64_t>{kAllocPtrIdx}));
  arguments.push_back(LLVM::ExtractValueOp::create(
      builder, loc, desc, ArrayRef<int64_t>{kAlignedPtrIdx}));
  arguments.push_back(LLVM::ExtractValueOp::create(
      builder, loc, desc, ArrayRef<int64_t>{kOffsetIdx}));
  for (int64_t i = 0; i < rank; ++i)
    arguments.push_back(LLVM::ExtractValueOp::create(
        builder, loc, desc, ArrayRef<int64_t>{kSizesIdx, i}));
  for (int64_t i = 0; i < rank; ++i)
    arguments.push_back(LLVM::ExtractValueOp::create(
        builder, loc, desc, ArrayRef<int64_t>{kStridesIdx, i}));
}

static Value loadDescriptor(OpBuilder &builder, Location loc, Value array,
                            unsigned index, Type descriptorType) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Value idx = LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                       builder.getI64IntegerAttr(index));
  Value slot = LLVM::GEPOp::create(builder, loc, ptrType, ptrType, array,
                                   ValueRange{idx});
  Value pointer = LLVM::LoadOp::create(builder, loc, ptrType, slot);
  return LLVM::LoadOp::create(builder, loc, descriptorType, pointer);
}

static Value makeAxisZeroSlice(OpBuilder &builder, Location loc, Value full,
                               Type sliceDescriptorType, Type elementType,
                               Value iteration) {
  MemRefDescriptor source(full);
  int64_t fullRank = memrefRankFromStructType(full.getType());
  int64_t sliceRank = memrefRankFromStructType(sliceDescriptorType);
  assert(fullRank == sliceRank + 1 && "Scan slice must drop exactly one dim");

  MemRefDescriptor slice =
      MemRefDescriptor::poison(builder, loc, sliceDescriptorType);
  slice.setAllocatedPtr(builder, loc, source.allocatedPtr(builder, loc));
  Value offset = source.offset(builder, loc);
  Value delta = LLVM::MulOp::create(builder, loc, iteration,
                                    source.stride(builder, loc, 0));
  Value dataOffset = LLVM::AddOp::create(builder, loc, offset, delta);
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Value data = LLVM::GEPOp::create(builder, loc, ptrType, elementType,
                                   source.alignedPtr(builder, loc),
                                   ValueRange{dataOffset});
  slice.setAlignedPtr(builder, loc, data);
  slice.setOffset(builder, loc,
                  LLVM::ConstantOp::create(builder, loc, builder.getI64Type(),
                                           builder.getI64IntegerAttr(0)));
  for (int64_t i = 0; i < sliceRank; ++i) {
    slice.setSize(builder, loc, i, source.size(builder, loc, i + 1));
    slice.setStride(builder, loc, i, source.stride(builder, loc, i + 1));
  }
  return slice;
}

static LLVM::LLVMFuncOp
createScanTrampoline(OpBuilder &builder, ModuleOp module, Location loc,
                     ScanOp op, func::FuncOp bodyFn,
                     const LLVMTypeConverter &typeConverter, unsigned numState,
                     unsigned numScanInputs, unsigned numCaptures,
                     unsigned numScanOutputs) {
  StringRef bodyName = op.getBodyFunc();
  std::string name = (bodyName + "_trampoline").str();
  if (auto existing = module.lookupSymbol<LLVM::LLVMFuncOp>(name))
    return existing;

  MLIRContext *ctx = builder.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(ctx);
  Type i64Type = builder.getI64Type();
  Type i32Type = builder.getI32Type();
  auto fnType = LLVM::LLVMFunctionType::get(
      i32Type, {ptrType, i64Type, ptrType, ptrType, ptrType, ptrType});

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToEnd(module.getBody());
  auto trampoline = LLVM::LLVMFuncOp::create(builder, loc, name, fnType,
                                             LLVM::Linkage::Internal);
  Block *entry = trampoline.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value state = entry->getArgument(0);
  Value iteration = entry->getArgument(1);
  Value stateArray = entry->getArgument(2);
  Value inputArray = entry->getArgument(3);
  Value captureArray = entry->getArgument(4);
  Value outputArray = entry->getArgument(5);

  SmallVector<Type> bodyArgTypes;
  for (Type type : bodyFn.getArgumentTypes()) {
    SmallVector<Type, 1> converted;
    if (failed(typeConverter.convertType(type, converted)) ||
        converted.size() != 1) {
      trampoline.emitError("could not convert Scan body argument type ")
          << type;
      return {};
    }
    bodyArgTypes.push_back(converted.front());
  }
  unsigned inputArgCount = 1 + numState + numScanInputs + numCaptures;
  unsigned expected = inputArgCount + numState + numState + numScanOutputs;
  if (bodyArgTypes.size() != expected) {
    trampoline.emitError(
        "Scan body out-param argument count mismatch: expected ")
        << expected << ", got " << bodyArgTypes.size();
    return {};
  }

  SmallVector<Value> stateDescs;
  for (unsigned i = 0; i < numState; ++i)
    stateDescs.push_back(
        loadDescriptor(builder, loc, stateArray, i, bodyArgTypes[1 + i]));

  SmallVector<Value> inputSlices;
  for (unsigned i = 0; i < numScanInputs; ++i) {
    Type sliceType = bodyArgTypes[1 + numState + i];
    SmallVector<Type, 1> fullTypes;
    if (failed(typeConverter.convertType(op.getScanInputs()[i].getType(),
                                         fullTypes)) ||
        fullTypes.size() != 1) {
      op.emitOpError("could not convert scan input #")
          << i << " type " << op.getScanInputs()[i].getType()
          << " to one LLVM descriptor type";
      return {};
    }
    Value full = loadDescriptor(builder, loc, inputArray, i, fullTypes.front());
    auto sliceMemRefType =
        dyn_cast<MemRefType>(bodyFn.getArgumentTypes()[1 + numState + i]);
    Type elementType =
        sliceMemRefType
            ? typeConverter.convertType(sliceMemRefType.getElementType())
            : Type{};
    if (!elementType) {
      op.emitOpError("could not convert scan input slice element type at #")
          << i;
      return {};
    }
    inputSlices.push_back(makeAxisZeroSlice(builder, loc, full, sliceType,
                                            elementType, iteration));
  }

  SmallVector<Value> captureDescs;
  for (unsigned i = 0; i < numCaptures; ++i)
    captureDescs.push_back(
        loadDescriptor(builder, loc, captureArray, i,
                       bodyArgTypes[1 + numState + numScanInputs + i]));

  SmallVector<Value> stateScratchDescs;
  for (unsigned i = 0; i < numState; ++i)
    stateScratchDescs.push_back(
        loadDescriptor(builder, loc, captureArray, numCaptures + i,
                       bodyArgTypes[inputArgCount + i]));

  SmallVector<Value> outputSlices;
  for (unsigned i = 0; i < numScanOutputs; ++i) {
    unsigned outputArgIndex = inputArgCount + numState + numState + i;
    Type sliceType = bodyArgTypes[outputArgIndex];
    SmallVector<Type, 1> fullTypes;
    if (failed(typeConverter.convertType(op.getInits()[numState + i].getType(),
                                         fullTypes)) ||
        fullTypes.size() != 1) {
      op.emitOpError("could not convert scan output init #")
          << i << " type " << op.getInits()[numState + i].getType()
          << " to one LLVM descriptor type";
      return {};
    }
    Value full =
        loadDescriptor(builder, loc, outputArray, i, fullTypes.front());
    auto sliceMemRefType =
        dyn_cast<MemRefType>(bodyFn.getArgumentTypes()[outputArgIndex]);
    Type elementType =
        sliceMemRefType
            ? typeConverter.convertType(sliceMemRefType.getElementType())
            : Type{};
    if (!elementType) {
      op.emitOpError("could not convert scan output slice element type at #")
          << i;
      return {};
    }
    outputSlices.push_back(makeAxisZeroSlice(builder, loc, full, sliceType,
                                             elementType, iteration));
  }

  SmallVector<Value> callArgs{state};
  for (Value desc : stateDescs)
    expandMemRefStruct(builder, loc, desc,
                       memrefRankFromStructType(desc.getType()), callArgs);
  for (Value desc : inputSlices)
    expandMemRefStruct(builder, loc, desc,
                       memrefRankFromStructType(desc.getType()), callArgs);
  for (Value desc : captureDescs)
    expandMemRefStruct(builder, loc, desc,
                       memrefRankFromStructType(desc.getType()), callArgs);
  for (Value desc : stateScratchDescs)
    expandMemRefStruct(builder, loc, desc,
                       memrefRankFromStructType(desc.getType()), callArgs);
  // State output buffers alias state input buffers.
  for (Value desc : stateDescs)
    expandMemRefStruct(builder, loc, desc,
                       memrefRankFromStructType(desc.getType()), callArgs);
  for (Value desc : outputSlices)
    expandMemRefStruct(builder, loc, desc,
                       memrefRankFromStructType(desc.getType()), callArgs);

  LLVM::CallOp::create(builder, loc, TypeRange{},
                       FlatSymbolRefAttr::get(ctx, bodyName), callArgs);
  Value zero = LLVM::ConstantOp::create(builder, loc, i32Type,
                                        builder.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(builder, loc, zero);
  return trampoline;
}

static Value buildDescriptorPointerArray(OpBuilder &builder, Location loc,
                                         ValueRange descriptors) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Type i64Type = builder.getI64Type();
  Value one = LLVM::ConstantOp::create(builder, loc, i64Type,
                                       builder.getI64IntegerAttr(1));
  Type arrayType = LLVM::LLVMArrayType::get(
      ptrType, descriptors.empty() ? 1 : descriptors.size());
  Value array = LLVM::AllocaOp::create(builder, loc, ptrType, arrayType, one,
                                       /*alignment=*/8);
  for (auto [i, descriptor] : llvm::enumerate(descriptors)) {
    Value storage =
        LLVM::AllocaOp::create(builder, loc, ptrType, descriptor.getType(), one,
                               /*alignment=*/8);
    LLVM::StoreOp::create(builder, loc, descriptor, storage);
    Value idx = LLVM::ConstantOp::create(builder, loc, i64Type,
                                         builder.getI64IntegerAttr(i));
    Value slot = LLVM::GEPOp::create(builder, loc, ptrType, ptrType, array,
                                     ValueRange{idx});
    LLVM::StoreOp::create(builder, loc, storage, slot);
  }
  return array;
}

struct ScanOpLowering : public ConvertOpToLLVMPattern<ScanOp> {
  using ConvertOpToLLVMPattern::ConvertOpToLLVMPattern;

  LogicalResult
  matchAndRewrite(ScanOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    ModuleOp module = op->getParentOfType<ModuleOp>();
    Type ptrType = getPtrType();
    Type i64Type = rewriter.getI64Type();
    Type i32Type = rewriter.getI32Type();

    auto bodyFn = module.lookupSymbol<func::FuncOp>(op.getBodyFuncAttr());
    if (!bodyFn)
      return op.emitOpError("unconverted Scan body function not found");

    unsigned numState = op.getNumStateVariables();
    unsigned numInputs = op.getNumScanInputs();
    unsigned numOutputs = op.getNumScanOutputs();
    unsigned numCaptures = op.getCaptures().size();
    if (op.getStateScratch().size() != numState)
      return op.emitOpError("expected one state scratch buffer per state");
    LLVM::LLVMFuncOp trampoline = createScanTrampoline(
        rewriter, module, loc, op, bodyFn, *getTypeConverter(), numState,
        numInputs, numCaptures, numOutputs);
    if (!trampoline)
      return failure();

    Value trampolinePtr = LLVM::AddressOfOp::create(
        rewriter, loc, ptrType, trampoline.getSymNameAttr());
    Value stateArray = buildDescriptorPointerArray(
        rewriter, loc, adaptor.getInits().take_front(numState));
    Value inputArray =
        buildDescriptorPointerArray(rewriter, loc, adaptor.getScanInputs());
    SmallVector<Value> capturesAndScratch(adaptor.getCaptures());
    capturesAndScratch.append(adaptor.getStateScratch().begin(),
                              adaptor.getStateScratch().end());
    Value captureArray =
        buildDescriptorPointerArray(rewriter, loc, capturesAndScratch);
    Value outputArray = buildDescriptorPointerArray(
        rewriter, loc, adaptor.getInits().drop_front(numState));

    auto firstInputType =
        cast<MemRefType>(op.getScanInputs().front().getType());
    Value tripCount = getMemRefDimSize(
        firstInputType, 0, adaptor.getScanInputs().front(), rewriter, loc);

    SmallVector<Type> params = {ptrType, ptrType, i64Type, ptrType,
                                ptrType, ptrType, ptrType};
    FailureOr<LLVM::LLVMFuncOp> runFn =
        LLVM::lookupOrCreateFn(rewriter, module, kRunScan, params, i32Type);
    if (failed(runFn))
      return failure();
    LLVM::CallOp::create(rewriter, loc, *runFn,
                         ValueRange{adaptor.getCtx(), trampolinePtr, tripCount,
                                    stateArray, inputArray, captureArray,
                                    outputArray});
    rewriter.eraseOp(op);
    return success();
  }
};

} // namespace

void populateScanLoweringPatterns(const LLVMTypeConverter &converter,
                                  RewritePatternSet &patterns) {
  patterns.add<ScanOpLowering>(converter);
}

} // namespace hip
} // namespace mlir
