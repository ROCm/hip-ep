/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MainGraphAbiPass.cpp - Wrap @main_graph for the runtime -----------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMTypes.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"

#include <numeric>

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_MAINGRAPHABIPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Packed LLVM memref: {allocatedPtr, alignedPtr, offset, sizes[rank],
// strides[rank]}.
constexpr int64_t kAllocPtrIdx = 0;
constexpr int64_t kAlignedPtrIdx = 1;
constexpr int64_t kOffsetIdx = 2;
constexpr int64_t kSizesIdx = 3;
constexpr int64_t kStridesIdx = 4;

// convert-to-llvm turns one memref into 3 + 2*rank scalar args.
//   rank 1  ->  alloc, aligned, offset, size[0], stride[0]
int64_t scalarParamsForRank(int64_t rank) { return 3 + 2 * rank; }

// Includes the leading state pointer.
//   ranks {2, 1}  ->  1 + 7 + 5 = 13
int64_t scalarParamCount(ArrayRef<int64_t> ranks) {
  return std::accumulate(ranks.begin(), ranks.end(), int64_t{1},
                         [](int64_t total, int64_t rank) {
                           return total + scalarParamsForRank(rank);
                         });
}

// Packed memref struct for one input. Rank 1 in addrspace 1:
//   !llvm.struct<(ptr<1>, ptr<1>, i64, array<1xi64>, array<1xi64>)>
Type descriptorType(OpBuilder &builder, int64_t rank, unsigned addressSpace) {
  MLIRContext *ctx = builder.getContext();
  Type ptrType = LLVM::LLVMPointerType::get(ctx, addressSpace);
  Type i64Type = builder.getI64Type();
  Type extentArray = LLVM::LLVMArrayType::get(i64Type, rank);
  return LLVM::LLVMStructType::getLiteral(
      ctx, {ptrType, ptrType, i64Type, extentArray, extentArray});
}

// Reads each input's rank from hipdnn.input_shapes.
//   [array<i64: -1, 4>, array<i64: 8>]  ->  {2, 1}
FailureOr<SmallVector<int64_t>> collectInputRanks(ArrayAttr inputShapes,
                                                  LLVM::LLVMFuncOp graph) {
  auto invalidShape = llvm::find_if(inputShapes, [](Attribute shape) {
    return !isa<DenseI64ArrayAttr>(shape);
  });
  if (invalidShape != inputShapes.end()) {
    graph.emitError("hipdnn.input_shapes entries must be i64 arrays");
    return failure();
  }

  return llvm::map_to_vector(inputShapes.getAsRange<DenseI64ArrayAttr>(),
                             [](DenseI64ArrayAttr extents) {
                               return static_cast<int64_t>(extents.size());
                             });
}

// Pulls fields out in the same order convert-to-llvm uses for memref args.
//   %d : !llvm.struct<(ptr, ptr, i64, array<1xi64>, array<1xi64>)>
//     ->  %d[0], %d[1], %d[2], %d[3, 0], %d[4, 0]
SmallVector<Value> unpackDescriptor(OpBuilder &builder, Location loc,
                                    Value descriptor, int64_t rank) {
  auto extract = [&](ArrayRef<int64_t> path) -> Value {
    return LLVM::ExtractValueOp::create(builder, loc, descriptor, path);
  };
  auto extractExtents = [&](int64_t group) {
    return llvm::map_to_vector(llvm::seq<int64_t>(0, rank), [&](int64_t dim) {
      return extract({group, dim});
    });
  };

  SmallVector<Value> fields = {extract({kAllocPtrIdx}),
                               extract({kAlignedPtrIdx}),
                               extract({kOffsetIdx})};
  llvm::append_range(fields, extractExtents(kSizesIdx));
  llvm::append_range(fields, extractExtents(kStridesIdx));
  return fields;
}

// Loads the packed descriptor at inputs[inputIndex].
//   %slot = llvm.getelementptr %inputs[%i]
//   %ptr  = llvm.load %slot
//   %desc = llvm.load %ptr
Value loadDescriptor(OpBuilder &builder, Location loc, Value inputs,
                     int64_t inputIndex, Type packedType) {
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Type i64Type = builder.getI64Type();
  Value slotIndex = LLVM::ConstantOp::create(
      builder, loc, i64Type, builder.getI64IntegerAttr(inputIndex));
  Value slot = LLVM::GEPOp::create(builder, loc, ptrType, ptrType, inputs,
                                   ValueRange{slotIndex});
  Value descriptorPointer = LLVM::LoadOp::create(builder, loc, ptrType, slot);
  return LLVM::LoadOp::create(builder, loc, packedType, descriptorPointer);
}

// Renames the body and writes the wrapper that inference_compute calls.
// Before: llvm.func @main_graph(%state, %alloc, %aligned, %off, %sz, %st)
// After:  llvm.func @main_graph_internal(...)
//         llvm.func @main_graph(%state, %inputs) -> i32
void writeMainGraphWrapper(LLVM::LLVMFuncOp graph,
                           ArrayRef<int64_t> inputRanks) {
  LLVM::LLVMFunctionType graphType = graph.getFunctionType();
  OpBuilder builder(graph->getContext());
  Location loc = graph.getLoc();
  Type ptrType = LLVM::LLVMPointerType::get(builder.getContext());
  Type i32Type = builder.getI32Type();

  graph.setName("main_graph_internal");
  graph.setLinkage(LLVM::Linkage::Private);

  builder.setInsertionPoint(graph);
  auto wrapper = LLVM::LLVMFuncOp::create(
      builder, loc, "main_graph",
      LLVM::LLVMFunctionType::get(i32Type, {ptrType, ptrType}));
  wrapper.setLinkage(LLVM::Linkage::Private);
  wrapper->setAttr("passthrough",
                   builder.getArrayAttr({builder.getStringAttr("noinline")}));

  Block *entry = wrapper.addEntryBlock(builder);
  builder.setInsertionPointToStart(entry);
  Value state = entry->getArgument(0);
  Value inputs = entry->getArgument(1);

  SmallVector<Value> graphArguments{state};
  int64_t parameterIndex = 1;
  for (auto [inputIndex, rank] : llvm::enumerate(inputRanks)) {
    auto declaredPointer =
        cast<LLVM::LLVMPointerType>(graphType.getParamType(parameterIndex));
    parameterIndex += scalarParamsForRank(rank);
    Value descriptor = loadDescriptor(
        builder, loc, inputs, inputIndex,
        descriptorType(builder, rank, declaredPointer.getAddressSpace()));
    llvm::append_range(graphArguments,
                       unpackDescriptor(builder, loc, descriptor, rank));
  }

  LLVM::CallOp::create(builder, loc, graph, graphArguments);
  Value success = LLVM::ConstantOp::create(builder, loc, i32Type,
                                           builder.getI32IntegerAttr(0));
  LLVM::ReturnOp::create(builder, loc, success);
}

struct MainGraphAbiPass : impl::MainGraphAbiPassBase<MainGraphAbiPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto graph = module.lookupSymbol<LLVM::LLVMFuncOp>("main_graph");
    if (!graph) {
      return;
    }

    auto inputShapes = module->getAttrOfType<ArrayAttr>("hipdnn.input_shapes");
    if (!inputShapes) {
      graph.emitError("hipdnn.input_shapes is required to wrap @main_graph");
      signalPassFailure();
      return;
    }

    FailureOr<SmallVector<int64_t>> inputRanks =
        collectInputRanks(inputShapes, graph);
    if (failed(inputRanks)) {
      signalPassFailure();
      return;
    }

    int64_t expectedParams = scalarParamCount(*inputRanks);
    LLVM::LLVMFunctionType graphType = graph.getFunctionType();
    if (static_cast<int64_t>(graphType.getNumParams()) != expectedParams) {
      graph.emitError("@main_graph takes ")
          << graphType.getNumParams() << " parameters, but "
          << "hipdnn.input_shapes describes " << expectedParams;
      signalPassFailure();
      return;
    }

    writeMainGraphWrapper(graph, *inputRanks);
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
