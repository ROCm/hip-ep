/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- RecordGraphIOMetadataPass.cpp - Record graph I/O metadata ----------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/Support/MathExtras.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_RECORDGRAPHIOMETADATAPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

struct GraphIOMetadata {
  SmallVector<Attribute> shapes;
  SmallVector<int64_t> elementSizes;
};

// Dynamic extents become -1 so the runtime can store them as i64.
//   memref<?x4xf16>  ->  array<i64: -1, 4>
Attribute shapeMetadata(Builder &builder, ShapedType shapedType) {
  SmallVector<int64_t> shape =
      llvm::map_to_vector(shapedType.getShape(), [](int64_t extent) {
        return ShapedType::isDynamic(extent) ? int64_t{-1} : extent;
      });
  return builder.getDenseI64ArrayAttr(shape);
}

// Round up so a sub-byte element still fills one runtime byte.
//   i1 -> 1,  f16 -> 2
int64_t elementByteSize(ShapedType shapedType) {
  return llvm::divideCeil(shapedType.getElementType().getIntOrFloatBitWidth(),
                          8);
}

// Skips !hipsr.context. Fails if a type is unranked or not int/float.
//   (!hipsr.context, memref<?x4xf16>)  ->  shapes [-1, 4], sizes [2]
FailureOr<GraphIOMetadata> collectGraphIOMetadata(Builder &builder,
                                                  TypeRange types,
                                                  func::FuncOp graph,
                                                  StringRef kind) {
  SmallVector<Type> tensorTypes = llvm::to_vector(llvm::make_filter_range(
      types, [](Type type) { return !isa<ContextType>(type); }));

  auto unranked = llvm::find_if(tensorTypes, [](Type type) {
    auto shapedType = dyn_cast<ShapedType>(type);
    return !shapedType || !shapedType.hasRank();
  });
  if (unranked != tensorTypes.end()) {
    graph.emitError("expected ranked ")
        << kind << " type in @main_graph, got " << *unranked;
    return failure();
  }

  auto unsupported = llvm::find_if(tensorTypes, [](Type type) {
    return !cast<ShapedType>(type).getElementType().isIntOrFloat();
  });
  if (unsupported != tensorTypes.end()) {
    graph.emitError("unsupported element type in @main_graph ")
        << kind << ": " << cast<ShapedType>(*unsupported).getElementType();
    return failure();
  }

  auto shapedTypes = llvm::map_range(
      tensorTypes, [](Type type) { return cast<ShapedType>(type); });
  return GraphIOMetadata{
      llvm::map_to_vector(
          shapedTypes,
          [&](ShapedType type) { return shapeMetadata(builder, type); }),
      llvm::map_to_vector(shapedTypes, elementByteSize),
  };
}

// Stamps the hipdnn.* attributes that generate-interface reads.
void writeGraphIOMetadata(ModuleOp module, Builder &builder,
                          const GraphIOMetadata &inputs,
                          const GraphIOMetadata &outputs) {
  module->setAttr("hipdnn.input_count",
                  builder.getI64IntegerAttr(inputs.shapes.size()));
  module->setAttr("hipdnn.input_shapes", builder.getArrayAttr(inputs.shapes));
  module->setAttr("hipdnn.input_element_sizes",
                  builder.getDenseI64ArrayAttr(inputs.elementSizes));
  module->setAttr("hipdnn.output_count",
                  builder.getI64IntegerAttr(outputs.shapes.size()));
  module->setAttr("hipdnn.output_shapes", builder.getArrayAttr(outputs.shapes));
  module->setAttr("hipdnn.output_element_sizes",
                  builder.getDenseI64ArrayAttr(outputs.elementSizes));
}

struct RecordGraphIOMetadataPass
    : impl::RecordGraphIOMetadataPassBase<RecordGraphIOMetadataPass> {
  void runOnOperation() override {
    ModuleOp module = getOperation();
    auto graph = module.lookupSymbol<func::FuncOp>("main_graph");
    if (!graph) {
      return;
    }

    Builder builder(module.getContext());
    FunctionType functionType = graph.getFunctionType();
    FailureOr<GraphIOMetadata> inputs = collectGraphIOMetadata(
        builder, functionType.getInputs(), graph, "input");
    FailureOr<GraphIOMetadata> outputs = collectGraphIOMetadata(
        builder, functionType.getResults(), graph, "output");
    if (failed(inputs) || failed(outputs)) {
      signalPassFailure();
      return;
    }

    writeGraphIOMetadata(module, builder, *inputs, *outputs);
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
