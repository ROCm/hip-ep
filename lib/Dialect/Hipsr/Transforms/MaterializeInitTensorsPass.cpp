/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- MaterializeInitTensorsPass.cpp - Materialize placeholder inits -----===//
//
// Rewrites each pool domain into the virtual 3-region form: every
// hipsr.placeholder shape region becomes an scf.execute_region yielding
// !shape.shape, every placeholder result becomes a tensor.empty built from
// that shape, and the data ops keep their order.
//
// Before:
//   %init = hipsr.placeholder(%ctx) ins(%a, %b) : tensor<?x512xf16>
//       shape_region { ^bb0(%a_shape: !shape.shape, %b_shape: !shape.shape):
//         hipsr.shape_yield2 %result_shape : !shape.shape }
//   %0 = hipsr.matmul(%ctx) ins(%a, %b) outs(%init) : tensor<?x512xf16>
// After:
//   %shape = scf.execute_region -> !shape.shape { ... }
//   %init = tensor.empty(%d0) : tensor<?x512xf16>
//   %0 = hipsr.matmul(%ctx) ins(%a, %b) outs(%init) : tensor<?x512xf16>
//
// The phases run per domain: collect and group the placeholders, build the
// shape regions, build the tensor allocations, then replace and erase.
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/Hipsr/Transforms/Passes.h"

#include "hip/Dialect/Hipsr/IR/HipsrDialect.h"
#include "hip/Dialect/Hipsr/IR/HipsrOps.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Shape/IR/Shape.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Visitors.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallVector.h"

namespace mlir {
namespace hipsr {

#define GEN_PASS_DEF_MATERIALIZEINITTENSORSPASS
#include "hip/Dialect/Hipsr/Transforms/Passes.h.inc"

namespace {

// Collects a domain's placeholders and moves them to the front of the domain
// block. SSA form already orders them topologically, so grouping only has to
// keep their relative order.
FailureOr<SmallVector<PlaceholderOp>>
collectAndGroupPlaceholders(PoolDomainOp poolDomain) {
  Block &domainBlock = poolDomain.getBody().front();

  SmallVector<PlaceholderOp> placeholders;
  for (Operation &operation : domainBlock) {
    auto placeholder = dyn_cast<PlaceholderOp>(&operation);
    if (!placeholder) {
      continue;
    }
    if (placeholder.getShapeRegion().empty()) {
      placeholder.emitOpError(
          "shape region must be populated by -hipsr-populate-shape-region");
      return failure();
    }
    placeholders.push_back(placeholder);
  }

  // Placeholders already sitting at the front are skipped rather than moved:
  // splicing an operation before itself corrupts the block's operation list.
  Block::iterator insertionPoint = domainBlock.begin();
  for (PlaceholderOp placeholder : placeholders) {
    if (&*insertionPoint == placeholder.getOperation()) {
      ++insertionPoint;
      continue;
    }
    placeholder->moveBefore(&domainBlock, insertionPoint);
  }

  return placeholders;
}

// Returns the shape a placeholder input contributes to the shape graph. An
// input produced by another placeholder reads that placeholder's shape from its
// scf.execute_region, because the placeholder itself is erased later; any other
// input is a value that outlives the pass, so its shape is taken directly.
Value getShapeForInput(
    Value input, Location loc,
    const DenseMap<PlaceholderOp, scf::ExecuteRegionOp>
        &placeholderToExecuteRegion,
    OpBuilder &builder) {
  if (auto producer = input.getDefiningOp<PlaceholderOp>()) {
    scf::ExecuteRegionOp executeRegion =
        placeholderToExecuteRegion.lookup(producer);
    if (!executeRegion) {
      return nullptr;
    }
    return executeRegion.getResult(cast<OpResult>(input).getResultNumber());
  }
  return builder.create<shape::ShapeOfOp>(
      loc, shape::ShapeType::get(builder.getContext()), input);
}

// Moves a placeholder's shape region body into a new scf.execute_region placed
// just before it. hipsr.shape_yield2 is bound to hipsr.placeholder by HasParent
// and cannot come along, so the transferred body gets the scf terminator.
scf::ExecuteRegionOp
createExecuteRegionAndTransferBody(PlaceholderOp placeholder,
                                   OpBuilder &builder) {
  Location loc = placeholder.getLoc();
  SmallVector<Type> shapeTypes(placeholder.getNumResults(),
                               shape::ShapeType::get(builder.getContext()));

  builder.setInsertionPoint(placeholder);
  auto executeRegion = builder.create<scf::ExecuteRegionOp>(loc, shapeTypes);
  executeRegion.getRegion().takeBody(placeholder.getShapeRegion());

  auto shapeYield =
      cast<ShapeYield2Op>(executeRegion.getRegion().front().getTerminator());
  builder.setInsertionPoint(shapeYield);
  builder.create<scf::YieldOp>(shapeYield.getLoc(), shapeYield.getShapes());
  shapeYield.erase();

  return executeRegion;
}

// Rewrites a transferred body to read from the enclosing domain instead of the
// shape region's own block arguments, then drops those arguments: an
// scf.execute_region region takes none.
LogicalResult replaceShapeRegionArguments(
    PlaceholderOp placeholder, Block &shapeBlock,
    const DenseMap<PlaceholderOp, scf::ExecuteRegionOp>
        &placeholderToExecuteRegion,
    OpBuilder &builder) {
  ValueRange inputs = placeholder.getInputs();
  if (shapeBlock.getNumArguments() != inputs.size()) {
    return placeholder.emitOpError("shape region takes ")
           << shapeBlock.getNumArguments()
           << " arguments but the placeholder has " << inputs.size()
           << " inputs";
  }

  // The shapes go at the top of the body so they dominate every use of the
  // argument they replace.
  builder.setInsertionPointToStart(&shapeBlock);

  for (auto [argument, input] :
       llvm::zip_equal(shapeBlock.getArguments(), inputs)) {
    Value shape = getShapeForInput(input, placeholder.getLoc(),
                                   placeholderToExecuteRegion, builder);
    if (!shape) {
      return placeholder.emitOpError(
          "input has no shape computation; the producing placeholder was not "
          "materialized first");
    }
    argument.replaceAllUsesWith(shape);
  }

  shapeBlock.eraseArguments(0, shapeBlock.getNumArguments());
  return success();
}

// Turns every placeholder shape region into an scf.execute_region yielding one
// !shape.shape per placeholder result, and maps each placeholder to it so later
// placeholders can consume the shapes.
FailureOr<DenseMap<PlaceholderOp, scf::ExecuteRegionOp>>
createShapeComputations(ArrayRef<PlaceholderOp> placeholders,
                        OpBuilder &builder) {
  for (PlaceholderOp placeholder : placeholders) {
    if (placeholder.getPlaceholderType() != PlaceholderType::Normal) {
      placeholder.emitOpError("barrier placeholders are not materialized yet");
      return failure();
    }
  }

  DenseMap<PlaceholderOp, scf::ExecuteRegionOp> placeholderToExecuteRegion;
  for (PlaceholderOp placeholder : placeholders) {
    scf::ExecuteRegionOp executeRegion =
        createExecuteRegionAndTransferBody(placeholder, builder);
    if (failed(replaceShapeRegionArguments(placeholder,
                                           executeRegion.getRegion().front(),
                                           placeholderToExecuteRegion,
                                           builder))) {
      return failure();
    }
    placeholderToExecuteRegion[placeholder] = executeRegion;
  }

  return placeholderToExecuteRegion;
}

// Reads the dynamic extents of a result out of its computed shape. Static
// dimensions are already carried by the result type, so only the dynamic ones
// need an SSA value, converted to index for tensor.empty.
SmallVector<Value> extractDynamicDimensions(Value shapeValue,
                                            RankedTensorType tensorType,
                                            Location loc, OpBuilder &builder) {
  SmallVector<Value> dynamicDimensions;
  for (int64_t dimension : llvm::seq<int64_t>(0, tensorType.getRank())) {
    if (!tensorType.isDynamicDim(dimension)) {
      continue;
    }
    Value extent =
        builder.create<shape::GetExtentOp>(loc, shapeValue, dimension);
    dynamicDimensions.push_back(
        builder.create<shape::SizeToIndexOp>(loc, extent));
  }
  return dynamicDimensions;
}

Value createTensorEmptyFromShape(Value placeholderResult, Value shapeValue,
                                 Location loc, OpBuilder &builder) {
  auto tensorType = cast<RankedTensorType>(placeholderResult.getType());
  SmallVector<Value> dynamicDimensions =
      extractDynamicDimensions(shapeValue, tensorType, loc, builder);
  return builder.create<tensor::EmptyOp>(loc, tensorType, dynamicDimensions);
}

// Materializes one tensor.empty per placeholder result. They all go after the
// last shape computation, so once the placeholders are erased the domain reads
// as shape computations, then allocations, then data operations.
DenseMap<Value, Value> createTensorAllocs(
    ArrayRef<PlaceholderOp> placeholders,
    const DenseMap<PlaceholderOp, scf::ExecuteRegionOp>
        &placeholderToExecuteRegion,
    OpBuilder &builder) {
  builder.setInsertionPointAfter(
      placeholderToExecuteRegion.lookup(placeholders.back()));

  DenseMap<Value, Value> placeholderResultToInitTensor;
  for (PlaceholderOp placeholder : placeholders) {
    scf::ExecuteRegionOp executeRegion =
        placeholderToExecuteRegion.lookup(placeholder);
    for (auto [result, shapeValue] : llvm::zip_equal(
             placeholder.getResults(), executeRegion.getResults())) {
      placeholderResultToInitTensor[result] = createTensorEmptyFromShape(
          result, shapeValue, placeholder.getLoc(), builder);
    }
  }
  return placeholderResultToInitTensor;
}

// Hands every placeholder result over to the tensor.empty that stands in for
// it and drops the placeholder. A placeholder result can feed a later
// placeholder's inputs, which is why the erase cannot come before the
// replacement, and why every placeholder in the domain has to go in the same
// run: an input that now reads a tensor.empty is not a legal shape-graph input.
void replaceAndCleanup(
    ArrayRef<PlaceholderOp> placeholders,
    const DenseMap<Value, Value> &placeholderResultToInitTensor) {
  for (PlaceholderOp placeholder : placeholders) {
    for (Value result : placeholder.getResults()) {
      result.replaceAllUsesWith(placeholderResultToInitTensor.lookup(result));
    }
    placeholder.erase();
  }
}

LogicalResult materializePoolDomain(PoolDomainOp poolDomain) {
  FailureOr<SmallVector<PlaceholderOp>> placeholders =
      collectAndGroupPlaceholders(poolDomain);
  if (failed(placeholders)) {
    return failure();
  }
  if (placeholders->empty()) {
    return success();
  }

  OpBuilder builder(poolDomain.getContext());
  FailureOr<DenseMap<PlaceholderOp, scf::ExecuteRegionOp>>
      placeholderToExecuteRegion =
          createShapeComputations(*placeholders, builder);
  if (failed(placeholderToExecuteRegion)) {
    return failure();
  }

  DenseMap<Value, Value> placeholderResultToInitTensor =
      createTensorAllocs(*placeholders, *placeholderToExecuteRegion, builder);
  replaceAndCleanup(*placeholders, placeholderResultToInitTensor);
  return success();
}

struct MaterializeInitTensorsPass
    : impl::MaterializeInitTensorsPassBase<MaterializeInitTensorsPass> {
  void runOnOperation() override {
    WalkResult walkResult = getOperation().walk([](PoolDomainOp poolDomain) {
      if (failed(materializePoolDomain(poolDomain))) {
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (walkResult.wasInterrupted()) {
      signalPassFailure();
    }
  }
};

} // namespace

} // namespace hipsr
} // namespace mlir
