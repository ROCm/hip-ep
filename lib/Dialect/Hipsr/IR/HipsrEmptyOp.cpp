/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */

#include "hip/Dialect/Hipsr/IR/HipsrEmptyOp.h"

// Needs the full EmptyYieldOp type: the implicit-terminator trait's generated
// methods and verify() below both use it.
#include "hip/Dialect/Hipsr/IR/HipsrEmptyYieldOp.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h" // tensor::EmptyOp
#include "llvm/ADT/Sequence.h"

using namespace mlir;
using namespace mlir::hipsr;

LogicalResult EmptyOp::verify() {
  // The trait already guarantees a single block ending in EmptyYieldOp. Its
  // yielded tensors become this op's results, so just check count and types.
  auto yieldOp = cast<EmptyYieldOp>(getRegion().front().getTerminator());

  if (yieldOp.getTensors().size() != getNumResults()) {
    return emitOpError() << "has " << getNumResults()
                         << " result(s) but its empty_yield yields "
                         << yieldOp.getTensors().size() << " value(s)";
  }

  for (unsigned idx : llvm::seq<unsigned>(0, getNumResults())) {
    Type resultType = getResultTypes()[idx];
    Type yieldType = yieldOp.getTensors()[idx].getType();
    if (resultType != yieldType) {
      return emitOpError() << "result #" << idx << " type " << resultType
                           << " does not match the yielded value type "
                           << yieldType;
    }
  }

  return success();
}

SmallVector<SmallVector<Value>> EmptyOp::getShapes() {
  // Only valid once the region is populated with the tensor.empty producers.
  assert(!getRegion().empty() &&
         "region must be populated before calling getShapes");
  auto yieldOp = cast<EmptyYieldOp>(getRegion().front().getTerminator());
  SmallVector<SmallVector<Value>> dimsPerResult;
  for (Value t : yieldOp.getTensors()) {
    SmallVector<Value> dims;
    if (auto emptyTensor = t.getDefiningOp<tensor::EmptyOp>())
      dims = llvm::to_vector(emptyTensor.getDynamicSizes());
    dimsPerResult.push_back(std::move(dims));
  }
  return dimsPerResult;
}

SmallVector<RankedTensorType> EmptyOp::getTensorTypes() {
  SmallVector<RankedTensorType> types;
  for (Value result : getResults())
    types.push_back(cast<RankedTensorType>(result.getType()));
  return types;
}

#define GET_OP_CLASSES
#include "hip/Dialect/Hipsr/IR/HipsrEmptyOp.cpp.inc"
