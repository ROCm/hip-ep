/*
 * Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
 * Licensed under the MIT License.
 */
//===- HipResultTypeInferenceImpl.cpp -------------------------------------===//
//
// Per-op `InferTypeOpInterface::inferReturnTypes` impls for HIP DPS ops.
//
// Every Hip_DpsOp's result types equal the types of its `outs` operand(s)
// (DPS post-tensor-bufferize convention). Memref-mode ops produce zero
// result types — the destination operand carries the writes via its
// memref descriptor and the SSA result list is empty.
//
// One section per op below, mirroring `HipReifyResultShapesImpl.cpp`.
// `LoopOp::inferReturnTypes` lives in `HipDialect.cpp` because it sits
// next to that op's `verify` / `getEffects` / `getDpsInitsMutable`
// peers (control-flow op, not a DPS compute op).
//
//===----------------------------------------------------------------------===//

#include "hip/Dialect/IR/HipDialect.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::hip;

namespace {

// Workhorse: append `out`'s type to `results` if `out` is a ranked tensor.
// Memref-typed `out` produces no result type (post-bufferize DPS convention;
// matches the contract enforced by `verifyDpsComputeOp` in `HipDialect.cpp`).
//
// All five ops in this file have a single `$output` operand. If a future
// multi-result Hip_DpsOp lands its InferType impl here, it can call this
// helper once per output operand.
LogicalResult appendDpsResultIfTensor(Value out,
                                      SmallVectorImpl<Type> &results) {
  if (auto t = dyn_cast<RankedTensorType>(out.getType()))
    results.push_back(t);
  return success();
}

} // namespace

//===----------------------------------------------------------------------===//
// RopeOp
//===----------------------------------------------------------------------===//

LogicalResult RopeOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> /*location*/,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  RopeOpAdaptor adaptor(operands, attributes, properties, regions);
  return appendDpsResultIfTensor(adaptor.getOutput(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// RmsNormOp
//===----------------------------------------------------------------------===//

LogicalResult RmsNormOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> /*location*/,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  RmsNormOpAdaptor adaptor(operands, attributes, properties, regions);
  return appendDpsResultIfTensor(adaptor.getOutput(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// QMoEOp
//===----------------------------------------------------------------------===//

LogicalResult QMoEOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> /*location*/,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  QMoEOpAdaptor adaptor(operands, attributes, properties, regions);
  return appendDpsResultIfTensor(adaptor.getOutput(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// MatMulNBitsOp
//===----------------------------------------------------------------------===//

LogicalResult MatMulNBitsOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> /*location*/,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  MatMulNBitsOpAdaptor adaptor(operands, attributes, properties, regions);
  return appendDpsResultIfTensor(adaptor.getOutput(), inferredReturnTypes);
}

//===----------------------------------------------------------------------===//
// GemmOp
//===----------------------------------------------------------------------===//

LogicalResult GemmOp::inferReturnTypes(
    MLIRContext * /*context*/, std::optional<Location> /*location*/,
    ValueRange operands, DictionaryAttr attributes, OpaqueProperties properties,
    RegionRange regions, SmallVectorImpl<Type> &inferredReturnTypes) {
  GemmOpAdaptor adaptor(operands, attributes, properties, regions);
  return appendDpsResultIfTensor(adaptor.getOutput(), inferredReturnTypes);
}
